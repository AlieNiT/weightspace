from __future__ import annotations

import json
import os

from .base import LLM, LLMResponse, ToolCall, normalize_thinking


def to_openai_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if m["role"] == "assistant":
            msg = {"role": "assistant", "content": m.get("content")}
            if m.get("tool_calls"):
                msg["tool_calls"] = [
                    {
                        "id": t["id"],
                        "type": "function",
                        "function": {"name": t["name"], "arguments": json.dumps(t["arguments"])},
                    }
                    for t in m["tool_calls"]
                ]
            out.append(msg)
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def to_openai_tools(tools: list[dict]) -> list[dict]:
    return [
        {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
        for t in tools
    ]


class OpenAIChat(LLM):
    """Any OpenAI-compatible Chat Completions endpoint (OpenAI, OpenRouter, vLLM, Ollama...).

    thinking_style says how `thinking` is sent:
      "reasoning_effort"  OpenAI reasoning models (off -> `off_effort`, e.g. "none" or "minimal")
      "openrouter"        OpenRouter's unified `reasoning` field
      "chat_template"     vLLM / SGLang hybrid models (Qwen3...): `chat_template_kwargs.enable_thinking`
      None                never send anything
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client=None,
        thinking=None,
        thinking_style: str | None = "reasoning_effort",
        off_effort: str = "none",
        **params,
    ):
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key or os.environ.get("OPENAI_API_KEY"),
                base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            )
        self.client = client
        self.model = model
        self.name = f"openai:{model}"
        self.params = params
        self.thinking = normalize_thinking(thinking)
        self.thinking_style = thinking_style
        self.off_effort = off_effort

    def _thinking_params(self, kw: dict) -> None:
        t, style = normalize_thinking(self.thinking), self.thinking_style
        if t is None or style is None:
            return
        extra = dict(kw.get("extra_body") or {})
        if style == "reasoning_effort":
            kw["reasoning_effort"] = self.off_effort if t == "off" else t
        elif style == "openrouter":
            extra["reasoning"] = {"enabled": False} if t == "off" else {"effort": t}
        elif style == "chat_template":
            ctk = dict(extra.get("chat_template_kwargs") or {})
            ctk["enable_thinking"] = t != "off"
            extra["chat_template_kwargs"] = ctk
        else:
            raise ValueError(f"unknown thinking_style {style!r}")
        if extra:
            kw["extra_body"] = extra

    def complete(self, messages, tools=None):
        kw = dict(self.params)
        self._thinking_params(kw)
        if tools:
            kw["tools"] = to_openai_tools(tools)
        r = self.client.chat.completions.create(model=self.model, messages=to_openai_messages(messages), **kw)
        msg = r.choices[0].message
        calls = []
        for t in msg.tool_calls or []:
            try:
                args = json.loads(t.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": t.function.arguments}
            calls.append(ToolCall(t.id, t.function.name, args))
        usage = r.usage.model_dump() if getattr(r, "usage", None) is not None else {}
        return LLMResponse(text=msg.content, tool_calls=calls, usage=usage, raw=r)


def to_responses_input(messages: list[dict]) -> tuple[str | None, list[dict]]:
    instructions, items = [], []
    for m in messages:
        if m["role"] == "system":
            instructions.append(m["content"])
        elif m["role"] == "assistant":
            if m.get("openai_output"):
                items.extend(m["openai_output"])
                continue
            if m.get("content"):
                items.append({"type": "message", "role": "assistant", "content": m["content"]})
            for t in m.get("tool_calls") or []:
                items.append(
                    {"type": "function_call", "call_id": t["id"], "name": t["name"], "arguments": json.dumps(t["arguments"])}
                )
        elif m["role"] == "tool":
            items.append({"type": "function_call_output", "call_id": m["tool_call_id"], "output": m["content"]})
        else:
            items.append({"type": "message", "role": m["role"], "content": m["content"]})
    return ("\n\n".join(instructions) or None), items


class OpenResponses(OpenAIChat):
    """Open Responses (https://www.openresponses.org), the open spec based on the OpenAI Responses API.

    Works with OpenAI and any compliant server via `base_url`. Needed by models that only allow
    reasoning together with tools there (e.g. gpt-6). Runs stateless (`store=False`) and replays
    encrypted reasoning items between turns.
    """

    def __init__(self, model: str, **kw):
        kw.setdefault("thinking_style", "reasoning_effort")
        super().__init__(model, **kw)
        self.name = f"openresponses:{model}"

    def complete(self, messages, tools=None):
        instructions, items = to_responses_input(messages)
        kw = {"store": False, "include": ["reasoning.encrypted_content"], **self.params}
        t = normalize_thinking(self.thinking)
        if t is not None and self.thinking_style is not None:
            kw["reasoning"] = {"effort": self.off_effort if t == "off" else t}
        if instructions:
            kw["instructions"] = instructions
        if tools:
            kw["tools"] = [
                {"type": "function", "name": t["name"], "description": t["description"], "parameters": t["parameters"]}
                for t in tools
            ]
        r = self.client.responses.create(model=self.model, input=items, **kw)
        output = [o.model_dump(exclude_none=True) if hasattr(o, "model_dump") else dict(o) for o in r.output]
        texts, calls = [], []
        for o in output:
            if o.get("type") == "message":
                texts += [c.get("text", "") for c in o.get("content", []) if c.get("type") == "output_text"]
            elif o.get("type") == "function_call":
                try:
                    args = json.loads(o.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": o.get("arguments")}
                calls.append(ToolCall(o["call_id"], o["name"], args))
        usage = r.usage.model_dump() if getattr(r, "usage", None) is not None else {}
        return LLMResponse(text="\n".join(texts) or None, tool_calls=calls, usage=usage, raw=r, extra={"openai_output": output})


OpenAIResponses = OpenResponses
