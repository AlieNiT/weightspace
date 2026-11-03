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
