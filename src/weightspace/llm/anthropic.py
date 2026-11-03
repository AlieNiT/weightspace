from __future__ import annotations

import os
from typing import Any

from .base import LLM, LLMResponse, ToolCall, normalize_thinking

THINKING_BUDGET = {"low": 2048, "medium": 8192, "high": 16384, "xhigh": 24576, "max": 32000}


def to_anthropic(messages: list[dict]) -> tuple[str, list[dict]]:
    system: list = []
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m["role"]
        if role == "system":
            system.append(m["content"])
        elif role == "user":
            out.append({"role": "user", "content": [{"type": "text", "text": m["content"]}]})
        elif role == "assistant":
            if m.get("anthropic_content"):
                blocks = m["anthropic_content"]
            else:
                blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                blocks += [
                    {"type": "tool_use", "id": t["id"], "name": t["name"], "input": t["arguments"]}
                    for t in m.get("tool_calls") or []
                ]
            out.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
            if out and out[-1]["role"] == "user" and all(b["type"] == "tool_result" for b in out[-1]["content"]):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    merged: list[dict] = []
    for m in out:
        if merged and merged[-1]["role"] == m["role"] == "user":
            merged[-1]["content"].extend(m["content"])
        else:
            merged.append(m)
    return "\n\n".join(system), merged


class AnthropicChat(LLM):
    """Anthropic Messages API.

    thinking_style: "adaptive" (thinking.type=adaptive + output_config.effort, newer models), "budget"
    (thinking.type=enabled + budget_tokens, older models) or "auto" (adaptive, falling back to budget if the
    model rejects it).
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        max_tokens: int = 16000,
        client=None,
        thinking=None,
        thinking_style: str = "auto",
        cache: bool = True,
        **params,
    ):
        if client is None:
            from anthropic import Anthropic

            client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.client = client
        self.model = model
        self.name = f"anthropic:{model}"
        self.max_tokens = max_tokens
        self.params = params
        self.thinking = normalize_thinking(thinking)
        if thinking_style not in ("auto", "adaptive", "budget"):
            raise ValueError("thinking_style must be 'auto', 'adaptive' or 'budget'")
        self.thinking_style = thinking_style
        self.cache = cache

    def _thinking_kwargs(self, style: str) -> tuple[dict, int]:
        t = normalize_thinking(self.thinking)
        if t is None:
            return {}, self.max_tokens
        if t == "off":
            return {"thinking": {"type": "disabled"}}, self.max_tokens
        if style == "budget":
            budget = THINKING_BUDGET[t]
            return {"thinking": {"type": "enabled", "budget_tokens": budget}}, max(self.max_tokens, budget + 4096)
        return {"thinking": {"type": "adaptive"}, "output_config": {"effort": t}}, self.max_tokens

    def complete(self, messages, tools=None):
        system, msgs = to_anthropic(messages)
        if self.cache and msgs and msgs[-1]["content"]:
            # prompt caching: everything up to the newest block is reused on the next step
            msgs[-1]["content"][-1] = {**msgs[-1]["content"][-1], "cache_control": {"type": "ephemeral"}}
        kw = dict(self.params)
        if system:
            kw["system"] = system
        if tools:
            kw["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools]
        style = "adaptive" if self.thinking_style == "auto" else self.thinking_style
        tk, max_tokens = self._thinking_kwargs(style)
        try:
            r = self.client.messages.create(model=self.model, max_tokens=max_tokens, messages=msgs, **kw, **tk)
        except Exception as e:
            if self.thinking_style != "auto" or "thinking" not in tk or "adaptive" not in str(e):
                raise
            self.thinking_style = "budget"
            tk, max_tokens = self._thinking_kwargs("budget")
            r = self.client.messages.create(model=self.model, max_tokens=max_tokens, messages=msgs, **kw, **tk)
        if self.thinking_style == "auto":
            self.thinking_style = style
        text, calls, blocks = [], [], []
        for b in r.content:
            blocks.append(b.model_dump(exclude_none=True) if hasattr(b, "model_dump") else dict(b))
            if b.type == "text":
                text.append(b.text)
            elif b.type == "tool_use":
                calls.append(ToolCall(b.id, b.name, dict(b.input)))
        usage = r.usage.model_dump() if getattr(r, "usage", None) is not None else {}
        return LLMResponse(
            text="\n".join(text) or None,
            tool_calls=calls,
            usage=usage,
            raw=r,
            extra={"anthropic_content": blocks},
        )
