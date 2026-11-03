from __future__ import annotations

import os
from typing import Any

from ..usage import Usage
from .base import LLM, LLMResponse, ToolCall, normalize_speed, normalize_thinking

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

    stream: send requests with streaming (recommended for long Opus turns: avoids idle-connection timeouts and
    the SDK's refusal of very large non-streaming `max_tokens`). "auto" streams when `max_tokens` > 21333.
    max_retries / timeout: passed to the SDK client (exponential backoff on 408/409/429/5xx and connection errors).
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
        speed=None,
        cache: bool = True,
        stream: bool | str = "auto",
        max_retries: int = 4,
        timeout: float | None = 900,
        **params,
    ):
        if client is None:
            from anthropic import Anthropic

            client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"), max_retries=max_retries, timeout=timeout)
        self.stream = stream
        self.client = client
        self.model = model
        self.name = f"anthropic:{model}"
        self.max_tokens = max_tokens
        self.params = params
        self.thinking = normalize_thinking(thinking)
        self.speed = normalize_speed(speed)
        if thinking_style not in ("auto", "adaptive", "budget"):
            raise ValueError("thinking_style must be 'auto', 'adaptive' or 'budget'")
        self.thinking_style = thinking_style
        self.cache = cache
        # models with always-on thinking (e.g. Opus 5.5) reject "disabled"; "off" then means lowest effort
        self.off_style = "disabled"

    def _thinking_kwargs(self, style: str) -> tuple[dict, int]:
        t = normalize_thinking(self.thinking)
        if t is None:
            return {}, self.max_tokens
        if t == "off":
            if self.off_style == "low_effort":
                return {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}}, self.max_tokens
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
        if self.speed:
            kw["speed"] = self.speed
        if system:
            kw["system"] = system
        if tools:
            kw["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools]
        style = "adaptive" if self.thinking_style == "auto" else self.thinking_style
        tk, max_tokens = self._thinking_kwargs(style)
        try:
            r = self._create(model=self.model, max_tokens=max_tokens, messages=msgs, **kw, **tk)
        except Exception as e:
            if tk.get("thinking") == {"type": "disabled"} and "disabled" in str(e):
                self.off_style = "low_effort"
            elif self.thinking_style == "auto" and "thinking" in tk and "adaptive" in str(e):
                self.thinking_style = "budget"
                style = "budget"
            else:
                raise
            tk, max_tokens = self._thinking_kwargs(style)
            r = self._create(model=self.model, max_tokens=max_tokens, messages=msgs, **kw, **tk)
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
        thinking_tokens = (usage.get("output_tokens_details") or {}).get("thinking_tokens")
        if thinking_tokens is not None:
            usage["thinking_tokens"] = thinking_tokens
        return LLMResponse(
            text="\n".join(text) or None,
            tool_calls=calls,
            usage=usage,
            raw=r,
            extra={"anthropic_content": blocks},
            tokens=Usage.from_anthropic(usage),
            stop_reason=getattr(r, "stop_reason", None),
        )

    def _create(self, **kw):
        stream = self.stream
        if stream == "auto":
            stream = kw["max_tokens"] > 21333
        if stream:
            with self.client.messages.stream(**kw) as s:
                return s.get_final_message()
        return self.client.messages.create(**kw)
