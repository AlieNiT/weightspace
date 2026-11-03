"""Provider-neutral chat interface.

Messages use an OpenAI-like neutral shape:
  {"role": "system"|"user", "content": str}
  {"role": "assistant", "content": str|None, "tool_calls": [{"id", "name", "arguments"}]}
  {"role": "tool", "tool_call_id": str, "name": str, "content": str}
Tools are {"name", "description", "parameters": <JSON schema>}.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class LLMResponse:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    raw: Any = None
    extra: dict = field(default_factory=dict)  # provider state to replay (e.g. thinking blocks)

    def to_message(self) -> dict:
        msg = {"role": "assistant", "content": self.text, "tool_calls": [t.to_dict() for t in self.tool_calls]}
        msg.update(self.extra)
        return msg


THINKING_LEVELS = ("off", "low", "medium", "high", "xhigh", "max")


def normalize_thinking(value) -> str | None:
    """None = provider default; False/"off" = no reasoning; True = "medium"; or "low"/"medium"/"high"."""
    if value is None:
        return None
    if value is False:
        return "off"
    if value is True:
        return "medium"
    v = str(value).lower()
    if v in ("none", "disabled", "false", "no"):
        return "off"
    if v in ("on", "true", "enabled"):
        return "medium"
    if v not in THINKING_LEVELS:
        raise ValueError(f"thinking must be None, a bool, or one of {THINKING_LEVELS}")
    return v


def normalize_speed(value) -> str | None:
    """None = provider default; True/"fast" = fast mode; False/"standard"; other tiers pass through (e.g. "ultrafast")."""
    if value is None:
        return None
    if value is True:
        return "fast"
    if value is False:
        return "standard"
    return str(value).lower()


class LLM:
    """Implement `complete` to plug in any provider.

    `thinking` and `speed` are set by `Session(thinking=..., speed=...)`; backends map them to their own
    reasoning switch and fast-inference option.
    """

    name = "llm"
    thinking: str | None = None
    speed: str | None = None

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        raise NotImplementedError
