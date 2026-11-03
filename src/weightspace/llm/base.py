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


class LLM:
    """Implement `complete` to plug in any provider."""

    name = "llm"

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        raise NotImplementedError
