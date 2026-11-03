from __future__ import annotations

import itertools
from collections.abc import Callable

from .base import LLM, LLMResponse, ToolCall

_ids = itertools.count()


def call(name: str, /, **arguments) -> ToolCall:
    return ToolCall(f"call_{next(_ids)}", name, arguments)


def reply(text: str | None = None, *calls: ToolCall) -> LLMResponse:
    return LLMResponse(text=text, tool_calls=list(calls))


class ScriptedLLM(LLM):
    """Replays a fixed script; each step is an LLMResponse or fn(messages, tools) -> LLMResponse.

    Useful for tests and for replaying logged runs without paying for tokens.
    """

    name = "scripted"

    def __init__(self, steps: list[LLMResponse | Callable[[list[dict], list[dict]], LLMResponse]]):
        self.steps = list(steps)
        self.seen: list[list[dict]] = []
        self.thinking_seen: list[str | None] = []

    def complete(self, messages, tools=None):
        self.seen.append([dict(m) for m in messages])
        self.thinking_seen.append(self.thinking)
        if not self.steps:
            return reply("(script exhausted)")
        step = self.steps.pop(0)
        return step(messages, tools or []) if callable(step) else step
