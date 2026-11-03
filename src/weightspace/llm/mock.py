from __future__ import annotations

import itertools
import json
from collections.abc import Callable, Iterable

from .base import LLM, LLMResponse, ToolCall

_ids = itertools.count()


def call(name: str, /, **arguments) -> ToolCall:
    return ToolCall(f"call_{next(_ids)}", name, arguments)


def reply(text: str | None = None, *calls: ToolCall) -> LLMResponse:
    return LLMResponse(text=text, tool_calls=list(calls))


class ScriptedLLM(LLM):
    """Replays a fixed script; each step is an LLMResponse or fn(messages, tools) -> LLMResponse.

    Useful for tests and for replaying logged runs without paying for tokens (see `from_transcript`).
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

    @classmethod
    def from_transcript(cls, transcript: str | Iterable[dict]) -> ScriptedLLM:
        """Replay the assistant turns of a logged run (`transcript.jsonl` or `Result.transcript`).

        Re-running the same task over the same weights and probes reproduces the run's tool calls, edits and
        answer offline. Tool outputs are recomputed, so a replay against different weights shows how the same
        program behaves there.
        """
        if isinstance(transcript, str):
            with open(transcript) as f:
                messages = [json.loads(line) for line in f if line.strip()]
        else:
            messages = list(transcript)
        steps: list[LLMResponse | Callable] = []
        for m in messages:
            if m.get("role") != "assistant":
                continue
            calls = [ToolCall(t["id"], t["name"], dict(t.get("arguments") or {})) for t in m.get("tool_calls") or []]
            extra = {k: v for k, v in m.items() if k not in ("role", "content", "tool_calls")}
            steps.append(LLMResponse(text=m.get("content"), tool_calls=calls, extra=extra))
        return cls(steps)
