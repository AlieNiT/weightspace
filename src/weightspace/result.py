"""What a run returns and saves."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

from .space import DiffSummary, WeightSpace
from .usage import Usage

STOP_REASONS = ("submitted", "answered", "max_steps", "token_budget", "cost_budget", "time_budget", "interrupted", "error")


@dataclass
class ForwardCall:
    probe: str
    arguments: dict
    output: str
    cost: int
    seconds: float


@dataclass
class Result:
    answer: str
    confidence: float | None
    mode: str
    thinking: str | None
    diff: DiffSummary
    forward_calls: list[ForwardCall]
    forward_budget: int
    steps: int
    transcript: list[dict]
    usage: dict
    ws: WeightSpace = field(repr=False)
    speed: str | None = None
    run_id: str = ""
    llm: str = ""
    stop_reason: str = "submitted"
    seconds: float = 0.0
    tokens: Usage = field(default_factory=Usage)
    cost: float | None = None
    timings: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def forward_used(self) -> int:
        return sum(c.cost for c in self.forward_calls)

    @property
    def ok(self) -> bool:
        return self.stop_reason in ("submitted", "answered")

    def __str__(self) -> str:
        conf = f" (confidence {self.confidence:.2f})" if self.confidence is not None else ""
        think = f" thinking={self.thinking}" if self.thinking else ""
        think += f" speed={self.speed}" if self.speed else ""
        stop = "" if self.stop_reason == "submitted" else f" stop={self.stop_reason}"
        cost = f" cost=${self.cost:.4f}" if self.cost is not None else ""
        tokens = f" tokens={self.tokens.total_tokens:,}" if self.tokens.requests else ""
        return (
            f"[{self.mode}] {self.answer}{conf}\n"
            f"steps={self.steps}{think}{stop} forward passes={self.forward_used}/{self.forward_budget}{tokens}{cost}\n"
            f"{self.diff}"
        )

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "llm": self.llm,
            "mode": self.mode,
            "thinking": self.thinking,
            "speed": self.speed,
            "answer": self.answer,
            "confidence": self.confidence,
            "stop_reason": self.stop_reason,
            "error": self.error,
            "steps": self.steps,
            "seconds": round(self.seconds, 3),
            "forward_used": self.forward_used,
            "forward_budget": self.forward_budget,
            "forward_calls": [asdict(c) for c in self.forward_calls],
            "changed_params": self.diff.changed_params,
            "total_params": self.diff.total_params,
            "changed_fraction": self.diff.fraction,
            "tensors": [asdict(t) for t in self.diff.tensors],
            "tokens": self.tokens.to_dict(),
            "cost_usd": self.cost,
            "timings": self.timings,
            "usage": self.usage,
        }

    def save(self, directory: str) -> None:
        """Write `transcript.jsonl`, `result.json` and, if any weight changed, `patch.npz`."""
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "transcript.jsonl"), "w") as f:
            for m in self.transcript:
                f.write(json.dumps(m, default=str) + "\n")
        with open(os.path.join(directory, "result.json"), "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        if self.diff.changed_params:
            self.ws.save_patch(os.path.join(directory, "patch.npz"))
