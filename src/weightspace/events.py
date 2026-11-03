"""Run events and the callbacks that observe them (console, logging, transcript files, OpenTelemetry).

A `Session` emits one `Event` for each thing that happens in a run:

  run_start   {mode, llm, task, tools, forward_budget}
  step_start  {}
  llm_start   {messages, tools}
  llm_end     {seconds, usage, stop_reason, tool_calls, error?}
  tool_start  {call_id, name, arguments}
  tool_end    {call_id, name, seconds, output_chars, error, cost?}
  message     {message}                       every message appended to the transcript
  run_end     {stop_reason, steps, seconds, usage, cost, answer, error?}

A callback is any `fn(event)` or an object with `on_event(event)`. Callbacks run synchronously and may be called
from worker threads when tool calls run in parallel, so keep them quick and thread-safe. Exceptions in a callback
are logged and never break the run.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, TextIO, runtime_checkable

logger = logging.getLogger("weightspace")


@dataclass(frozen=True)
class Event:
    type: str
    run_id: str
    step: int
    data: dict[str, Any] = field(default_factory=dict)
    time: float = field(default_factory=time.time)


@runtime_checkable
class Callback(Protocol):
    def on_event(self, event: Event) -> None: ...


CallbackLike = Callback | Callable[[Event], None]


class Emitter:
    """Fans events out to callbacks; never lets a failing callback break the run."""

    def __init__(self, callbacks: Iterable[CallbackLike] = ()):
        self._fns: list[Callable[[Event], None]] = []
        for cb in callbacks:
            self.add(cb)

    def add(self, cb: CallbackLike) -> None:
        self._fns.append(cb.on_event if isinstance(cb, Callback) else cb)

    def emit(self, type: str, run_id: str, step: int, **data) -> None:
        if not self._fns:
            return
        ev = Event(type, run_id, step, data)
        for fn in self._fns:
            try:
                fn(ev)
            except Exception:
                logger.exception("weightspace callback %r failed on %s", fn, type)


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + " ..."


class ConsoleLogger:
    """Human-readable live view of a run (what `Session(verbose=True)` uses)."""

    def __init__(self, stream: TextIO | None = None, max_chars: int = 2000):
        self.stream = stream
        self.max_chars = max_chars
        self._lock = threading.Lock()

    def _print(self, text: str) -> None:
        with self._lock:
            print(text, file=self.stream, flush=True)

    def on_event(self, ev: Event) -> None:
        d = ev.data
        if ev.type == "message":
            m = d["message"]
            role = m["role"]
            lines = [f"--- {role}{' ' + m.get('name', '') if role == 'tool' else ''}"]
            if m.get("content"):
                lines.append(_clip(str(m["content"]), self.max_chars))
            lines += [
                f">>> {c['name']}({_clip(json.dumps(c['arguments'], default=str), 1500)})" for c in m.get("tool_calls") or []
            ]
            self._print("\n".join(lines))
        elif ev.type == "llm_end":
            u = d.get("usage") or {}
            self._print(
                f"[step {ev.step}] llm {d['seconds']:.1f}s in={u.get('input_tokens', 0):,} "
                f"(cached {u.get('cached_input_tokens', 0):,}) out={u.get('output_tokens', 0):,}"
                + (f" error={d['error']}" if d.get("error") else "")
            )
        elif ev.type == "run_end":
            cost = f" ${d['cost']:.4f}" if d.get("cost") is not None else ""
            self._print(f"[run {ev.run_id}] {d['stop_reason']} after {d['steps']} steps, {d['seconds']:.1f}s{cost}")


class LoggingCallback:
    """Sends events to the `weightspace` logger: run/step milestones at INFO, everything else at DEBUG."""

    INFO = {"run_start", "run_end", "llm_end"}

    def __init__(self, logger_: logging.Logger | None = None):
        self.log = logger_ or logger

    def on_event(self, ev: Event) -> None:
        if ev.type == "message":
            return
        level = logging.INFO if ev.type in self.INFO else logging.DEBUG
        if self.log.isEnabledFor(level):
            data = {k: v for k, v in ev.data.items() if k not in ("arguments", "task", "answer")}
            self.log.log(level, "%s run=%s step=%d %s", ev.type, ev.run_id, ev.step, json.dumps(data, default=str))


class TranscriptWriter:
    """Appends every transcript message to a JSONL file as it happens (crash-safe, `Session(log_path=...)`)."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()

    def on_event(self, ev: Event) -> None:
        if ev.type != "message":
            return
        line = json.dumps(ev.data["message"], default=str) + "\n"
        with self._lock, open(self.path, "a") as f:
            f.write(line)


class EventLog:
    """Writes every event (not just messages) to a JSONL file, for dashboards and offline analysis."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()

    def on_event(self, ev: Event) -> None:
        rec = {"type": ev.type, "run_id": ev.run_id, "step": ev.step, "time": ev.time, **ev.data}
        with self._lock, open(self.path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
