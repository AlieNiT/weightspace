"""OpenTelemetry tracing for runs, following the GenAI semantic conventions.

    from weightspace.otel import OpenTelemetryCallback
    wsp.fix(model, "...", llm=..., callbacks=[OpenTelemetryCallback()])

One `invoke_agent weightspace` span per run, with a `chat <model>` span per LLM request and an
`execute_tool <name>` span per tool call underneath. Needs `pip install "weightspace[otel]"` and an SDK/exporter
configured by your application (spans are no-ops otherwise).
"""

from __future__ import annotations

import json
import threading
from typing import Any

from .events import Event


def _provider(llm_name: str) -> str:
    p = llm_name.split(":", 1)[0] if ":" in llm_name else "openai"
    return {"openresponses": "openai", "openai-responses": "openai"}.get(p, p)


class OpenTelemetryCallback:
    def __init__(self, tracer=None, *, capture_content: bool = False):
        from opentelemetry import trace

        self._trace = trace
        self.tracer = tracer or trace.get_tracer("weightspace")
        self.capture_content = capture_content
        self._spans: dict[Any, Any] = {}
        self._llm: dict[str, str] = {}
        self._lock = threading.Lock()

    def _start(self, key, name: str, parent_key, attrs: dict) -> None:
        with self._lock:
            parent = self._spans.get(parent_key)
        ctx = self._trace.set_span_in_context(parent) if parent is not None else None
        span = self.tracer.start_span(name, context=ctx, attributes={k: v for k, v in attrs.items() if v is not None})
        with self._lock:
            self._spans[key] = span

    def _end(self, key, attrs: dict, error: str | None = None) -> None:
        with self._lock:
            span = self._spans.pop(key, None)
        if span is None:
            return
        for k, v in attrs.items():
            if v is not None:
                span.set_attribute(k, v)
        if error:
            span.set_status(self._trace.Status(self._trace.StatusCode.ERROR, error))
        span.end()

    def on_event(self, ev: Event) -> None:
        d, run = ev.data, ev.run_id
        if ev.type == "run_start":
            self._llm[run] = d["llm"]
            self._start(
                (run,),
                "invoke_agent weightspace",
                None,
                {
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.agent.name": "weightspace",
                    "gen_ai.conversation.id": run,
                    "gen_ai.request.model": d["llm"].split(":", 1)[-1],
                    "gen_ai.provider.name": _provider(d["llm"]),
                    "weightspace.mode": d["mode"],
                    "weightspace.forward_budget": d["forward_budget"],
                },
            )
        elif ev.type == "llm_start":
            llm = self._llm.get(run, "")
            self._start(
                (run, "llm", ev.step),
                f"chat {llm.split(':', 1)[-1]}",
                (run,),
                {
                    "gen_ai.operation.name": "chat",
                    "gen_ai.provider.name": _provider(llm),
                    "gen_ai.request.model": llm.split(":", 1)[-1],
                    "gen_ai.conversation.id": run,
                    "weightspace.step": ev.step,
                },
            )
        elif ev.type == "llm_end":
            u = d.get("usage") or {}
            self._end(
                (run, "llm", ev.step),
                {
                    "gen_ai.usage.input_tokens": u.get("input_tokens"),
                    "gen_ai.usage.output_tokens": u.get("output_tokens"),
                    "gen_ai.usage.cache_read.input_tokens": u.get("cached_input_tokens"),
                    "gen_ai.usage.cache_creation.input_tokens": u.get("cache_write_tokens"),
                    "gen_ai.usage.reasoning.output_tokens": u.get("reasoning_tokens"),
                    "gen_ai.response.finish_reasons": [d["stop_reason"]] if d.get("stop_reason") else None,
                    "weightspace.tool_calls": d.get("tool_calls"),
                },
                d.get("error"),
            )
        elif ev.type == "tool_start":
            attrs = {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": d["name"],
                "gen_ai.tool.call.id": d["call_id"],
                "weightspace.step": ev.step,
            }
            if self.capture_content:
                attrs["gen_ai.tool.call.arguments"] = json.dumps(d.get("arguments"), default=str)
            self._start((run, "tool", d["call_id"]), f"execute_tool {d['name']}", (run,), attrs)
        elif ev.type == "tool_end":
            self._end(
                (run, "tool", d["call_id"]),
                {"weightspace.output_chars": d.get("output_chars"), "weightspace.forward_cost": d.get("cost")},
                "tool raised" if d.get("error") else None,
            )
        elif ev.type == "run_end":
            u = d.get("usage") or {}
            self._end(
                (run,),
                {
                    "weightspace.stop_reason": d["stop_reason"],
                    "weightspace.steps": d["steps"],
                    "weightspace.cost_usd": d.get("cost"),
                    "weightspace.changed_params": d.get("changed_params"),
                    "gen_ai.usage.input_tokens": u.get("input_tokens"),
                    "gen_ai.usage.output_tokens": u.get("output_tokens"),
                },
                d.get("error"),
            )
            self._llm.pop(run, None)
