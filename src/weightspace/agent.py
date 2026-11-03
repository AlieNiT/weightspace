"""The agent loop: an LLM, a WeightSpace, tools, and a forward-pass budget."""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import prompts
from .events import CallbackLike, ConsoleLogger, Emitter, TranscriptWriter
from .llm import LLM, LLMResponse, get_llm, normalize_speed, normalize_thinking
from .probes import Probe
from .render import extract_blocks, fmt_array, parse_index, parse_weights, render_space
from .result import ForwardCall, Result
from .sandbox import Sandbox
from .space import WeightSpace
from .tools import Tool, tool
from .usage import Pricing, Usage

EDIT_MODES = ("fix", "write", "improve")

__all__ = ["Session", "SessionError", "Tool", "tool", "Result", "ForwardCall", "EDIT_MODES"]


class SessionError(RuntimeError):
    """The LLM request failed mid-run. `.result` holds everything up to the failure (transcript, edits, usage)."""

    def __init__(self, message: str, result: Result):
        super().__init__(message)
        self.result = result


class Session:
    """One weight-space programming run.

    ws:             the weights (see `WeightSpace.from_torch`, `from_hf`, `from_numpy`, `from_file`)
    llm:            an `LLM` or a spec like "openai:gpt-5" / "anthropic:claude-sonnet-4-5"
    mode:           "read" | "fix" | "write" | "improve"
    context:        natural-language description of architecture, input/output layout, environment...
    probes:         forward passes the LLM may spend (see `Probe`); each call costs `probe.cost`
    forward_budget: total forward-pass budget across probes
    python:         give the LLM a persistent Python sandbox over the weights
    thinking:       reasoning of the editing LLM: None = provider default, False/"off" = no thinking
                    (provider reasoning disabled and the LLM told to act directly), True or "low"/"medium"/"high"
    speed:          inference speed of the editing LLM: None = provider default, "fast" (or True) = fast mode
                    (same model, faster output, higher price), "standard", or a provider tier like "ultrafast"
    callbacks:      observers of run events (see `weightspace.events`), e.g. `OpenTelemetryCallback()`
    pricing:        `Pricing` (USD per million tokens) or a dict/tuple of it, to report and cap cost
    token_budget, cost_budget, time_budget:
                    stop asking for more work once the run has used this many tokens / USD / seconds; the LLM then
                    gets one last turn to submit
    parallel_tools: run read-only tool calls of one turn concurrently (number of worker threads; 1 = off)
    no_grad:        run sandbox code under `torch.no_grad()` so autograd-based training fails
    """

    def __init__(
        self,
        ws: WeightSpace,
        llm: LLM | str,
        *,
        mode: str = "read",
        context: str = "",
        probes: Iterable[Probe | Callable] = (),
        forward_budget: int = 0,
        python: bool = True,
        tools: Iterable[Tool] = (),
        namespace: dict | None = None,
        max_steps: int = 60,
        max_tool_chars: int = 12000,
        read_limit: int = 4096,
        sandbox_timeout: float | None = 300,
        system_prompt: str | None = None,
        thinking: bool | str | None = None,
        speed: bool | str | None = None,
        log_path: str | None = None,
        verbose: bool = False,
        callbacks: Iterable[CallbackLike] = (),
        pricing: Pricing | dict | tuple | None = None,
        token_budget: int | None = None,
        cost_budget: float | None = None,
        time_budget: float | None = None,
        parallel_tools: int = 8,
        no_grad: bool = True,
    ):
        if mode not in prompts.MODES:
            raise ValueError(f"mode must be one of {list(prompts.MODES)}")
        self.ws = ws
        self.llm = get_llm(llm)
        self.mode = mode
        self.context = context
        self.probes = [p if isinstance(p, Probe) else Probe(p) for p in probes]
        self.forward_budget = forward_budget
        self.max_steps = max_steps
        self.max_tool_chars = max_tool_chars
        self.read_limit = read_limit
        self.thinking = normalize_thinking(thinking)
        if self.thinking is not None:
            self.llm.thinking = self.thinking
        self.speed = normalize_speed(speed)
        if self.speed is not None:
            self.llm.speed = self.speed
        self.system_prompt = system_prompt or prompts.SYSTEM
        if self.thinking == "off":
            self.system_prompt += prompts.NO_THINKING
        self.log_path = log_path
        self.verbose = verbose
        self.pricing = Pricing.coerce(pricing)
        if cost_budget is not None and self.pricing is None:
            raise ValueError("cost_budget needs pricing=")
        self.token_budget, self.cost_budget, self.time_budget = token_budget, cost_budget, time_budget
        self.parallel_tools = max(1, int(parallel_tools))
        self.run_id = uuid.uuid4().hex[:12]
        self.forward_calls: list[ForwardCall] = []
        self.usage: dict[str, float] = {}
        self.tokens = Usage()
        self.timings: dict[str, dict[str, float]] = {}
        self._answer: tuple[str, float | None] | None = None
        self._forward_lock = threading.Lock()
        self._timing_lock = threading.Lock()
        self._t0 = time.monotonic()

        self.events = Emitter(callbacks)
        if verbose:
            self.events.add(ConsoleLogger())
        if log_path:
            self.events.add(TranscriptWriter(log_path))

        self.can_edit = mode in EDIT_MODES
        ns = {"np": np, "R": ws.R}
        try:
            import torch

            ns["torch"] = torch
        except ImportError:
            pass
        if self.can_edit:
            ns.update(W=ws.W, diff=ws.diff, checkpoint=ws.checkpoint, restore=ws.restore, revert=ws.revert)
        ns.update(namespace or {})
        self.sandbox = Sandbox(ns, timeout=sandbox_timeout, max_output=max_tool_chars, no_grad=no_grad) if python else None
        self.tools: dict[str, Tool] = {}
        for t in [*self._builtin_tools(), *self._probe_tools(), *tools]:
            self.tools[t.name] = t
        self.session_key = self.run_id
        if hasattr(self.llm, "session_key") and getattr(self.llm, "session_key", None) is None:
            self.llm.session_key = self.run_id

    # tools --------------------------------------------------------------------------
    def _builtin_tools(self) -> list[Tool]:
        ws = self.ws
        out = [
            tool(
                "list_tensors",
                "List weight tensors (name, shape, dtype, numel). Optional glob/substring filter.",
                parallel=True,
                pattern={"type": "string", "optional": True},
            )(self._list_tensors),
            tool(
                "describe_tensor",
                "Statistics of one tensor: quantiles, row/col norms, singular values.",
                parallel=True,
                name={"type": "string"},
            )(ws.describe),
            tool(
                "read_tensor",
                f"Read raw values of a tensor or a numpy-style slice of it, e.g. index='0:4, :16'. "
                f"At most {self.read_limit} values per call.",
                parallel=True,
                name={"type": "string"},
                index={"type": "string", "optional": True},
                precision={"type": "integer", "optional": True},
            )(self._read_tensor),
        ]
        if self.sandbox:
            names = "np, torch, R (read-only weights)" + (
                ", W (writable weights: edit in place or assign W[name] = array), diff(), checkpoint(label), "
                "restore(label), revert(name=None)"
                if self.can_edit
                else ""
            )
            out.append(
                tool(
                    "python",
                    f"Run Python in a persistent namespace with {names}. The last expression is printed. "
                    "Use it for static analysis of weights and for writing weights. It cannot run the network.",
                    code={"type": "string"},
                )(self.sandbox.run)
            )
        if self.can_edit:
            out += [
                tool(
                    "write_tensor",
                    "Overwrite a tensor or a slice of it with values (nested list or scalar).",
                    name={"type": "string"},
                    values={"description": "nested list of numbers, or a scalar"},
                    index={"type": "string", "optional": True},
                )(self._write_tensor),
                tool("diff", "Summarise all edits made so far versus the original weights.")(lambda: str(ws.diff())),
                tool("checkpoint", "Save the current weights under a label.", label={"type": "string"})(
                    lambda label: (ws.checkpoint(label), f"saved checkpoint {label!r}")[1]
                ),
                tool("restore", "Restore weights saved under a label.", label={"type": "string"})(
                    lambda label: (ws.restore(label), f"restored {label!r}\n{ws.diff()}")[1]
                ),
                tool(
                    "revert",
                    "Revert one tensor (or all, if name is omitted) to the original weights.",
                    name={"type": "string", "optional": True},
                )(lambda name=None: (ws.revert(name), f"reverted {name or 'all tensors'}\n{ws.diff()}")[1]),
            ]
        out.append(
            tool(
                "submit",
                "Finish: give the final answer (and for edits, a summary of what you changed and why).",
                answer={"type": "string"},
                confidence={"type": "number", "optional": True},
            )(self._submit)
        )
        return out

    def _probe_tools(self) -> list[Tool]:
        out = []
        for p in self.probes:

            def run(_p=p, **kw):
                return self._forward(_p, kw)

            out.append(Tool(f"forward_{p.name}", f"[forward pass, costs {p.cost}] {p.description}", p.parameters, run))
        return out

    def _list_tensors(self, pattern=None):
        names = self.ws.names(pattern)
        lines = [f"{n}  {list(self.ws.shape(n))}  {self.ws.dtype(n)}  {self.ws.numel(n):,}" for n in names]
        total = sum(self.ws.numel(n) for n in names)
        return "\n".join(lines) + f"\n{len(names)} tensors, {total:,} weights"

    def _read_tensor(self, name, index=None, precision=4):
        arr = np.asarray(self.ws[name][parse_index(index)])
        if arr.size > self.read_limit:
            return f"slice has {arr.size:,} values (shape {arr.shape}); limit is {self.read_limit}. Read a smaller slice."
        return f"{name}[{index or ':'}] shape={arr.shape}\n{fmt_array(arr, precision or 4)}"

    def _write_tensor(self, name, values, index=None):
        arr = self.ws.edit(name)
        arr[parse_index(index)] = np.asarray(values, dtype=arr.dtype)
        return f"wrote {name}[{index or ':'}]\n{self.ws.diff()}"

    @property
    def forward_used(self) -> int:
        return sum(c.cost for c in self.forward_calls)

    def _forward(self, p: Probe, args: dict) -> str:
        with self._forward_lock:
            remaining = self.forward_budget - self.forward_used
            if p.cost > remaining:
                return f"forward budget exhausted ({remaining} left, this probe costs {p.cost}). Submit your answer."
            self.ws.sync()
            t0 = time.monotonic()
            out = p.fn(**args)
            text = out if isinstance(out, str) else json.dumps(out, indent=1, default=_jsonable)
            self.forward_calls.append(ForwardCall(p.name, args, text, p.cost, time.monotonic() - t0))
        return text + f"\n[forward budget: {remaining - p.cost} of {self.forward_budget} left]"

    def _submit(self, answer, confidence=None):
        try:
            conf = None if confidence is None else float(confidence)
        except (TypeError, ValueError):
            conf = None
        self._answer = (str(answer), conf)
        return "submitted"

    # observability ------------------------------------------------------------------
    def _emit(self, type: str, step: int, **data) -> None:
        self.events.emit(type, self.run_id, step, **data)

    def _time(self, key: str, seconds: float) -> None:
        with self._timing_lock:
            t = self.timings.setdefault(key, {"calls": 0, "seconds": 0.0})
            t["calls"] += 1
            t["seconds"] = round(t["seconds"] + seconds, 4)

    def _append(self, messages: list[dict], msg: dict, step: int) -> None:
        messages.append(msg)
        self._emit("message", step, message=msg)

    @property
    def cost(self) -> float | None:
        return self.pricing.cost(self.tokens) if self.pricing else None

    def _over_budget(self) -> str | None:
        if self.token_budget is not None and self.tokens.total_tokens >= self.token_budget:
            return "token_budget"
        if self.cost_budget is not None and (self.cost or 0.0) >= self.cost_budget:
            return "cost_budget"
        if self.time_budget is not None and time.monotonic() - self._t0 >= self.time_budget:
            return "time_budget"
        return None

    def _add_usage(self, usage: dict) -> None:
        for k, v in (usage or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                self.usage[k] = self.usage.get(k, 0) + v

    def _complete(self, messages: list[dict], specs: list[dict] | None, step: int) -> LLMResponse:
        self._emit("llm_start", step, messages=len(messages), tools=len(specs or []))
        t0 = time.monotonic()
        try:
            resp = self.llm.complete(messages, specs)
        except BaseException as e:
            seconds = time.monotonic() - t0
            self._time("llm", seconds)
            self._emit("llm_end", step, seconds=seconds, usage={}, stop_reason=None, tool_calls=0, error=repr(e))
            raise
        seconds = time.monotonic() - t0
        self._time("llm", seconds)
        self._add_usage(resp.usage)
        tokens = resp.tokens if resp.tokens is not None else Usage(requests=1)
        self.tokens = self.tokens + tokens
        self._emit(
            "llm_end",
            step,
            seconds=seconds,
            usage=tokens.to_dict(),
            stop_reason=resp.stop_reason,
            tool_calls=len(resp.tool_calls),
        )
        return resp

    # tool execution -----------------------------------------------------------------
    def _exec_tool(self, name: str, args: dict, call_id: str = "", step: int = 0) -> str:
        t = self.tools.get(name)
        self._emit("tool_start", step, call_id=call_id, name=name, arguments=args)
        t0, error = time.monotonic(), False
        used = self.forward_used
        if t is None:
            out = f"unknown tool {name!r}; available: {', '.join(self.tools)}"
            error = True
        elif set(args) == {"_raw"}:
            out = f"invalid JSON arguments for {name}: {str(args['_raw'])[:200]!r}"
            error = True
        else:
            try:
                out = t.fn(**args)
            except Exception as e:  # surface errors to the LLM instead of crashing the run
                out, error = f"{type(e).__name__}: {e}", True
        out = out if isinstance(out, str) else json.dumps(out, default=_jsonable)
        if len(out) > self.max_tool_chars:
            out = out[: self.max_tool_chars] + f"\n... [truncated {len(out) - self.max_tool_chars:,} chars]"
        seconds = time.monotonic() - t0
        self._time(f"tool:{name}", seconds)
        self._emit(
            "tool_end",
            step,
            call_id=call_id,
            name=name,
            seconds=seconds,
            output_chars=len(out),
            error=error,
            cost=self.forward_used - used,
        )
        return out

    def _exec_calls(self, calls, step: int) -> list[str]:
        """Run one turn's tool calls in order; consecutive read-only calls run concurrently."""
        outs: list[str] = [""] * len(calls)
        i = 0
        while i < len(calls):
            j = i
            while j < len(calls) and self._parallel_ok(calls[j].name):
                j += 1
            if j - i > 1 and self.parallel_tools > 1:
                with ThreadPoolExecutor(min(self.parallel_tools, j - i), thread_name_prefix="weightspace-tool") as pool:
                    futs = [pool.submit(self._exec_tool, c.name, c.arguments, c.id, step) for c in calls[i:j]]
                    for k, f in enumerate(futs):
                        outs[i + k] = f.result()
                i = j
                continue
            c = calls[i]
            outs[i] = self._exec_tool(c.name, c.arguments, c.id, step)
            i += 1
        return outs

    def _parallel_ok(self, name: str) -> bool:
        t = self.tools.get(name)
        return t is not None and t.parallel

    # loop ---------------------------------------------------------------------------
    def _intro(self, task: str) -> str:
        names = self.ws.names()
        shown = names[:200]
        inv = "\n".join(f"  {n} {list(self.ws.shape(n))} {self.ws.dtype(n)}" for n in shown)
        if len(names) > len(shown):
            inv += f"\n  ... {len(names) - len(shown)} more (use list_tensors)"
        probes = "\n".join(f"  forward_{p.name} (cost {p.cost}): {p.description}" for p in self.probes) or "  none"
        parts = [prompts.MODES[self.mode], f"## Task\n{task}"]
        if self.context:
            parts.append(f"## Context\n{self.context}")
        parts.append(f"## Weights ({len(names)} tensors, {self.ws.numel():,} weights)\n{inv}")
        parts.append(f"## Forward passes (budget {self.forward_budget})\n{probes}")
        return "\n\n".join(parts)

    def _start(self, task: str) -> None:
        self._t0 = time.monotonic()
        self._emit(
            "run_start",
            0,
            mode=self.mode,
            llm=getattr(self.llm, "name", type(self.llm).__name__),
            task=task,
            tools=list(self.tools),
            forward_budget=self.forward_budget,
        )

    def run(self, task: str) -> Result:
        """Run the agent loop until the LLM submits, a limit is hit, or the run is interrupted (Ctrl-C)."""
        self._start(task)
        messages: list[dict] = []
        for m in ({"role": "system", "content": self.system_prompt}, {"role": "user", "content": self._intro(task)}):
            self._append(messages, m, 0)
        specs = [t.spec() for t in self.tools.values()]
        steps, nudges, stop = 0, 0, None
        try:
            while self._answer is None and steps < self.max_steps:
                stop = self._over_budget()
                if stop:
                    break
                steps += 1
                self._emit("step_start", steps)
                resp = self._complete(messages, specs, steps)
                self._append(messages, resp.to_message(), steps)
                if not resp.tool_calls:
                    if nudges >= 2:
                        self._answer = (resp.text or "", None)
                        stop = "answered"
                        break
                    nudges += 1
                    self._append(
                        messages, {"role": "user", "content": "Continue using the tools, or call `submit` when done."}, steps
                    )
                    continue
                for c, out in zip(resp.tool_calls, self._exec_calls(resp.tool_calls, steps), strict=True):
                    self._append(messages, {"role": "tool", "tool_call_id": c.id, "name": c.name, "content": out}, steps)
            if self._answer is None:
                stop = stop or "max_steps"
                why = "Step limit reached." if stop == "max_steps" else f"Run limit reached ({stop.replace('_', ' ')})."
                self._append(messages, {"role": "user", "content": f"{why} Call `submit` now."}, steps)
                resp = self._complete(messages, [self.tools["submit"].spec()], steps + 1)
                self._append(messages, resp.to_message(), steps + 1)
                for c in resp.tool_calls:
                    if c.name == "submit":
                        self._exec_tool("submit", c.arguments, c.id, steps + 1)
                if self._answer is None:
                    self._answer = (resp.text or "(no answer)", None)
        except KeyboardInterrupt:
            return self._finish(messages, steps, "interrupted")
        except Exception as e:
            res = self._finish(messages, steps, "error", error=f"{type(e).__name__}: {e}")
            raise SessionError(f"run {self.run_id} failed at step {steps}: {e}", res) from e
        return self._finish(messages, steps, stop or "submitted")

    def run_text(self, task: str, precision: int = 3, max_chars: int = 2_000_000) -> Result:
        """No tools: all weights go into the prompt, one reply comes back (Experiments 1-5 style)."""
        dump = render_space(self.ws, precision)
        if len(dump) > max_chars:
            raise ValueError(f"weights render to {len(dump):,} chars (> {max_chars:,}); use run() with tools instead")
        self._start(task)
        fmt = prompts.TEXT_MODE_EDIT if self.can_edit else prompts.TEXT_MODE_READ
        user = f"{self._intro(task).split('## Weights')[0]}## Weights\n{dump}\n\n{fmt}"
        messages: list[dict] = []
        for m in ({"role": "system", "content": self.system_prompt}, {"role": "user", "content": user}):
            self._append(messages, m, 0)
        try:
            resp = self._complete(messages, None, 1)
        except KeyboardInterrupt:
            return self._finish(messages, 0, "interrupted")
        except Exception as e:
            res = self._finish(messages, 0, "error", error=f"{type(e).__name__}: {e}")
            raise SessionError(f"run {self.run_id} failed: {e}", res) from e
        self._append(messages, resp.to_message(), 1)
        text = resp.text or ""
        if self.can_edit:
            notes = []
            for code in extract_blocks(text, "python"):
                sb = Sandbox({"np": np, "R": self.ws.R, "W": self.ws.W})
                notes.append(sb.run(code))
            for block in extract_blocks(text, "weights"):
                for name, arr in parse_weights(block).items():
                    self.ws.set(name, arr.reshape(self.ws.shape(name)))
            if notes:
                self._append(
                    messages, {"role": "tool", "tool_call_id": "text-mode", "name": "python", "content": "\n".join(notes)}, 1
                )
        answer = _field(text, "ANSWER") or text.strip()
        conf = _field(text, "CONFIDENCE")
        try:
            confidence = float(conf) if conf else None
        except ValueError:
            confidence = None
        self._answer = (answer, confidence)
        return self._finish(messages, 1, "submitted")

    def _finish(self, messages: list[dict], steps: int, stop_reason: str, error: str | None = None) -> Result:
        try:
            self.ws.sync()
        except Exception as e:  # never lose the transcript because a flush failed
            error = error or f"sync failed: {type(e).__name__}: {e}"
        answer, conf = self._answer or ("", None)
        seconds = time.monotonic() - self._t0
        res = Result(
            answer=answer,
            confidence=conf,
            mode=self.mode,
            thinking=self.thinking,
            diff=self.ws.diff(),
            forward_calls=self.forward_calls,
            forward_budget=self.forward_budget,
            steps=steps,
            transcript=messages,
            usage=self.usage,
            ws=self.ws,
            speed=self.speed,
            run_id=self.run_id,
            llm=getattr(self.llm, "name", type(self.llm).__name__),
            stop_reason=stop_reason,
            seconds=seconds,
            tokens=self.tokens,
            cost=self.cost,
            timings=self.timings,
            error=error,
        )
        self._emit(
            "run_end",
            steps,
            stop_reason=stop_reason,
            steps=steps,
            seconds=seconds,
            usage=self.tokens.to_dict(),
            cost=res.cost,
            answer=answer,
            changed_params=res.diff.changed_params,
            error=error,
        )
        return res


def _field(text: str, key: str) -> str | None:
    import re

    m = re.findall(rf"^\s*\**{key}\**\s*:\s*(.+)$", text, re.M | re.I)
    return m[-1].strip() if m else None


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return str(o)
