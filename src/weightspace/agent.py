"""The agent loop: an LLM, a WeightSpace, tools, and a forward-pass budget."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import prompts
from .llm import LLM, get_llm
from .probes import Probe
from .render import describe, extract_blocks, fmt_array, parse_index, parse_weights, render_space
from .sandbox import Sandbox
from .space import DiffSummary, WeightSpace

EDIT_MODES = ("fix", "write", "improve")


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def tool(name: str, description: str, /, **props) -> Callable[[Callable], Tool]:
    """Small helper for custom tools: `tool("name", "desc", x={"type": "integer"})(fn)`."""
    required = [k for k, v in props.items() if not v.pop("optional", False)]

    def wrap(fn):
        return Tool(name, description, {"type": "object", "properties": props, "required": required}, fn)

    return wrap


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
    diff: DiffSummary
    forward_calls: list[ForwardCall]
    forward_budget: int
    steps: int
    transcript: list[dict]
    usage: dict
    ws: WeightSpace = field(repr=False)

    @property
    def forward_used(self) -> int:
        return sum(c.cost for c in self.forward_calls)

    def __str__(self) -> str:
        conf = f" (confidence {self.confidence:.2f})" if self.confidence is not None else ""
        return (
            f"[{self.mode}] {self.answer}{conf}\n"
            f"steps={self.steps} forward passes={self.forward_used}/{self.forward_budget}\n{self.diff}"
        )

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "transcript.jsonl"), "w") as f:
            for m in self.transcript:
                f.write(json.dumps(m, default=str) + "\n")
        summary = {
            "mode": self.mode,
            "answer": self.answer,
            "confidence": self.confidence,
            "steps": self.steps,
            "forward_used": self.forward_used,
            "forward_budget": self.forward_budget,
            "forward_calls": [c.__dict__ for c in self.forward_calls],
            "changed_params": self.diff.changed_params,
            "total_params": self.diff.total_params,
            "changed_fraction": self.diff.fraction,
            "tensors": [t.__dict__ for t in self.diff.tensors],
            "usage": self.usage,
        }
        with open(os.path.join(directory, "result.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
        if self.diff.changed_params:
            self.ws.save_patch(os.path.join(directory, "patch.npz"))


class Session:
    """One weight-space programming run.

    ws:             the weights (see `WeightSpace.from_torch`, `from_hf`, `from_numpy`, `from_file`)
    llm:            an `LLM` or a spec like "openai:gpt-5" / "anthropic:claude-sonnet-4-5"
    mode:           "read" | "fix" | "write" | "improve"
    context:        natural-language description of architecture, input/output layout, environment...
    probes:         forward passes the LLM may spend (see `Probe`); each call costs `probe.cost`
    forward_budget: total forward-pass budget across probes
    python:         give the LLM a persistent Python sandbox over the weights
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
        log_path: str | None = None,
        verbose: bool = False,
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
        self.system_prompt = system_prompt or prompts.SYSTEM
        self.log_path = log_path
        self.verbose = verbose
        self.forward_calls: list[ForwardCall] = []
        self.usage: dict[str, float] = {}
        self._answer: tuple[str, float | None] | None = None

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
        self.sandbox = Sandbox(ns, timeout=sandbox_timeout, max_output=max_tool_chars) if python else None
        self.tools: dict[str, Tool] = {}
        for t in [*self._builtin_tools(), *self._probe_tools(), *tools]:
            self.tools[t.name] = t

    # tools --------------------------------------------------------------------------
    def _builtin_tools(self) -> list[Tool]:
        ws = self.ws
        out = [
            tool(
                "list_tensors",
                "List weight tensors (name, shape, dtype, numel). Optional glob/substring filter.",
                pattern={"type": "string", "optional": True},
            )(self._list_tensors),
            tool(
                "describe_tensor", "Statistics of one tensor: quantiles, row/col norms, singular values.", name={"type": "string"}
            )(lambda name: describe(name, ws[name], ws.dtype(name))),
            tool(
                "read_tensor",
                f"Read raw values of a tensor or a numpy-style slice of it, e.g. index='0:4, :16'. "
                f"At most {self.read_limit} values per call.",
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
        arr = self.ws[name][parse_index(index)]
        arr = np.asarray(arr)
        if arr.size > self.read_limit:
            return f"slice has {arr.size:,} values (shape {arr.shape}); limit is {self.read_limit}. Read a smaller slice."
        return f"{name}[{index or ':'}] shape={arr.shape}\n{fmt_array(arr, precision or 4)}"

    def _write_tensor(self, name, values, index=None):
        arr = self.ws.edit(name)
        arr[parse_index(index)] = np.asarray(values, dtype=arr.dtype)
        return f"wrote {name}[{index or ':'}]\n{self.ws.diff()}"

    def _forward(self, p: Probe, args: dict) -> str:
        remaining = self.forward_budget - sum(c.cost for c in self.forward_calls)
        if p.cost > remaining:
            return f"forward budget exhausted ({remaining} left, this probe costs {p.cost}). Submit your answer."
        self.ws.sync()
        t0 = time.time()
        out = p.fn(**args)
        text = out if isinstance(out, str) else json.dumps(out, indent=1, default=_jsonable)
        self.forward_calls.append(ForwardCall(p.name, args, text, p.cost, time.time() - t0))
        return text + f"\n[forward budget: {remaining - p.cost} of {self.forward_budget} left]"

    def _submit(self, answer, confidence=None):
        self._answer = (str(answer), None if confidence is None else float(confidence))
        return "submitted"

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

    def _log(self, msg: dict) -> None:
        if self.verbose:
            role = msg["role"]
            body = msg.get("content") or ""
            calls = msg.get("tool_calls") or []
            print(f"--- {role}{' ' + msg.get('name', '') if role == 'tool' else ''}")
            if body:
                print(body if len(body) < 2000 else body[:2000] + " ...")
            for c in calls:
                print(f">>> {c['name']}({json.dumps(c['arguments'])[:1500]})")
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(msg, default=str) + "\n")

    def _add_usage(self, usage: dict) -> None:
        for k, v in (usage or {}).items():
            if isinstance(v, (int, float)):
                self.usage[k] = self.usage.get(k, 0) + v

    def _exec_tool(self, name: str, args: dict) -> str:
        t = self.tools.get(name)
        if t is None:
            return f"unknown tool {name!r}"
        try:
            out = t.fn(**args)
        except Exception as e:  # surface errors to the LLM instead of crashing the run
            out = f"{type(e).__name__}: {e}"
        out = out if isinstance(out, str) else json.dumps(out, default=_jsonable)
        if len(out) > self.max_tool_chars:
            out = out[: self.max_tool_chars] + f"\n... [truncated {len(out) - self.max_tool_chars:,} chars]"
        return out

    def run(self, task: str) -> Result:
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": self._intro(task)}]
        for m in messages:
            self._log(m)
        specs = [t.spec() for t in self.tools.values()]
        steps, nudges = 0, 0
        while self._answer is None and steps < self.max_steps:
            steps += 1
            resp = self.llm.complete(messages, specs)
            self._add_usage(resp.usage)
            msg = resp.to_message()
            messages.append(msg)
            self._log(msg)
            if not resp.tool_calls:
                if nudges >= 2:
                    self._answer = (resp.text or "", None)
                    break
                nudges += 1
                messages.append({"role": "user", "content": "Continue using the tools, or call `submit` when done."})
                continue
            for c in resp.tool_calls:
                out = self._exec_tool(c.name, c.arguments)
                tmsg = {"role": "tool", "tool_call_id": c.id, "name": c.name, "content": out}
                messages.append(tmsg)
                self._log(tmsg)
        if self._answer is None:
            messages.append({"role": "user", "content": "Step limit reached. Call `submit` now."})
            resp = self.llm.complete(messages, [self.tools["submit"].spec()])
            self._add_usage(resp.usage)
            messages.append(resp.to_message())
            for c in resp.tool_calls:
                if c.name == "submit":
                    self._exec_tool("submit", c.arguments)
            if self._answer is None:
                self._answer = (resp.text or "(no answer)", None)
        self.ws.sync()
        return self._result(messages, steps)

    def run_text(self, task: str, precision: int = 3, max_chars: int = 2_000_000) -> Result:
        """No tools: all weights go into the prompt, one reply comes back (Experiments 1-5 style)."""
        dump = render_space(self.ws, precision)
        if len(dump) > max_chars:
            raise ValueError(f"weights render to {len(dump):,} chars (> {max_chars:,}); use run() with tools instead")
        fmt = prompts.TEXT_MODE_EDIT if self.can_edit else prompts.TEXT_MODE_READ
        user = f"{self._intro(task).split('## Weights')[0]}## Weights\n{dump}\n\n{fmt}"
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user}]
        resp = self.llm.complete(messages, None)
        self._add_usage(resp.usage)
        messages.append(resp.to_message())
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
                messages.append({"role": "tool", "tool_call_id": "text-mode", "name": "python", "content": "\n".join(notes)})
            self.ws.sync()
        answer = _field(text, "ANSWER") or text.strip()
        conf = _field(text, "CONFIDENCE")
        try:
            confidence = float(conf) if conf else None
        except ValueError:
            confidence = None
        self._answer = (answer, confidence)
        return self._result(messages, 1)

    def _result(self, messages, steps) -> Result:
        answer, conf = self._answer
        return Result(
            answer=answer,
            confidence=conf,
            mode=self.mode,
            diff=self.ws.diff(),
            forward_calls=self.forward_calls,
            forward_budget=self.forward_budget,
            steps=steps,
            transcript=messages,
            usage=self.usage,
            ws=self.ws,
        )


def _field(text: str, key: str) -> str | None:
    m = re.findall(rf"^\s*\**{key}\**\s*:\s*(.+)$", text, re.M | re.I)
    return m[-1].strip() if m else None


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return str(o)
