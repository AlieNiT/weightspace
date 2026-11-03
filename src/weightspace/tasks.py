"""One-call entry points: read / fix / write / improve."""

from __future__ import annotations

from collections.abc import Mapping

from .agent import Result, Session
from .space import WeightSpace


def as_weightspace(model) -> WeightSpace:
    """Wrap a WeightSpace, torch module, HF model, dict of arrays, or weights file path."""
    if isinstance(model, WeightSpace):
        return model
    if isinstance(model, str):
        return WeightSpace.from_file(model)
    if isinstance(model, Mapping):
        return WeightSpace.from_numpy(model)
    try:
        import torch

        if isinstance(model, torch.nn.Module):
            return WeightSpace.from_torch(model)
    except ImportError:
        pass
    raise TypeError(f"cannot build a WeightSpace from {type(model).__name__}")


def _run(mode, model, task, llm, text_mode=False, **kw) -> Result:
    ws = as_weightspace(model)
    text_kw = {k: kw.pop(k) for k in ("precision", "max_chars") if k in kw}
    s = Session(ws, llm, mode=mode, **kw)
    return s.run_text(task, **text_kw) if text_mode else s.run(task)


def read(model, question: str, llm, **kw) -> Result:
    """Answer a question about a network from its weights. Never edits."""
    return _run("read", model, question, llm, **kw)


def fix(model, problem: str, llm, **kw) -> Result:
    """Find and fix the cause of a described failure by editing weights."""
    return _run("fix", model, problem, llm, **kw)


def write(model, spec: str, llm, *, zero_init: bool = True, **kw) -> Result:
    """Program a network from scratch (weights zeroed first unless zero_init=False)."""
    ws = as_weightspace(model)
    if zero_init:
        ws.zero_()
        ws.commit()
    return _run("write", ws, spec, llm, **kw)


def improve(model, goal: str, llm, **kw) -> Result:
    """Make a network better at a goal, usually with probes and a forward budget."""
    return _run("improve", model, goal, llm, **kw)
