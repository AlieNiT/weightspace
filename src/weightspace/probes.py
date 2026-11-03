"""Forward passes the LLM may spend, defined by the user."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Probe:
    """A user-defined way to run the network, exposed to the LLM as a tool.

    `fn` is called with the LLM's arguments after the current weights are
    synced into the model. It decides what the LLM gets to see (scores,
    rollouts, attention maps, generated text...).
    """

    fn: Callable[..., Any]
    name: str = ""
    description: str = ""
    parameters: dict = field(default_factory=dict)
    cost: int = 1

    def __post_init__(self):
        self.name = self.name or self.fn.__name__
        self.description = self.description or inspect.getdoc(self.fn) or f"Run {self.name}."
        if not self.parameters:
            self.parameters = _schema_from_signature(self.fn)

    def __call__(self, **kw):
        return self.fn(**kw)


def probe(fn=None, *, name: str = "", description: str = "", parameters: dict | None = None, cost: int = 1):
    """Decorator: `@probe` or `@probe(cost=2, parameters={...})`."""

    def wrap(f):
        return Probe(f, name=name, description=description, parameters=parameters or {}, cost=cost)

    return wrap(fn) if fn is not None else wrap


_JSON_TYPES = {int: "integer", float: "number", str: "string", bool: "boolean", list: "array", dict: "object"}


def _schema_from_signature(fn) -> dict:
    props, required = {}, []
    for p in inspect.signature(fn).parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        ann = p.annotation
        origin = getattr(ann, "__origin__", ann)
        props[p.name] = {"type": _JSON_TYPES.get(origin, "string")} if ann is not p.empty else {}
        if p.default is p.empty:
            required.append(p.name)
        else:
            props[p.name]["default"] = p.default if isinstance(p.default, (int, float, str, bool, type(None))) else str(p.default)
    return {"type": "object", "properties": props, "required": required}
