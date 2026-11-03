"""Tools the LLM can call, and the helper to declare custom ones."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class Tool:
    """A function exposed to the LLM. `parallel=True` marks it read-only and safe to run concurrently with other
    parallel tools requested in the same turn."""

    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]
    parallel: bool = False

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def tool(name: str, description: str, /, *, parallel: bool = False, **props) -> Callable[[Callable], Tool]:
    """Small helper for custom tools: `tool("name", "desc", x={"type": "integer"})(fn)`.

    Mark a property `{"optional": True}` to leave it out of `required`.
    """
    props = {k: dict(v) for k, v in props.items()}
    required = [k for k, v in props.items() if not v.pop("optional", False)]

    def wrap(fn):
        return Tool(name, description, {"type": "object", "properties": props, "required": required}, fn, parallel)

    return wrap
