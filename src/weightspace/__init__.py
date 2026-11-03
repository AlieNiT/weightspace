"""weightspace: let an LLM read and write neural network weights directly.

import weightspace as wsp
res = wsp.fix(model, "the car never gets over the ramp", llm="openai:gpt-5",
              context=ENV_DOC, probes=[rollout], forward_budget=10)
print(res)                  # answer + how many weights changed
res.save("runs/fix-1")      # transcript, result.json, sparse patch
"""

from .agent import ForwardCall, Result, Session, Tool, tool
from .llm import LLM, LLMResponse, ScriptedLLM, ToolCall, get_llm
from .probes import Probe, probe
from .render import describe, fmt_array, parse_weights, render_space
from .space import DiffSummary, NumpyStore, Store, WeightSpace
from .tasks import as_weightspace, fix, improve, read, write

__version__ = "0.2.0"

__all__ = [
    "WeightSpace",
    "Store",
    "NumpyStore",
    "DiffSummary",
    "Session",
    "Result",
    "ForwardCall",
    "Tool",
    "tool",
    "Probe",
    "probe",
    "LLM",
    "LLMResponse",
    "ToolCall",
    "ScriptedLLM",
    "get_llm",
    "read",
    "fix",
    "write",
    "improve",
    "as_weightspace",
    "describe",
    "fmt_array",
    "render_space",
    "parse_weights",
]
