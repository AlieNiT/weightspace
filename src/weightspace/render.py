"""Turning tensors into text an LLM can read, and back."""

from __future__ import annotations

import re
from typing import Any

import numpy as np


def fmt_num(x: float, precision: int = 3) -> str:
    s = f"{x:.{precision}g}"
    return "0" if s in ("0", "-0") else s


def fmt_array(arr: np.ndarray, precision: int = 3) -> str:
    """One line per row (last axis); leading axes are labelled."""
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return fmt_num(float(arr), precision)
    if arr.ndim == 1:
        return " ".join(fmt_num(float(v), precision) for v in arr)
    flat = arr.reshape(-1, arr.shape[-1])
    lead = list(np.ndindex(*arr.shape[:-1]))
    rows = []
    for idx, row in zip(lead, flat, strict=True):
        label = ",".join(str(i) for i in idx)
        rows.append(f"[{label}] " + " ".join(fmt_num(float(v), precision) for v in row))
    return "\n".join(rows)


_SLICE_PART = re.compile(r"^\s*(-?\d+)?\s*(?::\s*(-?\d+)?\s*(?::\s*(-?\d+)?\s*)?)?$")


def parse_index(index: str | None):
    """Parse numpy-style index text like "0:8, :, 3" without eval."""
    if index is None or not str(index).strip():
        return ()
    text = str(index).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    out: list[Any] = []
    for part in text.split(","):
        part = part.strip()
        if part in ("", ":", "..."):
            out.append(Ellipsis if part == "..." else slice(None))
            continue
        m = _SLICE_PART.match(part)
        if not m:
            raise ValueError(f"bad index component {part!r}")
        if ":" not in part:
            out.append(int(m.group(1)))
        else:
            a, b, c = (int(g) if g is not None else None for g in m.groups())
            out.append(slice(a, b, c))
    return tuple(out)


def stats(arr: np.ndarray) -> dict:
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    if a.size == 0:
        return {"numel": 0}
    q = np.quantile(a, [0.01, 0.25, 0.5, 0.75, 0.99])
    return {
        "numel": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        "abs_mean": float(np.abs(a).mean()),
        "frac_zero": float(np.mean(a == 0)),
        "q01_q25_q50_q75_q99": [float(v) for v in q],
    }


def describe(name: str, arr: np.ndarray, dtype: str = "", top_k: int = 8) -> str:
    """Structural summary: stats, row/col norms, spectrum for matrices."""
    arr = np.asarray(arr)
    s = stats(arr)
    lines = [f"{name}: shape={tuple(arr.shape)} dtype={dtype or arr.dtype} numel={s['numel']:,}"]
    if s["numel"] == 0:
        return lines[0]
    lines.append(
        "mean={} std={} min={} max={} |x|mean={} zeros={:.3g}%".format(
            *(fmt_num(s[k], 4) for k in ("mean", "std", "min", "max", "abs_mean")), 100 * s["frac_zero"]
        )
    )
    lines.append("quantiles 1/25/50/75/99%: " + " ".join(fmt_num(v, 4) for v in s["q01_q25_q50_q75_q99"]))
    if arr.ndim >= 2:
        m = np.asarray(arr, dtype=np.float64).reshape(arr.shape[0], -1)
        for axis, label in ((1, "row"), (0, "col")):
            norms = np.linalg.norm(m, axis=axis)
            order = np.argsort(norms)[::-1]
            top = ", ".join(f"{i}:{fmt_num(norms[i], 3)}" for i in order[:top_k])
            low = ", ".join(f"{i}:{fmt_num(norms[i], 3)}" for i in order[::-1][:top_k])
            lines.append(f"{label} L2 norms ({norms.size}): largest {top} | smallest {low}")
        if min(m.shape) <= 4096 and m.size <= 16_000_000:
            sv = np.linalg.svd(m, compute_uv=False)
            energy = np.cumsum(sv**2) / max(np.sum(sv**2), 1e-30)
            eff = int(np.searchsorted(energy, 0.9) + 1)
            lines.append(
                f"singular values (top {min(top_k, sv.size)}): "
                + " ".join(fmt_num(v, 3) for v in sv[:top_k])
                + f" | rank for 90% energy: {eff}/{sv.size}"
            )
    return "\n".join(lines)


def render_space(ws, precision: int = 3) -> str:
    """Full text dump in the same format `parse_weights` reads back."""
    parts = []
    for n in ws.names():
        arr = ws[n]
        parts.append(f"### {n} {list(arr.shape)}\n{fmt_array(arr, precision)}")
    return "\n\n".join(parts)


_HEADER = re.compile(r"^###\s+(\S+)(?:\s+\[([\d,\s]*)\])?\s*$")
_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def parse_weights(text: str) -> dict[str, np.ndarray]:
    """Parse `### name [shape]` blocks of numbers (row labels like [0,1] are ignored)."""
    out: dict[str, np.ndarray] = {}
    name, shape = None, None
    nums: list[float] = []

    def done():
        if name is None:
            return
        arr = np.array(nums, dtype=np.float32)
        if shape is not None:
            arr = arr.reshape(shape)
        out[name] = arr

    for line in text.splitlines():
        m = _HEADER.match(line.strip())
        if m:
            done()
            name = m.group(1)
            shape = tuple(int(x) for x in m.group(2).split(",") if x.strip()) if m.group(2) is not None else None
            nums = []
            continue
        if name is None:
            continue
        body = re.sub(r"^\s*\[[\d,\s]*\]", "", line)
        nums.extend(float(x) for x in _NUM.findall(body))
    done()
    return out


def extract_blocks(text: str, lang: str) -> list[str]:
    return [m.group(1) for m in re.finditer(rf"```{lang}[^\n]*\n(.*?)```", text, re.S)]
