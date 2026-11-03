"""Turning tensors into text an LLM can read, and back."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from . import _chunks


def fmt_num(x: float, precision: int = 3) -> str:
    s = f"{x:.{precision}g}"
    return "0" if s in ("0", "-0") else s


def fmt_array(arr: np.ndarray, precision: int = 3) -> str:
    """One line per row (last axis); leading axes are labelled."""
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return fmt_num(float(arr), precision)
    if arr.ndim == 1:
        return " ".join(fmt_num(v, precision) for v in arr.tolist())
    flat = arr.reshape(-1, arr.shape[-1])
    lead = list(np.ndindex(*arr.shape[:-1]))
    rows = []
    for idx, row in zip(lead, flat.tolist(), strict=True):
        label = ",".join(str(i) for i in idx)
        rows.append(f"[{label}] " + " ".join(fmt_num(v, precision) for v in row))
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


EXACT_QUANTILE_MAX = 1 << 22  # above this size, quantiles use a parallel histogram pass instead of a full sort
_BINS = 4096


def _moments(x: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """mean, std, min, max, mean |x|, fraction of zeros: one float64 pass per block, merged with Chan's formula."""

    def block(i: int, j: int):
        raw = x[i:j]
        c = raw.astype(np.float64)
        m = float(c.sum()) / c.size
        dev = c - m
        m2 = float(np.dot(dev, dev))
        lo, hi, zeros = float(raw.min()), float(raw.max()), int(np.count_nonzero(raw == 0))
        return c.size, m, m2, lo, hi, float(np.abs(c, out=c).sum()), zeros

    n, mean, m2 = 0, 0.0, 0.0
    lo, hi, abs_sum, zeros = np.inf, -np.inf, 0.0, 0
    for bn, bm, bm2, blo, bhi, babs, bz in _chunks.map_blocks(block, x.size):
        tot = n + bn
        delta = bm - mean
        mean += delta * bn / tot
        m2 += bm2 + delta * delta * n * bn / tot
        n = tot
        lo, hi, abs_sum, zeros = min(lo, blo), max(hi, bhi), abs_sum + babs, zeros + bz
    return mean, (m2 / n) ** 0.5, lo, hi, abs_sum / n, zeros / n


def quantiles(x: np.ndarray, qs, lo: float | None = None, hi: float | None = None) -> list[float]:
    """Same values as `np.quantile(x, qs)` (linear interpolation). Large arrays get one parallel histogram pass,
    then only the values in the bins holding the wanted ranks are gathered and partitioned."""
    x = x.reshape(-1)
    n = x.size
    lo = float(x.min()) if lo is None else lo
    hi = float(x.max()) if hi is None else hi
    if n <= EXACT_QUANTILE_MAX or not (np.isfinite(lo) and np.isfinite(hi)) or lo == hi:
        return [float(v) for v in np.quantile(x.astype(np.float64), qs)]
    edges = np.linspace(lo, hi, _BINS + 1)
    counts = np.sum(_chunks.map_blocks(lambda i, j: np.histogram(x[i:j], edges)[0], n), axis=0)
    cum = np.cumsum(counts)
    pos = [q * (n - 1) for q in qs]
    ranks = sorted({int(np.floor(p)) for p in pos} | {min(int(np.floor(p)) + 1, n - 1) for p in pos})
    by_bin: dict[int, list[int]] = {}
    for r in ranks:
        by_bin.setdefault(int(np.searchsorted(cum, r, side="right")), []).append(r)
    value: dict[int, float] = {}
    for b, rs in by_bin.items():
        left, right, last = np.float64(edges[b]), np.float64(edges[b + 1]), b == _BINS - 1

        def take(i: int, j: int, left=left, right=right, last=last):
            c = x[i:j]
            return c[(c >= left) & ((c <= right) if last else (c < right))]

        vals = np.concatenate(_chunks.map_blocks(take, n)).astype(np.float64)
        base = int(cum[b - 1]) if b else 0
        part = np.partition(vals, [r - base for r in rs])
        value.update({r: float(part[r - base]) for r in rs})
    out = []
    for p in pos:
        f = int(np.floor(p))
        out.append(value[f] + (p - f) * (value[min(f + 1, n - 1)] - value[f]))
    return out


def stats(arr: np.ndarray) -> dict:
    a = np.asarray(arr).reshape(-1)
    if a.size == 0:
        return {"numel": 0}
    if a.dtype.kind != "f":
        a = a.astype(np.float64)
    mean, std, lo, hi, abs_mean, frac_zero = _moments(a)
    return {
        "numel": int(a.size),
        "mean": mean,
        "std": std,
        "min": lo,
        "max": hi,
        "abs_mean": abs_mean,
        "frac_zero": frac_zero,
        "q01_q25_q50_q75_q99": quantiles(a, [0.01, 0.25, 0.5, 0.75, 0.99], lo, hi),
    }


SVD_MAX_DIM = 256  # exact SVD below this; Gram-matrix eigenvalues above (much faster, same leading spectrum)


def singular_values(m: np.ndarray) -> np.ndarray:
    """Singular values, descending. Small matrices use an exact SVD; larger ones the eigenvalues of the smaller
    Gram matrix, computed with BLAS in the input precision (float32 stays float32) and solved in float64."""
    r, c = m.shape
    if min(r, c) <= SVD_MAX_DIM:
        return np.linalg.svd(np.asarray(m, dtype=np.float64), compute_uv=False)
    if m.dtype not in (np.float32, np.float64):
        m = m.astype(np.float32)
    g = m.T @ m if c < r else m @ m.T
    ev = np.linalg.eigvalsh(g.astype(np.float64))[::-1]
    return np.sqrt(np.clip(ev, 0, None))


def norms(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row and column L2 norms of a 2-D array, accumulated in float64 over row blocks."""
    rows_per = max(1, _chunks.CHUNK // max(1, m.shape[1]))

    def block(i: int, j: int):
        b = m[i:j].astype(np.float64)
        return np.einsum("ij,ij->i", b, b), np.einsum("ij,ij->j", b, b)

    parts = _chunks.map_blocks(block, m.shape[0], rows_per, elements=m.size)
    rows = np.concatenate([p[0] for p in parts])
    cols = np.sum([p[1] for p in parts], axis=0)
    return np.sqrt(rows), np.sqrt(cols)


SPECTRUM_MAX_NUMEL = 256_000_000


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
        m = arr.reshape(arr.shape[0], -1)
        if m.dtype.kind != "f":
            m = m.astype(np.float64)
        for norms_, label in zip(norms(m), ("row", "col"), strict=True):
            order = np.argsort(norms_)[::-1]
            top = ", ".join(f"{i}:{fmt_num(norms_[i], 3)}" for i in order[:top_k])
            low = ", ".join(f"{i}:{fmt_num(norms_[i], 3)}" for i in order[::-1][:top_k])
            lines.append(f"{label} L2 norms ({norms_.size}): largest {top} | smallest {low}")
        if min(m.shape) <= 4096 and m.size <= SPECTRUM_MAX_NUMEL:
            sv = singular_values(m)
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
