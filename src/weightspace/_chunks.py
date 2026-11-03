"""Block-wise passes over large arrays. NumPy releases the GIL inside its kernels, so big tensors are split
into blocks that run on a small shared thread pool; small ones stay on the calling thread."""

from __future__ import annotations

import os
import threading
import zlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

import numpy as np

T = TypeVar("T")

CHUNK = 1 << 20  # elements per block
PARALLEL_MIN = 1 << 22  # below this many elements, one thread is faster than the pool overhead
WORKERS = max(1, min(8, os.cpu_count() or 1))

_pool: ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()


def _executor() -> ThreadPoolExecutor:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ThreadPoolExecutor(WORKERS, thread_name_prefix="weightspace-block")
    return _pool


def map_blocks(fn: Callable[[int, int], T], n: int, block: int | None = None, *, elements: int | None = None) -> list[T]:
    """`fn(start, stop)` over `[0, n)` in blocks of `block` (default CHUNK), results in order.
    `elements` is the total work size used to decide on threading (defaults to `n`)."""
    block = max(1, block or CHUNK)
    spans = [(i, min(i + block, n)) for i in range(0, n, block)]
    if len(spans) < 2 or WORKERS < 2 or (elements if elements is not None else n) < PARALLEL_MIN:
        return [fn(a, b) for a, b in spans]
    return list(_executor().map(lambda s: fn(*s), spans))


_BITS = {2: np.int16, 4: np.int32, 8: np.int64}


def bits(a: np.ndarray) -> np.ndarray:
    """Integer view of a float array, so equality is bitwise (NaN == NaN, a sign flip of 0 counts as an edit)."""
    t = _BITS.get(a.dtype.itemsize) if a.dtype.kind == "f" else None
    return a.view(t) if t is not None else a


def flat(a: np.ndarray) -> np.ndarray:
    return a.reshape(-1)


def copy(src: np.ndarray) -> np.ndarray:
    """`src.copy()`, done in parallel blocks for large contiguous arrays."""
    if src.size < PARALLEL_MIN or not src.flags.c_contiguous:
        return src.copy()
    out = np.empty_like(src)
    copy_into(out, src)
    return out


def copy_into(dst: np.ndarray, src: np.ndarray) -> None:
    """`dst[...] = src` for same-shape arrays, in parallel blocks when both are large and contiguous."""
    if dst.size < PARALLEL_MIN or not (dst.flags.c_contiguous and src.flags.c_contiguous) or dst.shape != src.shape:
        dst[...] = src
        return
    d, s = flat(dst), flat(src)
    map_blocks(lambda i, j: np.copyto(d[i:j], s[i:j]), d.size)


def changed_indices(cur: np.ndarray, orig: np.ndarray) -> np.ndarray:
    """Flat int64 indices where `cur` and `orig` differ bitwise."""
    a, b = bits(flat(cur)), bits(flat(orig))
    parts = map_blocks(lambda i, j: np.flatnonzero(a[i:j] != b[i:j]) + i, a.size)
    return np.concatenate(parts).astype(np.int64, copy=False) if parts else np.empty(0, np.int64)


def equal(cur: np.ndarray, orig: np.ndarray) -> bool:
    a, b = bits(flat(cur)), bits(flat(orig))
    return all(map_blocks(lambda i, j: bool(np.array_equal(a[i:j], b[i:j])), a.size))


def sum_squares(a: np.ndarray) -> float:
    x = flat(a)

    def block(i: int, j: int) -> float:
        c = x[i:j].astype(np.float64)
        return float(np.dot(c, c))

    return float(sum(map_blocks(block, x.size)))


def fingerprint(a: np.ndarray) -> tuple:
    """Cheap content key: shape, dtype and a CRC32 per block."""
    x = flat(a)
    crcs = map_blocks(lambda i, j: zlib.crc32(np.ascontiguousarray(x[i:j]).data), x.size)
    return (a.shape, a.dtype.str, *crcs)
