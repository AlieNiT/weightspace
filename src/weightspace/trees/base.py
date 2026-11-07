"""Shared staging, validation, and immutable descriptions for fixed tree structures."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from .. import _chunks
from ..space import NumpyStore


def freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(freeze(v) for v in value)
    return value


def digest(value) -> str:
    # Native inference structure can contain reserved infinity thresholds.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def check_version(module, low: tuple[int, int], high: tuple[int, int]) -> None:
    version = tuple(int(p) for p in module.__version__.split(".")[:2])
    if not low <= version < high:
        bounds = f">={'.'.join(map(str, low))},<{'.'.join(map(str, high))}"
        raise ValueError(f"unsupported {module.__name__} version {module.__version__}; require {bounds}")


class TreeStore(NumpyStore):
    """Arrays are private staging buffers. Native state changes only after full validation."""

    def __init__(self, arrays, info, trees, structure):
        super().__init__(arrays)
        self._schema = {n: (a.shape, a.dtype) for n, a in self._t.items()}
        self._installed = {n: a.copy() for n, a in self._t.items()}
        self._info = info
        self._trees = trees
        self._structure = digest(structure)
        self._model_info = freeze(info)
        self.validate()

    @property
    def model_info(self):
        return self._model_info

    def shape(self, name):
        return self._schema[name][0]

    def describe_tree(self, tree_id, *, offset=0, limit=100):
        if isinstance(tree_id, bool) or not isinstance(tree_id, int) or not 0 <= tree_id < len(self._trees):
            raise ValueError(f"tree_id must be in [0, {len(self._trees)})")
        if not isinstance(offset, int) or offset < 0 or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("offset must be nonnegative and limit must be between 1 and 100")
        tree = self._trees[tree_id]
        return freeze({"tree_id": tree_id, "offset": offset, "total": len(tree), "nodes": tree[offset : offset + limit]})

    def validate(self, arrays: Mapping[str, np.ndarray] | None = None):
        arrays = self._t if arrays is None else arrays
        if set(arrays) != set(self._schema):
            raise ValueError("tree parameter names must remain fixed")
        for name, (shape, dtype) in self._schema.items():
            a = arrays[name]
            if a.shape != shape or a.dtype != dtype:
                raise ValueError(f"{name}: expected shape {shape} and dtype {dtype}")
            if not np.isfinite(a).all():
                raise ValueError(f"{name}: all values must be finite")

    def flush(self, names):
        self.validate()
        if all(_chunks.equal(a, self._installed[n]) for n, a in self._t.items()):
            return
        self._install()
        self._installed = {n: a.copy() for n, a in self._t.items()}

    def _install(self):
        raise NotImplementedError

    def initialize_for_write(self):
        for a in self._t.values():
            a.fill(0)
        return {"preserved_border_groups": []}

    def patch_metadata(self, arrays):
        h = hashlib.sha256()
        for n in sorted(arrays):
            a = np.ascontiguousarray(arrays[n])
            h.update(n.encode())
            h.update(a.tobytes())
        return {
            "version": 1,
            "backend": self._info["backend"],
            "schema": {n: {"shape": list(s), "dtype": str(d)} for n, (s, d) in self._schema.items()},
            "structure": self._structure,
            "baseline": h.hexdigest(),
        }
