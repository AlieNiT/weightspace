"""A model-agnostic view of a network's weights as named numpy arrays.

The LLM never sees the model object, only tensors. Edits are tracked against
the original weights so we can report how much changed, roll back, and export
a sparse patch.
"""

from __future__ import annotations

import fnmatch
import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field

import numpy as np

from . import _chunks


class Store:
    """Backend holding the live tensors. Subclasses adapt torch, HF, files..."""

    def names(self) -> list[str]:
        raise NotImplementedError

    def shape(self, name: str) -> tuple[int, ...]:
        raise NotImplementedError

    def dtype(self, name: str) -> str:
        raise NotImplementedError

    def get(self, name: str) -> np.ndarray:
        """Return a writable float array; edits to it must survive `flush`."""
        raise NotImplementedError

    def flush(self, names: list[str]) -> None:
        """Push edits of `names` back to the underlying model, if any."""

    @property
    def model_info(self) -> Mapping:
        return {}

    def validate(self, arrays=None) -> None:
        """Validate proposed parameters without changing native state."""

    def describe_tree(self, tree_id, *, offset=0, limit=100):
        raise TypeError("this store has no trees")

    def initialize_for_write(self):
        for n in self.names():
            self.get(n).fill(0)
        return {}

    def patch_metadata(self, arrays):
        return None

    def save_model(self, path):
        raise TypeError("native predictive export is available only for tree models; use save() for arrays")


class NumpyStore(Store):
    def __init__(self, tensors: Mapping[str, np.ndarray]):
        self._t = {}
        for k, v in tensors.items():
            arr = np.asarray(v)
            if not np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float32)
            self._t[k] = arr
        self._original_dtypes = {k: str(np.asarray(v).dtype) for k, v in tensors.items()}

    def names(self):
        return list(self._t)

    def shape(self, name):
        return tuple(self._t[name].shape)

    def dtype(self, name):
        return self._original_dtypes[name]

    def get(self, name):
        return self._t[name]


@dataclass
class TensorDiff:
    name: str
    numel: int
    changed: int
    l2_delta: float
    max_abs_delta: float
    rel_delta: float


@dataclass
class DiffSummary:
    total_params: int
    changed_params: int
    tensors: list[TensorDiff] = field(default_factory=list)

    @property
    def fraction(self) -> float:
        return self.changed_params / self.total_params if self.total_params else 0.0

    def __str__(self) -> str:
        if not self.tensors:
            return "no weights changed"
        lines = [
            f"{self.changed_params:,} of {self.total_params:,} weights changed "
            f"({100 * self.fraction:.4g}%) in {len(self.tensors)} tensor(s)"
        ]
        for t in self.tensors:
            lines.append(
                f"  {t.name}: {t.changed:,}/{t.numel:,} changed, "
                f"|delta|={t.l2_delta:.4g} ({100 * t.rel_delta:.3g}% of norm), max={t.max_abs_delta:.4g}"
            )
        return "\n".join(lines)


class _ReadOnly(Mapping):
    def __init__(self, ws: WeightSpace):
        self._ws = ws

    def __getitem__(self, name):
        arr = self._ws._store.get(name).view()
        arr.flags.writeable = False
        return arr

    def __iter__(self):
        return iter(self._ws.names())

    def __len__(self):
        return len(self._ws.names())

    def __repr__(self):
        return f"<read-only weights: {len(self)} tensors>"


class _Writable(Mapping):
    """Mapping handed to the sandbox. Touching a tensor snapshots its original."""

    def __init__(self, ws: WeightSpace):
        self._ws = ws

    def __getitem__(self, name):
        return self._ws.edit(name)

    def __setitem__(self, name, value):
        self._ws.set(name, value)

    def __iter__(self):
        return iter(self._ws.names())

    def __len__(self):
        return len(self._ws.names())

    def __repr__(self):
        return f"<writable weights: {len(self)} tensors>"


def _tensor_diff(name: str, cur: np.ndarray, orig: np.ndarray, base_sq: float) -> TensorDiff | None:
    a, b = _chunks.flat(cur), _chunks.flat(orig)
    ab, bb = _chunks.bits(a), _chunks.bits(b)

    def block(i: int, j: int) -> tuple[int, float, float]:
        ne = ab[i:j] != bb[i:j]
        n = int(np.count_nonzero(ne))
        if not n:
            return 0, 0.0, 0.0
        d = a[i:j][ne].astype(np.float64) - b[i:j][ne]
        return n, float(np.dot(d, d)), float(np.abs(d).max())

    parts = _chunks.map_blocks(block, a.size)
    changed = sum(p[0] for p in parts)
    if not changed:
        return None
    l2, base = sum(p[1] for p in parts) ** 0.5, base_sq**0.5
    mx = max(p[2] for p in parts)
    return TensorDiff(name, int(b.size), changed, l2, mx, l2 / base if base else float("inf"))


class WeightSpace:
    """Named float tensors with edit tracking, checkpoints and patches."""

    def __init__(self, store: Store, *, on_sync: Callable[[], None] | None = None):
        self._store = store
        self._original: dict[str, np.ndarray] = {}
        self._checkpoints: dict[str, dict[str, np.ndarray | tuple[np.ndarray, np.ndarray]]] = {}
        self._on_sync = on_sync
        self._names = list(store.names())
        self._name_set = frozenset(self._names)
        self._base_sq: dict[str, float] = {}  # squared L2 norm of each original, computed once
        self._orig_fp: dict[str, tuple] = {}
        self._describe_cache: dict[str, tuple[tuple | None, str]] = {}
        self.initialization_info: Mapping = {}
        self.R = _ReadOnly(self)
        self.W = _Writable(self)

    # construction -----------------------------------------------------------------
    @classmethod
    def from_numpy(cls, tensors: Mapping[str, np.ndarray]) -> WeightSpace:
        return cls(NumpyStore(tensors))

    @classmethod
    def from_torch(cls, module, **kw) -> WeightSpace:
        from .adapters import TorchStore

        return cls(TorchStore(module, **kw))

    @classmethod
    def from_hf(cls, model_or_name, **kw) -> WeightSpace:
        from .adapters import load_hf

        return cls.from_torch(load_hf(model_or_name, **kw))

    @classmethod
    def from_file(cls, path: str, *, model_type: str | None = None, allow_pickle: bool = False, pool=None) -> WeightSpace:
        from .adapters import load_file, load_tree

        if model_type is not None:
            return load_tree(path, model_type=model_type, allow_pickle=allow_pickle, pool=pool)
        return cls(NumpyStore(load_file(path)))

    @classmethod
    def from_sklearn(cls, model) -> WeightSpace:
        from .trees.forest import ForestStore

        return cls(ForestStore(model))

    @classmethod
    def from_xgboost(cls, model) -> WeightSpace:
        from .trees.xgboost import XGBoostStore

        return cls(XGBoostStore(model))

    @classmethod
    def from_catboost(cls, model, *, pool=None) -> WeightSpace:
        from .trees.catboost import CatBoostStore

        return cls(CatBoostStore(model, pool=pool))

    @property
    def model_info(self) -> Mapping:
        """Immutable tree structure and output semantics; empty for tensor stores."""
        return self._store.model_info

    def describe_tree(self, tree_id: int, *, offset: int = 0, limit: int = 100):
        return self._store.describe_tree(tree_id, offset=offset, limit=limit)

    # reading ----------------------------------------------------------------------
    def names(self, pattern: str | None = None) -> list[str]:
        if pattern:
            return [n for n in self._names if fnmatch.fnmatch(n, pattern) or pattern in n]
        return list(self._names)

    def shape(self, name: str) -> tuple[int, ...]:
        return self._store.shape(name)

    def dtype(self, name: str) -> str:
        return self._store.dtype(name)

    def numel(self, name: str | None = None) -> int:
        if name is not None:
            return int(np.prod(self.shape(name), dtype=np.int64))
        return sum(self.numel(n) for n in self._names)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.R[name]

    def __contains__(self, name: str) -> bool:
        return name in self._name_set

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def original(self, name: str) -> np.ndarray:
        if name in self._original:
            return self._original[name]
        return self.R[name]

    def describe(self, name: str) -> str:
        """Text summary of one tensor (see `weightspace.describe`), cached by content."""
        from .render import describe

        if name not in self:
            raise KeyError(f"unknown tensor {name!r}")
        key = self._content_key(name)
        hit = self._describe_cache.get(name)
        if hit is not None and hit[0] == key:
            return hit[1]
        out = describe(name, self.R[name], self.dtype(name))
        self._describe_cache[name] = (key, out)
        return out

    def _content_key(self, name: str) -> tuple | None:
        """None while the tensor holds its original values, else a fingerprint of the current ones."""
        if name not in self._original:
            return None
        cur = _chunks.fingerprint(self._store.get(name))
        orig = self._orig_fp.get(name)
        if orig is None:
            orig = self._orig_fp[name] = _chunks.fingerprint(self._original[name])
        return None if cur == orig else cur

    # writing ----------------------------------------------------------------------
    def _snapshot(self, name: str) -> None:
        if name not in self._original:
            if name not in self:
                raise KeyError(f"unknown tensor {name!r}")
            self._original[name] = _chunks.copy(self._store.get(name))

    def edit(self, name: str) -> np.ndarray:
        """Return the live, writable array for `name` (in-place edits are tracked)."""
        self._snapshot(name)
        return self._store.get(name)

    def set(self, name: str, value) -> None:
        arr = self.edit(name)
        value = np.asarray(value, dtype=arr.dtype)
        if value.shape != arr.shape:
            value = np.broadcast_to(value, arr.shape)
        arr[...] = value

    def zero_(self, pattern: str | None = None) -> None:
        for n in self.names(pattern):
            self.edit(n)[...] = 0

    def touched(self) -> list[str]:
        return list(self._original)

    def sync(self) -> None:
        """Write edits back into the wrapped model (before a forward pass)."""
        self._store.flush(self.touched())
        if self._on_sync:
            self._on_sync()

    def initialize_for_write(self) -> Mapping:
        """Initialize editable values, synchronize, and accept the initialized baseline."""
        for n in self.names():
            self._snapshot(n)
        self.initialization_info = self._store.initialize_for_write()
        self.sync()
        self.commit()
        return self.initialization_info

    # diff / rollback ----------------------------------------------------------------
    def diff(self) -> DiffSummary:
        out = DiffSummary(total_params=self.numel(), changed_params=0)
        for name, orig in list(self._original.items()):
            t = _tensor_diff(name, self._store.get(name), orig, self._base_norm(name))
            if t is not None:
                out.tensors.append(t)
                out.changed_params += t.changed
        return out

    def _base_norm(self, name: str) -> float:
        sq = self._base_sq.get(name)
        if sq is None:
            sq = self._base_sq[name] = _chunks.sum_squares(self._original[name])
        return sq

    def revert(self, name: str | None = None) -> None:
        names = [name] if name else list(self._original)
        for n in names:
            if n in self._original:
                _chunks.copy_into(self._store.get(n), self._original[n])

    def checkpoint(self, label: str) -> None:
        """Save the current weights. Stored as a sparse delta from the originals (dense when most weights changed)."""
        saved: dict[str, np.ndarray | tuple[np.ndarray, np.ndarray]] = {}
        for n, orig in self._original.items():
            cur = self._store.get(n)
            idx = _chunks.changed_indices(cur, orig)
            if 3 * idx.size > cur.size:
                saved[n] = _chunks.copy(cur)
            elif idx.size:
                saved[n] = (idx, cur.reshape(-1)[idx])
        self._checkpoints[label] = saved

    def restore(self, label: str) -> None:
        if label not in self._checkpoints:
            raise KeyError(f"no checkpoint {label!r}; have {list(self._checkpoints)}")
        saved = self._checkpoints[label]
        for n, orig in self._original.items():
            arr, s = self._store.get(n), saved.get(n)
            if isinstance(s, np.ndarray):
                _chunks.copy_into(arr, s)
                continue
            _chunks.copy_into(arr, orig)
            if s is not None:
                arr.put(s[0], s[1])

    def checkpoints(self) -> list[str]:
        return list(self._checkpoints)

    def commit(self) -> None:
        """Accept current weights as the new baseline."""
        self._original.clear()
        self._base_sq.clear()
        self._orig_fp.clear()
        self._checkpoints.clear()

    # patches ----------------------------------------------------------------------
    def patch(self) -> dict[str, np.ndarray]:
        """Sparse patch: `<name>::idx` flat indices and `<name>::val` new values."""
        out: dict[str, np.ndarray] = {}
        metadata = self._store.patch_metadata({n: self.original(n) for n in self.names()}) if self.model_info else None
        if metadata is not None:
            out["__weightspace_tree_patch__"] = np.asarray(json.dumps(metadata, sort_keys=True))
        for name, orig in list(self._original.items()):
            cur = self._store.get(name)
            idx = _chunks.changed_indices(cur, orig)
            if idx.size:
                out[f"{name}::idx"] = idx
                out[f"{name}::val"] = cur.reshape(-1)[idx]
        return out

    def save_patch(self, path: str) -> None:
        np.savez_compressed(path, **self.patch())  # type: ignore[arg-type]

    def apply_patch(self, patch: str | Mapping[str, np.ndarray]) -> None:
        if isinstance(patch, str):
            with np.load(patch) as f:
                patch = {k: f[k] for k in f.files}
        metadata = self._store.patch_metadata({n: self._store.get(n) for n in self.names()}) if self.model_info else None
        if metadata is not None:
            self._apply_tree_patch(patch, metadata)
            return
        for key in patch:
            if not key.endswith("::idx"):
                continue
            name = key[: -len("::idx")]
            self.edit(name).put(patch[key], patch[f"{name}::val"])

    def _apply_tree_patch(self, patch, metadata):
        key = "__weightspace_tree_patch__"
        try:
            supplied = json.loads(str(patch[key].item()))
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            raise ValueError("tree patches require versioned compatibility metadata") from e
        if supplied != metadata:
            raise ValueError("tree patch backend, schema, topology, or baseline mismatch")
        proposed = self.to_numpy()
        edited = set()
        for entry in patch:
            if entry == key:
                continue
            name, sep, suffix = entry.rpartition("::")
            if not sep or name not in proposed or suffix not in ("idx", "val"):
                raise ValueError(f"invalid tree patch entry {entry!r}")
            if f"{name}::idx" not in patch or f"{name}::val" not in patch:
                raise ValueError(f"incomplete patch for {name}")
            edited.add(name)
        for name in edited:
            idx, val = np.asarray(patch[f"{name}::idx"]), np.asarray(patch[f"{name}::val"])
            if (
                idx.ndim != 1
                or val.ndim != 1
                or idx.shape != val.shape
                or idx.dtype.kind not in "iu"
                or np.any(idx < 0)
                or np.any(idx >= proposed[name].size)
                or np.unique(idx).size != idx.size
                or val.dtype != proposed[name].dtype
            ):
                raise ValueError(f"invalid indices, values, or dtype for {name}")
            proposed[name].put(idx, val)
        self._store.validate(proposed)
        for name in edited:
            self.set(name, proposed[name])

    def to_numpy(self) -> dict[str, np.ndarray]:
        return {n: self._store.get(n).copy() for n in self.names()}

    def save(self, path: str) -> None:
        from .adapters import save_file

        save_file(self.to_numpy(), path)

    def save_model(self, path: str) -> None:
        """Synchronize and export a predictive tree model in its backend's native format."""
        self.sync()
        self._store.save_model(path)

    def __repr__(self) -> str:
        return f"WeightSpace({len(self)} tensors, {self.numel():,} weights)"
