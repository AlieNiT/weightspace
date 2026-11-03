"""A model-agnostic view of a network's weights as named numpy arrays.

The LLM never sees the model object, only tensors. Edits are tracked against
the original weights so we can report how much changed, roll back, and export
a sparse patch.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field

import numpy as np


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


class WeightSpace:
    """Named float tensors with edit tracking, checkpoints and patches."""

    def __init__(self, store: Store, *, on_sync: Callable[[], None] | None = None):
        self._store = store
        self._original: dict[str, np.ndarray] = {}
        self._checkpoints: dict[str, dict[str, np.ndarray]] = {}
        self._on_sync = on_sync
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
    def from_file(cls, path: str) -> WeightSpace:
        from .adapters import load_file

        return cls(NumpyStore(load_file(path)))

    # reading ----------------------------------------------------------------------
    def names(self, pattern: str | None = None) -> list[str]:
        names = self._store.names()
        if pattern:
            names = [n for n in names if fnmatch.fnmatch(n, pattern) or pattern in n]
        return names

    def shape(self, name: str) -> tuple[int, ...]:
        return self._store.shape(name)

    def dtype(self, name: str) -> str:
        return self._store.dtype(name)

    def numel(self, name: str | None = None) -> int:
        if name is not None:
            return int(np.prod(self.shape(name), dtype=np.int64))
        return sum(self.numel(n) for n in self.names())

    def __getitem__(self, name: str) -> np.ndarray:
        return self.R[name]

    def __contains__(self, name: str) -> bool:
        return name in self._store.names()

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self.names())

    def original(self, name: str) -> np.ndarray:
        if name in self._original:
            return self._original[name]
        return self.R[name]

    # writing ----------------------------------------------------------------------
    def _snapshot(self, name: str) -> None:
        if name not in self._original:
            if name not in self:
                raise KeyError(f"unknown tensor {name!r}")
            self._original[name] = self._store.get(name).copy()

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

    # diff / rollback ----------------------------------------------------------------
    def diff(self) -> DiffSummary:
        out = DiffSummary(total_params=self.numel(), changed_params=0)
        for name, orig in self._original.items():
            cur = self._store.get(name)
            delta = cur.astype(np.float64) - orig.astype(np.float64)
            changed = int(np.count_nonzero(delta))
            if not changed:
                continue
            l2 = float(np.linalg.norm(delta))
            base = float(np.linalg.norm(orig.astype(np.float64)))
            out.tensors.append(
                TensorDiff(
                    name=name,
                    numel=int(orig.size),
                    changed=changed,
                    l2_delta=l2,
                    max_abs_delta=float(np.abs(delta).max()),
                    rel_delta=l2 / base if base else float("inf"),
                )
            )
            out.changed_params += changed
        return out

    def revert(self, name: str | None = None) -> None:
        names = [name] if name else list(self._original)
        for n in names:
            if n in self._original:
                self._store.get(n)[...] = self._original[n]

    def checkpoint(self, label: str) -> None:
        self._checkpoints[label] = {n: self._store.get(n).copy() for n in self._original}

    def restore(self, label: str) -> None:
        if label not in self._checkpoints:
            raise KeyError(f"no checkpoint {label!r}; have {list(self._checkpoints)}")
        saved = self._checkpoints[label]
        for n in self._original:
            self._store.get(n)[...] = saved.get(n, self._original[n])

    def checkpoints(self) -> list[str]:
        return list(self._checkpoints)

    def commit(self) -> None:
        """Accept current weights as the new baseline."""
        self._original.clear()
        self._checkpoints.clear()

    # patches ----------------------------------------------------------------------
    def patch(self) -> dict[str, np.ndarray]:
        """Sparse patch: `<name>::idx` flat indices and `<name>::val` new values."""
        out: dict[str, np.ndarray] = {}
        for t in self.diff().tensors:
            cur = self._store.get(t.name).reshape(-1)
            idx = np.flatnonzero(cur != self._original[t.name].reshape(-1))
            out[f"{t.name}::idx"] = idx.astype(np.int64)
            out[f"{t.name}::val"] = cur[idx].copy()
        return out

    def save_patch(self, path: str) -> None:
        np.savez_compressed(path, **self.patch())

    def apply_patch(self, patch: str | Mapping[str, np.ndarray]) -> None:
        if isinstance(patch, str):
            with np.load(patch) as f:
                patch = {k: f[k] for k in f.files}
        for key in patch:
            if not key.endswith("::idx"):
                continue
            name = key[: -len("::idx")]
            arr = self.edit(name).reshape(-1)
            arr[patch[key]] = patch[f"{name}::val"]

    def to_numpy(self) -> dict[str, np.ndarray]:
        return {n: self._store.get(n).copy() for n in self.names()}

    def save(self, path: str) -> None:
        from .adapters import save_file

        save_file(self.to_numpy(), path)

    def __repr__(self) -> str:
        return f"WeightSpace({len(self)} tensors, {self.numel():,} weights)"
