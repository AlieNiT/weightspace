"""Adapters from real models and files to a `Store`."""

from __future__ import annotations

import os

import numpy as np

from .space import Store


class TorchStore(Store):
    """Live view over a torch `nn.Module`'s parameters.

    float32/float64/float16 CPU tensors share memory with numpy, so edits are
    instant. Other dtypes (bf16, GPU tensors) are mirrored as float32 and
    copied back on `flush`.
    """

    def __init__(self, module, *, include_buffers: bool = False, pattern: str | None = None):
        import torch

        self.module = module
        self._torch = torch
        tensors = dict(module.named_parameters())
        if include_buffers:
            tensors.update({k: v for k, v in module.named_buffers() if v is not None and v.is_floating_point()})
        if pattern:
            import fnmatch

            tensors = {k: v for k, v in tensors.items() if fnmatch.fnmatch(k, pattern) or pattern in k}
        self._t = tensors
        self._arr: dict[str, np.ndarray] = {}
        self._mirrored: set[str] = set()

    def names(self):
        return list(self._t)

    def shape(self, name):
        return tuple(self._t[name].shape)

    def dtype(self, name):
        return str(self._t[name].dtype).replace("torch.", "")

    def get(self, name):
        arr = self._arr.get(name)
        if arr is None:
            t = self._t[name].detach()
            if t.device.type == "cpu" and t.dtype in (
                self._torch.float32,
                self._torch.float64,
                self._torch.float16,
            ):
                arr = t.numpy()
            else:
                arr = t.to("cpu", self._torch.float32).numpy().copy()
                self._mirrored.add(name)
            self._arr[name] = arr
        return arr

    def flush(self, names):
        with self._torch.no_grad():
            for n in names:
                if n in self._mirrored:
                    p = self._t[n]
                    p.copy_(self._torch.from_numpy(self._arr[n]).to(p.device, p.dtype))


def load_hf(model_or_name, **kw):
    """Accept a loaded `transformers` model or a hub id / local path."""
    if not isinstance(model_or_name, str):
        return model_or_name
    from transformers import AutoModelForCausalLM

    kw.setdefault("torch_dtype", "auto")
    return AutoModelForCausalLM.from_pretrained(model_or_name, **kw)


def load_file(path: str) -> dict[str, np.ndarray]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        from safetensors.numpy import load_file as st_load

        try:
            return st_load(path)
        except (TypeError, ValueError):
            from safetensors.torch import load_file as st_torch_load

            return {k: v.float().numpy() for k, v in st_torch_load(path).items()}
    if ext == ".npz":
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
    if ext in (".pt", ".pth", ".bin", ".ckpt"):
        import torch

        obj = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(obj, dict) and "state_dict" in obj:
            obj = obj["state_dict"]
        return {k: v.float().numpy() for k, v in obj.items() if hasattr(v, "is_floating_point") and v.is_floating_point()}
    raise ValueError(f"unsupported weights file: {path}")


def save_file(tensors: dict[str, np.ndarray], path: str) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        from safetensors.numpy import save_file as st_save

        st_save({k: np.ascontiguousarray(v) for k, v in tensors.items()}, path)
    elif ext == ".npz":
        np.savez_compressed(path, **tensors)
    else:
        raise ValueError(f"unsupported weights file: {path}")
