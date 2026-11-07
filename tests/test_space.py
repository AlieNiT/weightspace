import numpy as np
import pytest

from weightspace import WeightSpace


def test_edits_are_tracked_and_revertable(arrays):
    ws = WeightSpace.from_numpy(arrays)
    assert ws.numel() == 16
    assert ws.diff().changed_params == 0
    ws.W["a.weight"][0, :] = 0
    ws.set("a.bias", 1.0)
    d = ws.diff()
    assert d.changed_params == 3 + 4
    assert {t.name for t in d.tensors} == {"a.weight", "a.bias"}
    ws.revert("a.bias")
    assert ws.diff().changed_params == 3
    ws.revert()
    assert ws.diff().changed_params == 0


def test_read_only_view(arrays):
    ws = WeightSpace.from_numpy(arrays)
    with pytest.raises(ValueError):
        ws.R["a.weight"][0, 0] = 1
    assert ws.touched() == []


def test_checkpoints(arrays):
    ws = WeightSpace.from_numpy(arrays)
    ws.W["a.weight"][0, 0] = 5
    ws.checkpoint("one")
    ws.W["a.weight"][1, 1] = 7
    ws.W["a.bias"][0] = 9  # touched after the checkpoint
    ws.restore("one")
    assert ws["a.weight"][0, 0] == 5
    assert ws["a.weight"][1, 1] == arrays["a.weight"][1, 1]
    assert ws["a.bias"][0] == 0


def test_patch_roundtrip(arrays, tmp_path):
    ws = WeightSpace.from_numpy({k: v.copy() for k, v in arrays.items()})
    ws.W["a.weight"][2, 1] = 3.5
    path = str(tmp_path / "p.npz")
    ws.save_patch(path)
    other = WeightSpace.from_numpy({k: v.copy() for k, v in arrays.items()})
    other.apply_patch(path)
    np.testing.assert_array_equal(other["a.weight"], ws["a.weight"])
    assert other.diff().changed_params == 1


def test_torch_float32_is_live():
    torch = pytest.importorskip("torch")
    m = torch.nn.Linear(3, 2)
    ws = WeightSpace.from_torch(m)
    ws.W["weight"][:] = 0
    assert torch.all(m.weight == 0)


def test_torch_bf16_is_mirrored_and_flushed():
    torch = pytest.importorskip("torch")
    m = torch.nn.Linear(3, 2).to(torch.bfloat16)
    ws = WeightSpace.from_torch(m)
    assert ws.dtype("weight") == "bfloat16"
    ws.W["weight"][:] = 0.5
    assert not torch.all(m.weight == 0.5)
    ws.sync()
    assert torch.all(m.weight == 0.5)


def test_files(arrays, tmp_path):
    for ext in ("npz", "safetensors"):
        p = str(tmp_path / f"w.{ext}")
        WeightSpace.from_numpy(arrays).save(p)
        ws = WeightSpace.from_file(p)
        np.testing.assert_array_equal(ws["a.weight"], arrays["a.weight"])
