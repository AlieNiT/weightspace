import numpy as np
import pytest

import weightspace as wsp
import weightspace._chunks as chunks
import weightspace.render as render


@pytest.fixture
def threaded(monkeypatch):
    """Force the block/thread-pool paths on small arrays."""
    monkeypatch.setattr(chunks, "CHUNK", 97)
    monkeypatch.setattr(chunks, "PARALLEL_MIN", 0)
    monkeypatch.setattr(render, "EXACT_QUANTILE_MAX", 0)


@pytest.mark.parametrize(
    "a",
    [
        np.random.default_rng(0).normal(size=5000),
        np.where(np.random.default_rng(1).random(5000) < 0.7, 0, 1.5),
        np.round(np.random.default_rng(2).normal(size=4999), 1),
        1 + 1e-4 * np.random.default_rng(3).normal(size=3001),
    ],
)
def test_histogram_quantiles_are_exact(threaded, a):
    a = a.astype(np.float32)
    qs = [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0]
    assert render.quantiles(a, qs) == [float(v) for v in np.quantile(a.astype(np.float64), qs)]


def test_blocked_stats_match_numpy(threaded):
    a = np.random.default_rng(4).normal(3, 0.01, size=(70, 50)).astype(np.float32)
    s, a64 = render.stats(a), a.astype(np.float64)
    assert s["mean"] == pytest.approx(a64.mean(), rel=1e-12) and s["std"] == pytest.approx(a64.std(), rel=1e-9)
    rows, cols = render.norms(a)
    np.testing.assert_allclose(rows, np.linalg.norm(a64, axis=1), rtol=1e-12)
    np.testing.assert_allclose(cols, np.linalg.norm(a64, axis=0), rtol=1e-12)


def test_large_singular_values_match_svd():
    m = np.random.default_rng(5).normal(size=(600, 300)).astype(np.float32)
    ref = np.linalg.svd(m.astype(np.float64), compute_uv=False)
    np.testing.assert_allclose(render.singular_values(m)[:8], ref[:8], rtol=1e-5)


def test_diff_and_patch_are_bitwise(threaded):
    a = np.arange(1000, dtype=np.float32)
    a[7] = np.nan
    ws = wsp.WeightSpace.from_numpy({"a": a})
    ws.edit("a")[500] += 1  # touching keeps the NaN, which must not count as an edit
    d = ws.diff()
    assert d.changed_params == 1 and d.tensors[0].max_abs_delta == 1
    assert ws.patch()["a::idx"].tolist() == [500]


def test_sparse_and_dense_checkpoints(threaded):
    ws = wsp.WeightSpace.from_numpy({"a": np.zeros(1000, np.float32), "b": np.zeros(10, np.float32)})
    ws.W["a"][3] = 1
    ws.W["b"][:] = 2
    ws.checkpoint("c")
    assert isinstance(ws._checkpoints["c"]["a"], tuple) and isinstance(ws._checkpoints["c"]["b"], np.ndarray)
    ws.W["a"][:] = 5
    ws.W["b"][:] = 5
    ws.restore("c")
    assert ws["a"][3] == 1 and ws["a"].sum() == 1 and ws["b"].tolist() == [2] * 10


def test_describe_cache_follows_content():
    ws = wsp.WeightSpace.from_numpy({"a": np.random.default_rng(6).normal(size=(8, 8)).astype(np.float32)})
    first = ws.describe("a")
    arr = ws.W["a"]  # touched, still unchanged: cached text is reused
    assert ws.describe("a") is first
    arr[0, 0] += 10  # edited in place through a held reference
    edited = ws.describe("a")
    assert edited != first and ws.describe("a") is edited
    ws.revert()
    assert ws.describe("a") == first


def test_apply_patch_non_contiguous_store():
    ws = wsp.WeightSpace.from_numpy({"w": np.zeros((4, 3), np.float32).T})  # (3, 4) transposed view
    ws.apply_patch({"w::idx": np.array([6]), "w::val": np.array([9.0], np.float32)})
    assert ws["w"][1, 2] == 9 and ws["w"].sum() == 9
