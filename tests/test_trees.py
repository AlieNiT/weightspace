"""Tree-only tests: fitting happens exclusively in deterministic fixture setup."""

import copy
import json

import numpy as np
import pytest

import weightspace as wsp
from weightspace.agent import Session
from weightspace.cli import main
from weightspace.llm import ScriptedLLM, call, reply
from weightspace.tasks import as_weightspace


@pytest.fixture(params=["random-forest", "xgboost", "catboost"])
def case(request):
    x = np.array([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float32)
    y = np.array([-3.0, -3.0, 5.0, 5.0])
    if request.param == "random-forest":
        sk = pytest.importorskip("sklearn.ensemble")
        model = sk.RandomForestRegressor(n_estimators=2, max_depth=1, bootstrap=False, random_state=0).fit(x, y)
        ws = wsp.WeightSpace.from_sklearn(model)
        ext = ".joblib"
    elif request.param == "xgboost":
        xgb = pytest.importorskip("xgboost")
        model = xgb.XGBRegressor(n_estimators=2, max_depth=1, learning_rate=0.5, n_jobs=1).fit(x, y)
        ws = wsp.WeightSpace.from_xgboost(model)
        ext = ".json"
    else:
        cb = pytest.importorskip("catboost")
        model = cb.CatBoostRegressor(
            iterations=2, depth=1, learning_rate=0.5, verbose=False, thread_count=1, allow_writing_files=False
        ).fit(x, y)
        ws = wsp.WeightSpace.from_catboost(model)
        ext = ".cbm"
    return request.param, model, ws, x, ext


def test_parity_noop_and_metadata(case, monkeypatch):
    _, model, ws, x, _ = case
    before = model.predict(x)
    np.testing.assert_array_equal(as_weightspace(model)["trees.0.leaf_values"], ws["trees.0.leaf_values"])
    with pytest.raises(TypeError):
        ws.model_info["backend"] = "changed"
    with pytest.raises(TypeError):
        ws.describe_tree(0)["nodes"][0]["x"] = 1
    with pytest.raises(ValueError):
        ws.describe_tree(0, limit=101)
    with pytest.raises(ValueError):
        ws.describe_tree(-1)
    with pytest.raises(ValueError):
        ws["trees.0.leaf_values"][0, 0] = 0
    monkeypatch.setattr(ws._store, "_install", lambda: pytest.fail("unchanged sync must be a no-op"))
    ws.sync()
    ws.edit("trees.0.leaf_values")
    ws.sync()
    np.testing.assert_array_equal(model.predict(x), before)


def test_leaf_edit_tracking_and_native_export(case, tmp_path):
    backend, model, ws, x, ext = case
    before = model.predict(x)
    ws.checkpoint("initial")
    name = "trees.0.leaf_values"
    other = ws["trees.1.leaf_values"].copy()
    ws.W[name][:] += 2
    ws.checkpoint("edited")
    assert ws.diff().changed_params == ws.numel(name)
    ws.sync()
    np.testing.assert_allclose(model.predict(x), before + (1 if backend == "random-forest" else 2), atol=1e-6)
    np.testing.assert_array_equal(ws["trees.1.leaf_values"], other)
    ws.restore("initial")
    ws.sync()
    np.testing.assert_allclose(model.predict(x), before)
    ws.restore("edited")
    path = str(tmp_path / ("model" + ext))
    ws.save_model(path)
    if backend == "random-forest":
        with pytest.raises(ValueError, match="trusted"):
            wsp.WeightSpace.from_file(path, model_type=backend)
    loaded = wsp.WeightSpace.from_file(path, model_type=backend, allow_pickle=True)
    native = loaded._store._model
    if backend == "xgboost":
        import xgboost

        pred = native.predict(xgboost.DMatrix(x))
    else:
        pred = native.predict(x)
    np.testing.assert_allclose(pred, model.predict(x))
    ws.revert()
    ws.sync()
    np.testing.assert_allclose(model.predict(x), before)
    ws.commit()
    assert ws.diff().changed_params == 0


def test_atomic_validation_and_patch(case):
    _, model, ws, x, _ = case
    before = model.predict(x)
    target = copy.deepcopy(model)
    other = as_weightspace(target)
    name = "trees.0.leaf_values"
    ws.W[name][:] += 1
    patch = ws.patch()
    broken = dict(patch)
    broken[f"{name}::val"] = np.full_like(patch[f"{name}::val"], np.nan)
    with pytest.raises(ValueError, match="finite"):
        other.apply_patch(broken)
    assert other.diff().changed_params == 0
    with pytest.raises(ValueError, match="metadata"):
        other.apply_patch({k: v for k, v in patch.items() if not k.startswith("__")})
    broken = dict(patch)
    broken[f"{name}::idx"] = np.array([-1] * len(patch[f"{name}::idx"]))
    with pytest.raises(ValueError, match="indices"):
        other.apply_patch(broken)
    assert other.diff().changed_params == 0
    other.apply_patch(patch)
    other.sync()
    ws.sync()
    np.testing.assert_allclose(target.predict(x), model.predict(x))
    with pytest.raises(ValueError, match="mismatch"):
        other.apply_patch(patch)
    ws.W[name][0, 0] = np.inf
    installed = model.predict(x)
    with pytest.raises(ValueError, match="finite"):
        ws.sync()
    np.testing.assert_array_equal(model.predict(x), installed)
    ws.revert()
    ws.sync()
    np.testing.assert_allclose(model.predict(x), before)


@pytest.mark.parametrize("mode", ["read", "fix", "write", "improve"])
def test_tasks_and_original_object_probe(case, mode):
    _, model, ws, x, _ = case
    observed = []

    def predictions():
        observed.append(model.predict(x).copy())
        return observed[-1].tolist()

    calls = [] if mode == "read" else [call("python", code="W['trees.0.leaf_values'][:] += 1")]
    llm = ScriptedLLM(
        [
            reply(None, call("describe_tree", tree_id=0), *calls),
            reply(None, call("forward_predictions")),
            reply(None, call("forward_predictions")),
            reply(None, call("submit", answer="done")),
        ]
    )
    res = getattr(wsp, mode)(ws, "inspect or edit", llm, probes=[predictions], forward_budget=1)
    assert res.ok and res.forward_used == 1 and len(observed) == 1
    np.testing.assert_allclose(observed[0], model.predict(x))
    assert (res.diff.changed_params == 0) == (mode == "read")
    assert any("budget exhausted" in (m.get("content") or "") for m in res.transcript)
    if mode == "read":
        s = Session(ws, ScriptedLLM([]), mode="read")
        assert "write_tensor" not in s.tools and "W" not in s.sandbox.ns
        assert "_model" not in s.sandbox.ns and "pool" not in s.sandbox.ns


def test_invalid_probe_finalization_and_correction(case):
    _, model, ws, x, _ = case
    before = model.predict(x)
    probes = []

    def observe():
        probes.append(True)
        return "called"

    script = [
        reply(None, call("python", code="W['trees.0.leaf_values'][0, 0] = np.nan")),
        reply(None, call("forward_observe")),
        reply(None, call("submit", answer="invalid")),
    ]
    res = wsp.fix(ws, "invalid", ScriptedLLM(script), probes=[observe], forward_budget=1)
    assert res.stop_reason == "error" and not res.ok and "finite" in res.error
    assert res.transcript and not probes and res.forward_used == 0
    np.testing.assert_array_equal(before, model.predict(x))
    ws.revert()
    res = wsp.fix(ws, "correct", ScriptedLLM([reply(None, call("submit", answer="fixed"))]))
    assert res.ok


def test_text_mode_and_write_initialization(case):
    _, model, ws, x, _ = case
    baseline = ws.to_numpy()
    result = wsp.write(ws, "keep", ScriptedLLM([reply("ANSWER: unchanged")]), text_mode=True, zero_init=False)
    assert result.ok
    for n, a in baseline.items():
        np.testing.assert_array_equal(ws[n], a)
    info = ws.initialize_for_write()
    assert ws.diff().changed_params == 0
    for n in ws.names():
        if n not in info.get("preserved_border_groups", []):
            assert np.all(ws[n] == 0)
    s = Session(ws, ScriptedLLM([reply("```python\nW['trees.0.leaf_values'][:] = 3\n```\nANSWER: set")]), mode="fix")
    res = s.run_text("edit")
    assert res.ok and res.diff.changed_params == ws.numel("trees.0.leaf_values")
    user = res.transcript[1]["content"]
    assert "leaf_index" in user or "leaf_order" in user
    custom = Session(ws, ScriptedLLM([]), system_prompt="custom")
    assert custom.system_prompt == "custom"
    with pytest.raises(ValueError, match="chars"):
        Session(ws, ScriptedLLM([])).run_text("too big", max_chars=10)


def test_forest_class_probabilities_threshold_and_cli(tmp_path, capsys):
    sk = pytest.importorskip("sklearn.ensemble")
    x = np.array([[-2], [-1], [1], [2]], dtype=np.float32)
    m = sk.RandomForestClassifier(n_estimators=1, max_depth=1, bootstrap=False, random_state=1).fit(x, ["a", "a", "b", "b"])
    ws = wsp.WeightSpace.from_sklearn(m)
    p = m.predict_proba(x)
    ws.sync()
    np.testing.assert_array_equal(m.predict_proba(x), p)
    name = "trees.0.leaf_values"
    ws.set(name, [[0.25, 0.75], [0.9, 0.1]])
    ws.sync()
    np.testing.assert_allclose(m.predict_proba(x), [[0.25, 0.75]] * 2 + [[0.9, 0.1]] * 2)
    ws.set("trees.0.thresholds", -1.5)
    ws.sync()
    np.testing.assert_allclose(m.predict_proba(x)[1], [0.9, 0.1])
    before = m.predict_proba(x)
    for bad in ([[1, 1], [0, 1]], [[-0.1, 1.1], [0, 1]]):
        ws.set(name, bad)
        with pytest.raises(ValueError, match="probabilities"):
            ws.sync()
        np.testing.assert_array_equal(m.predict_proba(x), before)
    ws.initialize_for_write()
    np.testing.assert_allclose(m.predict_proba(x), 0.5)
    path = str(tmp_path / "forest.joblib")
    ws.save_model(path)
    assert main(["inspect", path, "--model-type", "random-forest", "--allow-pickle"]) == 0
    assert "leaf_values" in capsys.readouterr().out


def test_forest_rejections_and_multiclass():
    sk = pytest.importorskip("sklearn.ensemble")
    with pytest.raises(ValueError, match="fitted"):
        wsp.WeightSpace.from_sklearn(sk.RandomForestClassifier())
    x = np.arange(12).reshape(-1, 1)
    model = sk.RandomForestClassifier(n_estimators=1, random_state=0).fit(x, np.arange(12) % 3)
    ws = wsp.WeightSpace.from_sklearn(model)
    assert ws.model_info["classes"] == (0, 1, 2)
    ws.initialize_for_write()
    np.testing.assert_allclose(model.predict_proba(x), 1 / 3)
    model = sk.RandomForestRegressor(n_estimators=1).fit(x, np.ones((12, 2)))
    with pytest.raises(ValueError, match="multi-output"):
        wsp.WeightSpace.from_sklearn(model)


def test_patch_shape_and_topology_mismatch(case):
    _, model, ws, _, _ = case
    other = as_weightspace(copy.deepcopy(model))
    ws.W["trees.0.leaf_values"][:] += 1
    p = ws.patch()
    meta = json.loads(str(p["__weightspace_tree_patch__"]))
    meta["structure"] = "wrong"
    p["__weightspace_tree_patch__"] = np.asarray(json.dumps(meta))
    with pytest.raises(ValueError, match="mismatch"):
        other.apply_patch(p)
    assert other.touched() == []
    a = ws.edit("trees.0.leaf_values")
    shape = a.shape
    a.shape = (a.size,)
    with pytest.raises(ValueError, match="shape"):
        ws.sync()
    a.shape = shape


def test_malformed_staging_retains_transcript(case):
    _, model, ws, x, _ = case
    before = model.predict(x)
    name = "trees.0.leaf_values"
    a = ws.edit(name)
    original = a.copy()
    a.dtype = np.uint8
    llm = ScriptedLLM([reply(None, call("submit", answer="invalid shape"))])
    res = wsp.fix(ws, "inspect malformed staging", llm)
    assert not res.ok and res.transcript and "shape" in res.error
    assert "diff unavailable" in res.error
    np.testing.assert_array_equal(model.predict(x), before)
    a.dtype = original.dtype
    a.shape = original.shape
    a[...] = original


def test_known_threshold_routes(case):
    backend, model, ws, _, _ = case
    ws.set("trees.0.leaf_values", [[-7], [13]])
    ws.set("trees.1.leaf_values", 0)
    if backend == "catboost":
        # CatBoost regression retains its fitted bias of one.
        name = ws.names("borders")[0]
        border_index = ws.describe_tree(0)["nodes"][0]["border_index"]
        ws.set(name, 0.5 + 2 * (np.arange(ws.numel(name)) - border_index))
        offset = model.get_scale_and_bias()[1]
        expected = np.array([-7, -7, 13]) + offset
    else:
        ws.set("trees.0.thresholds", 0.5)
        offset = 0 if backend == "random-forest" else 1  # native XGBoost base score
        expected = np.array([-7, 13 if backend == "xgboost" else -7, 13]) + offset
        if backend == "random-forest":
            expected = expected / 2
    ws.sync()
    points = np.array([[0], [0.5], [1]], dtype=np.float32)
    np.testing.assert_allclose(model.predict(points), expected, atol=1e-6)


def test_cli_write_and_predictive_export(case, tmp_path, capsys):
    backend, _, ws, _, ext = case
    path = str(tmp_path / ("input" + ext))
    native = str(tmp_path / ("output" + ext))
    weights = str(tmp_path / "weights.npz")
    ws.save_model(path)
    transcript = tmp_path / "run.jsonl"
    transcript.write_text(json.dumps({"role": "assistant", "content": "ANSWER: initialized"}) + "\n")
    args = [
        "write",
        path,
        "initialize the fitted structure",
        "--model-type",
        backend,
        "--allow-pickle",
        "--replay",
        str(transcript),
        "--text-mode",
        "--save-model",
        native,
        "--save-weights",
        weights,
    ]
    assert main(args) == 0
    assert "initialized" in capsys.readouterr().out
    loaded = wsp.WeightSpace.from_file(native, model_type=backend, allow_pickle=True)
    assert np.all(loaded["trees.0.leaf_values"] == 0)
    with np.load(weights) as arrays:
        assert np.all(arrays["trees.0.leaf_values"] == 0)


@pytest.mark.parametrize("case", ["random-forest"], indirect=True)
def test_forest_preserves_training_statistics(case):
    _, model, ws, _, _ = case
    native = model.estimators_[0].tree_
    leaves = native.children_left == -1
    internal_values = native.value[~leaves].copy()
    counts = native.weighted_n_node_samples.copy()
    impurity = native.impurity.copy()
    ws.W["trees.0.leaf_values"][:] += 4
    ws.sync()
    np.testing.assert_array_equal(native.value[~leaves], internal_values)
    np.testing.assert_array_equal(native.weighted_n_node_samples, counts)
    np.testing.assert_array_equal(native.impurity, impurity)


def test_native_baseline_patch_roundtrip(case, tmp_path):
    backend, _, ws, _, ext = case
    path = str(tmp_path / ("baseline" + ext))
    ws.save_model(path)
    loaded = wsp.WeightSpace.from_file(path, model_type=backend, allow_pickle=True)
    ws.W["trees.0.leaf_values"][:] += 1
    patch = str(tmp_path / "edit.npz")
    ws.save_patch(patch)
    loaded.apply_patch(patch)
    assert loaded.diff().changed_params == ws.numel("trees.0.leaf_values")
    for n in ws.names():
        np.testing.assert_array_equal(ws[n], loaded[n])


@pytest.fixture(params=["random-forest", "xgboost"])
def missing_only_case(request):
    # Both backends can represent missing-only splits using an infinite threshold.
    x = np.array([[0], [np.finfo(np.float32).max], [np.nan], [np.nan]], dtype=np.float32)
    if request.param == "random-forest":
        sk = pytest.importorskip("sklearn.ensemble")
        m = sk.RandomForestRegressor(n_estimators=1, bootstrap=False, random_state=0).fit(x, [0, 0, 3, 3])
    else:
        xgb = pytest.importorskip("xgboost")
        m = xgb.XGBRegressor(n_estimators=1, max_depth=1, n_jobs=1).fit(x, [0, 0, 3, 3])
    return m, as_weightspace(m), x


def test_reserved_missing_threshold_is_immutable(missing_only_case, tmp_path):
    model, ws, x = missing_only_case
    before = model.predict(x)
    assert ws.numel("trees.0.thresholds") == 0
    assert ws.describe_tree(0)["nodes"][0]["fixed_threshold"] == "inf"
    ws.sync()
    np.testing.assert_array_equal(model.predict(x), before)
    ws.W["trees.0.leaf_values"][:] += 1
    ws.sync()
    np.testing.assert_allclose(model.predict(x), before + 1, atol=1e-6)
    ws.save_patch(str(tmp_path / "missing-patch.npz"))
