"""Native categorical round trips are required, including a real edit/reload boundary."""

import copy
import json

import numpy as np
import pytest

from weightspace import WeightSpace
from weightspace.agent import Session
from weightspace.cli import main
from weightspace.llm import ScriptedLLM, reply


@pytest.fixture(params=["onehot", "partition"])
def xgb_categorical(request):
    xgb = pytest.importorskip("xgboost")
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(5)
    cat = np.tile(["a", "b", "c", "d"], 30)
    numeric = rng.normal(size=len(cat))
    x = pd.DataFrame({"kind": pd.Categorical(cat), "value": numeric})
    y = ((cat == "a") | (cat == "c")).astype(int)
    model = xgb.XGBClassifier(
        n_estimators=4,
        max_depth=2,
        n_jobs=1,
        enable_categorical=True,
        tree_method="hist",
        max_cat_to_onehot=8 if request.param == "onehot" else 1,
    ).fit(x, y)
    probe = pd.DataFrame(
        {
            "kind": pd.Categorical(["d", "c", "b", "a", None], categories=["d", "c", "b", "a"]),
            "value": [1.0, np.nan, -1, 2, 0],
        }
    )
    return xgb, model, x, probe


def test_xgboost_categories_recoding_and_missing(xgb_categorical, tmp_path):
    xgb, model, x, probe = xgb_categorical
    booster = model.get_booster()
    before = model.predict_proba(probe)
    margins = model.predict(probe, output_margin=True)
    original = json.loads(booster.save_raw(raw_format="json"))
    native_trees = original["learner"]["gradient_booster"]["model"]["trees"]
    assert any(any(t["split_type"]) for t in native_trees), "fixture must exercise native categorical splits"
    ws = WeightSpace.from_xgboost(model)
    baseline_path = str(tmp_path / "baseline.json")
    ws.save_model(baseline_path)
    baseline_model = xgb.XGBClassifier()
    baseline_model.load_model(baseline_path)
    baseline_ws = WeightSpace.from_xgboost(baseline_model)
    ws.sync()
    np.testing.assert_array_equal(model.predict_proba(probe), before)
    ws.W["trees.0.leaf_values"][:] += 0.75
    ws.sync()
    assert model.get_booster() is booster
    baseline_ws.apply_patch(ws.patch())
    baseline_ws.sync()
    np.testing.assert_array_equal(baseline_model.predict_proba(probe), model.predict_proba(probe))
    np.testing.assert_allclose(model.predict(probe, output_margin=True), margins + 0.75, atol=2e-7)
    updated = json.loads(booster.save_raw(raw_format="json"))
    for old, new in zip(native_trees, updated["learner"]["gradient_booster"]["model"]["trees"], strict=True):
        for key in ("categories", "categories_nodes", "categories_segments", "categories_sizes", "split_type", "default_left"):
            assert old[key] == new[key]
    for key in original["learner"]:
        if key != "gradient_booster":
            assert original["learner"][key] == updated["learner"][key]
    path = str(tmp_path / "categorical.ubj")
    ws.save_model(path)
    reloaded = xgb.XGBClassifier()
    reloaded.load_model(path)
    np.testing.assert_array_equal(reloaded.predict_proba(probe), model.predict_proba(probe))
    np.testing.assert_array_equal(reloaded.predict(x), model.predict(x))
    # XGBoost itself rejects unseen labels when it retains the recoding dictionary. Preserve that behavior.
    pd = pytest.importorskip("pandas")
    unseen = pd.DataFrame({"kind": pd.Categorical(["unseen"]), "value": [0.0]})
    for m in (model, reloaded):
        with pytest.raises(xgb.core.XGBoostError, match="categor"):
            m.predict(unseen)


@pytest.fixture(params=["onehot", "ctr"])
def catboost_categorical(request):
    cb = pytest.importorskip("catboost")
    x = [[str(i % 5), float(i % 7) if i % 11 else np.nan] for i in range(100)]
    y = [int(i % 5 > 1) for i in range(100)]
    pool = cb.Pool(x, y, cat_features=[0])
    model = cb.CatBoostClassifier(
        iterations=6,
        depth=3,
        random_seed=4,
        one_hot_max_size=10 if request.param == "onehot" else 2,
        verbose=False,
        thread_count=1,
        allow_writing_files=False,
    ).fit(pool)
    probe = cb.Pool([["0", 1.0], ["2", np.nan], ["unseen", 3.0], ["4", -5.0]], cat_features=[0])
    return cb, model, pool, probe, request.param


def test_catboost_categories_ctr_and_missing(catboost_categorical, tmp_path):
    cb, model, pool, probe, kind = catboost_categorical
    before = model.predict_proba(probe)
    margins = model.predict(probe, prediction_type="RawFormulaVal")
    with pytest.raises(ValueError, match="reference_pool"):
        WeightSpace.from_catboost(model)
    ws = WeightSpace.from_catboost(model, pool=pool)
    baseline_path = str(tmp_path / "baseline.cbm")
    ws.save_model(baseline_path)
    baseline_ws = WeightSpace.from_file(baseline_path, model_type="catboost", pool=pool)
    original = copy.deepcopy(ws._store._doc)
    split_type = "OneHotFeature" if kind == "onehot" else "OnlineCtr"
    assert any(s["split_type"] == split_type for t in original["oblivious_trees"] for s in t.get("splits") or [])
    ws.sync()
    np.testing.assert_array_equal(model.predict_proba(probe), before)
    ws.W["trees.0.leaf_values"][:] += 0.75
    ws.sync()
    np.testing.assert_allclose(model.predict(probe, prediction_type="RawFormulaVal"), margins + 0.75, atol=1e-12)
    baseline_ws.apply_patch(ws.patch())
    baseline_ws.sync()
    np.testing.assert_allclose(
        baseline_ws._store._model.predict(probe, prediction_type="RawFormulaVal"), margins + 0.75, atol=1e-12
    )
    path = str(tmp_path / "categorical.cbm")
    ws.save_model(path)
    loaded = cb.CatBoostClassifier().load_model(path)
    np.testing.assert_allclose(loaded.predict_proba(probe), model.predict_proba(probe), atol=1e-15)
    second = WeightSpace.from_catboost(loaded, pool=pool)
    assert original["features_info"] == second._store._doc["features_info"]
    assert original.get("ctr_data") == second._store._doc.get("ctr_data")
    assert original["scale_and_bias"] == second._store._doc["scale_and_bias"]
    s = Session(ws, ScriptedLLM([]))
    assert all(v is not pool and v is not model for v in s.sandbox.ns.values())


@pytest.fixture(params=[2, 3])
def boosted_classifier(request):
    x = np.arange(30, dtype=np.float32).reshape(-1, 1)
    y = np.arange(30) % request.param
    return x, y, request.param


@pytest.mark.parametrize("backend", ["xgboost", "catboost"])
def test_boosted_classification_dimensions(boosted_classifier, backend):
    x, y, classes = boosted_classifier
    if backend == "xgboost":
        lib = pytest.importorskip("xgboost")
        model = lib.XGBClassifier(n_estimators=2, max_depth=2, n_jobs=1).fit(x, y)
        ws = WeightSpace.from_xgboost(model)

        def raw():
            return model.predict(x, output_margin=True)
    else:
        lib = pytest.importorskip("catboost")
        model = lib.CatBoostClassifier(iterations=2, depth=2, verbose=False, thread_count=1, allow_writing_files=False).fit(x, y)
        ws = WeightSpace.from_catboost(model)

        def raw():
            return model.predict(x, prediction_type="RawFormulaVal")

    before = model.predict_proba(x)
    margins = raw()
    ws.sync()
    np.testing.assert_array_equal(before, model.predict_proba(x))
    ws.W["trees.0.leaf_values"][:, 0] += 1
    ws.sync()
    expected = margins.copy()
    if classes == 2:
        expected += 1
    else:
        expected[:, 0] += 1
    np.testing.assert_allclose(raw(), expected, atol=1e-6)


@pytest.fixture
def catboost_borders():
    cb = pytest.importorskip("catboost")
    x = np.arange(40, dtype=np.float32).reshape(-1, 1)
    model = cb.CatBoostRegressor(
        iterations=5,
        depth=2,
        verbose=False,
        thread_count=1,
        allow_writing_files=False,
        border_count=5,
    ).fit(x, x[:, 0])
    return model, WeightSpace.from_catboost(model), x


def test_shared_borders_order_and_initialization(catboost_borders):
    model, ws, x = catboost_borders
    name = ws.names("borders")[0]
    assert ws[name].size > 1
    before = model.predict(x)
    ws.W[name][:] = 0
    with pytest.raises(ValueError, match="strictly ordered"):
        ws.sync()
    np.testing.assert_array_equal(model.predict(x), before)
    ws.revert()
    borders = ws[name].copy()
    ws.W[name][:] += 0.25
    ws.sync()
    exported = WeightSpace.from_catboost(model)
    np.testing.assert_array_equal(exported[name], borders + 0.25)
    assert all(
        t["border"] in exported[name] for tree in exported._store._doc["oblivious_trees"] for t in tree.get("splits") or []
    )
    info = ws.initialize_for_write()
    assert name in info["preserved_border_groups"]
    np.testing.assert_array_equal(ws[name], borders + 0.25)


@pytest.mark.parametrize("backend", ["xgboost", "catboost"])
def test_boosted_native_failure_rolls_back(backend, monkeypatch):
    x = np.array([[-1.0], [1.0]])
    if backend == "xgboost":
        lib = pytest.importorskip("xgboost")
        m = lib.XGBRegressor(n_estimators=1, n_jobs=1).fit(x, [0, 2])
        ws = WeightSpace.from_xgboost(m)
        target = m.get_booster()
    else:
        lib = pytest.importorskip("catboost")
        m = lib.CatBoostRegressor(iterations=1, depth=1, verbose=False, thread_count=1, allow_writing_files=False).fit(x, [0, 2])
        ws = WeightSpace.from_catboost(m)
        target = m
    before = m.predict(x)
    ws.W["trees.0.leaf_values"][:] += 1
    original = target.load_model
    calls = []

    def fail_once(*a, **kw):
        calls.append(True)
        result = original(*a, **kw)
        if len(calls) == 1:
            raise RuntimeError("installation failed after loading")
        return result

    monkeypatch.setattr(target, "load_model", fail_once)
    with pytest.raises(RuntimeError, match="installation failed"):
        ws.sync()
    np.testing.assert_array_equal(m.predict(x), before)
    assert ws.diff().changed_params > 0
    ws.sync()
    np.testing.assert_allclose(m.predict(x), before + 1, atol=1e-6)


def test_cli_catboost_pool_factory(catboost_categorical, tmp_path, capsys):
    _, model, _, _, _ = catboost_categorical
    path = str(tmp_path / "m.cbm")
    model.save_model(path)
    factory = tmp_path / "pool.py"
    factory.write_text(
        "from catboost import Pool\ndef reference():\n"
        "    return Pool([[str(i % 5), float(i % 7)] for i in range(100)], cat_features=[0])\n"
    )
    assert main(["inspect", path, "--model-type", "catboost", "--catboost-pool", f"{factory}:reference"]) == 0
    assert "leaf_values" in capsys.readouterr().out


def test_xgboost_unsupported_boosters():
    xgb = pytest.importorskip("xgboost")
    with pytest.raises(ValueError, match="fitted"):
        WeightSpace.from_xgboost(xgb.XGBRegressor())
    x = np.array([[-1.0], [1.0]])
    for booster in ("gblinear", "dart"):
        m = xgb.XGBRegressor(n_estimators=1, n_jobs=1, booster=booster).fit(x, [0, 1])
        with pytest.raises(ValueError, match="gbtree"):
            WeightSpace.from_xgboost(m)


def test_catboost_asymmetric_rejected():
    cb = pytest.importorskip("catboost")
    x = np.arange(20).reshape(-1, 1)
    m = cb.CatBoostRegressor(
        iterations=1,
        grow_policy="Depthwise",
        verbose=False,
        thread_count=1,
        allow_writing_files=False,
    ).fit(x, np.arange(20))
    with pytest.raises(ValueError, match="symmetric"):
        WeightSpace.from_catboost(m)


def test_text_invalid_values_retain_result(catboost_borders):
    model, ws, x = catboost_borders
    before = model.predict(x)
    llm = ScriptedLLM([reply("```python\nW['trees.0.leaf_values'][:] = np.nan\n```\nANSWER: invalid")])
    res = Session(ws, llm, mode="fix").run_text("bad edit")
    assert not res.ok and res.error and res.transcript
    np.testing.assert_array_equal(model.predict(x), before)
