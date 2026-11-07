"""CatBoost symmetric trees, shared float borders, and preserved categorical CTRs."""

from __future__ import annotations

import copy
import json
import os
import tempfile

import numpy as np

from .base import TreeStore, check_version


class CatBoostStore(TreeStore):
    def __init__(self, model, *, pool=None):
        import catboost as cb

        check_version(cb, (1, 2), (1, 3))
        if tuple(int(p) for p in cb.__version__.split(".")) < (1, 2, 8):
            raise ValueError("CatBoost >=1.2.8,<1.3 is required")
        if not isinstance(model, cb.CatBoost):
            raise TypeError("expected a fitted CatBoost classifier or regressor")
        if not model.is_fitted():
            raise ValueError("CatBoost model must be fitted before adapting it")
        if model.get_text_feature_indices() or model.get_embedding_feature_indices():
            raise ValueError("CatBoost text and embedding models are unsupported; use numeric/categorical features")
        params = model.get_all_params()
        loss = params.get("loss_function", "")
        losses = {
            "RMSE",
            "MAE",
            "Quantile",
            "LogLinQuantile",
            "Lq",
            "Huber",
            "Expectile",
            "Tweedie",
            "Poisson",
            "MAPE",
            "Logloss",
            "CrossEntropy",
            "MultiClass",
            "MultiClassOneVsAll",
        }
        if loss.split(":")[0] not in losses:
            raise ValueError(f"unsupported CatBoost loss {loss!r}; ranking and multi-output targets are unsupported")
        if params.get("grow_policy", "SymmetricTree") != "SymmetricTree":
            raise ValueError("only CatBoost symmetric trees are supported; asymmetric grow policies are unsupported")
        if model.get_cat_feature_indices() and pool is None:
            raise ValueError("categorical CatBoost models require from_catboost(model, pool=reference_pool)")
        if pool is not None and not isinstance(pool, cb.Pool):
            raise TypeError("pool must be a CatBoost Pool containing the reference categorical values")
        self._model, self._pool, self._cb = model, pool, cb
        with tempfile.TemporaryDirectory(prefix="weightspace-catboost-") as tmp:
            path = os.path.join(tmp, "model.json")
            model.save_model(path, format="json", pool=pool)
            with open(path) as f:
                doc = json.load(f)
        if "oblivious_trees" not in doc or not doc["oblivious_trees"]:
            raise ValueError("unsupported CatBoost native schema: expected fitted oblivious_trees")
        self._doc = doc
        features = doc.get("features_info", {})
        # Float conditions in CTR projections define the preserved CTR tables. Lock their entire border groups.
        locked = set()
        for ctr in features.get("ctrs", []):
            for element in ctr.get("elements", []):
                if "float_feature_index" in element:
                    locked.add(element["float_feature_index"])
        arrays, trees = {}, []
        self._border_features = {}
        structure = copy.deepcopy(doc)
        # CBM round trips round training options. Only inference structure and
        # class/loss semantics belong in patch compatibility, not training logs.
        training_info = structure.pop("model_info", {})
        structure["output_semantics"] = {
            "loss": loss,
            "class_params": training_info.get("class_params"),
            "multiclass_params": training_info.get("multiclass_params"),
        }
        for i, feature in enumerate(features.get("float_features", [])):
            fi = feature["feature_index"]
            if feature["borders"] and fi not in locked:
                arrays[f"features.{fi}.borders"] = np.asarray(feature["borders"], dtype=np.float32)
                self._border_features[fi] = i
                structure["features_info"]["float_features"][i]["borders"] = [None] * len(feature["borders"])
        self._split_map = []
        dimensions = set()
        for i, t in enumerate(doc["oblivious_trees"]):
            splits = t.get("splits") or []
            count = 2 ** len(splits)
            dim, remainder = divmod(len(t["leaf_values"]), count)
            if remainder or not dim:
                raise ValueError("unsupported CatBoost leaf schema")
            dimensions.add(dim)
            arrays[f"trees.{i}.leaf_values"] = np.asarray(t["leaf_values"], dtype=np.float64).reshape(count, dim)
            st = structure["oblivious_trees"][i]
            st["leaf_values"] = [None] * len(t["leaf_values"])
            nodes = []
            for depth, split in enumerate(splits):
                node = {"depth": depth, **split, "operator": "> -> leaf bit 1", "leaf_bit": depth}
                if split["split_type"] == "FloatFeature":
                    fi = split["float_feature_index"]
                    feature = next(f for f in features["float_features"] if f["feature_index"] == fi)
                    positions = [j for j, b in enumerate(feature["borders"]) if b == split["border"]]
                    if len(positions) != 1:
                        raise ValueError("unsupported CatBoost schema: split border missing from float feature")
                    node.update(
                        flat_feature_index=feature["flat_feature_index"], nan_value_treatment=feature["nan_value_treatment"]
                    )
                    if fi in self._border_features:
                        j = positions[0]
                        self._split_map.append((i, depth, fi, j))
                        node.pop("border")
                        node.update(border_array=f"features.{fi}.borders", border_index=j)
                        st["splits"][depth]["border"] = None
                elif split["split_type"] == "OneHotFeature":
                    node["operator"] = "category hash == value -> leaf bit 1"
                elif split["split_type"] != "OnlineCtr":
                    raise ValueError(f"unsupported CatBoost split type {split['split_type']}")
                nodes.append(node)
            nodes.append({"leaf_count": count, "leaf_order": "row index is sum(split-result << depth)", "output_dimension": dim})
            trees.append(nodes)
        if len(dimensions) != 1 or (loss not in ("MultiClass", "MultiClassOneVsAll") and dimensions != {1}):
            raise ValueError("multi-output CatBoost models are unsupported")
        info = {
            "backend": "catboost",
            "tree_count": len(trees),
            "loss": loss,
            "output_dimension": dimensions.pop(),
            "feature_names": model.feature_names_,
            "classes": doc.get("model_info", {})
            .get("class_params", {})
            .get("class_names", np.asarray(getattr(model, "classes_", [])).tolist()),
            "aggregation": "sum leaf vectors, multiply by scale, add bias, apply loss transform",
            "scale_and_bias": doc.get("scale_and_bias"),
            "features": features,
            "locked_border_features": sorted(locked),
            "categorical": "hashes, one-hot rules, CTR definitions and tables are immutable",
            "training_statistics": "original fit; feature importance is not recomputed",
        }
        # Editable border values are described by arrays rather than stale immutable copies.
        info["features"] = copy.deepcopy(features)
        for fi, pos in self._border_features.items():
            info["features"]["float_features"][pos]["borders"] = f"features.{fi}.borders"
        super().__init__(arrays, info, trees, structure)

    def validate(self, arrays=None):
        super().validate(arrays)
        arrays = self._t if arrays is None else arrays
        for n, a in arrays.items():
            if n.endswith(".borders") and np.any(np.diff(a) <= 0):
                raise ValueError(f"{n}: borders must remain strictly ordered at float32 precision")

    def initialize_for_write(self):
        preserved = []
        for n, a in self._t.items():
            if n.endswith(".borders") and a.size > 1:
                preserved.append(n)
            else:
                a.fill(0)
        return {"preserved_border_groups": preserved}

    def _install(self):
        doc = copy.deepcopy(self._doc)
        for fi, pos in self._border_features.items():
            doc["features_info"]["float_features"][pos]["borders"] = self._t[f"features.{fi}.borders"].tolist()
        for i, depth, fi, j in self._split_map:
            doc["oblivious_trees"][i]["splits"][depth]["border"] = float(self._t[f"features.{fi}.borders"][j])
        for i, t in enumerate(doc["oblivious_trees"]):
            t["leaf_values"] = self._t[f"trees.{i}.leaf_values"].ravel().tolist()
        with tempfile.TemporaryDirectory(prefix="weightspace-catboost-") as tmp:
            path = os.path.join(tmp, "candidate.json")
            with open(path, "w") as f:
                json.dump(doc, f, allow_nan=False)
            candidate = self._cb.CatBoost()
            candidate.load_model(path, format="json")
            native = os.path.join(tmp, "candidate.cbm")
            backup = os.path.join(tmp, "backup.cbm")
            candidate.save_model(native, format="cbm")
            self._model.save_model(backup, format="cbm")
            try:
                self._model.load_model(native, format="cbm")
            except Exception:
                self._model.load_model(backup, format="cbm")
                raise

    def save_model(self, path):
        if os.path.splitext(path)[1].lower() != ".cbm":
            raise ValueError("CatBoost predictive exports require .cbm")
        self._model.save_model(path, format="cbm")
