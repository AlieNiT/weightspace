"""Reloadable XGBoost JSON, including native categorical and recoding data."""

from __future__ import annotations

import copy
import json
import os

import numpy as np

from .base import TreeStore, check_version


class XGBoostStore(TreeStore):
    def __init__(self, model):
        import xgboost as xgb

        check_version(xgb, (3, 1), (4, 0))
        if not isinstance(model, (xgb.Booster, xgb.XGBClassifier, xgb.XGBRegressor)):
            raise TypeError("expected an XGBoost Booster, XGBClassifier, or XGBRegressor")
        self._xgb = xgb
        try:
            self._model = model if isinstance(model, xgb.Booster) else model.get_booster()
            doc = json.loads(self._model.save_raw(raw_format="json"))
        except Exception as e:
            raise ValueError("XGBoost model must be fitted before adapting it") from e
        if doc.get("version", [0, 0])[:2] < [3, 1] or doc["version"][0] != 3:
            raise ValueError("unsupported XGBoost native schema; re-export with XGBoost >=3.1,<4")
        learner = doc["learner"]
        gb = learner["gradient_booster"]
        if gb["name"] != "gbtree":
            raise ValueError("only XGBoost gbtree is supported; DART and linear boosters are unsupported")
        objective = learner["objective"]["name"]
        if not (
            objective.startswith("reg:") or objective.startswith("binary:") or objective in ("multi:softprob", "multi:softmax")
        ):
            raise ValueError(f"unsupported XGBoost objective {objective!r}; use regression or classification, not ranking")
        if int(learner["learner_model_param"].get("num_target", 1)) != 1:
            raise ValueError("multi-output XGBoost targets are unsupported")
        native_trees = gb["model"]["trees"]
        if not native_trees:
            raise ValueError("XGBoost model must contain fitted trees")
        self._doc = doc
        structure = copy.deepcopy(doc)
        arrays, trees, self._maps = {}, [], []
        groups = gb["model"]["tree_info"]
        if len(groups) != len(native_trees):
            raise ValueError("unsupported XGBoost native schema: tree output groups do not match trees")
        for i, t in enumerate(native_trees):
            node_fields = ("left_children", "right_children", "split_conditions", "split_indices", "split_type", "default_left")
            if any(k not in t for k in node_fields) or len({len(t[k]) for k in node_fields}) != 1:
                raise ValueError("unsupported XGBoost native schema: inconsistent node arrays")
            if any(kind not in (0, 1) for kind in t["split_type"]):
                raise ValueError("unsupported XGBoost native schema: unknown split type")
            if int(t["tree_param"].get("size_leaf_vector", 1)) != 1:
                raise ValueError("XGBoost vector-leaf/multi-output trees are unsupported")
            leaf = np.flatnonzero(np.asarray(t["left_children"]) == -1)
            values = np.asarray(t["split_conditions"], dtype=np.float32)
            internal = np.flatnonzero(
                (np.asarray(t["left_children"]) != -1) & (np.asarray(t["split_type"]) == 0) & np.isfinite(values)
            )
            arrays[f"trees.{i}.leaf_values"] = values[leaf, None].copy()
            arrays[f"trees.{i}.thresholds"] = values[internal].copy()
            self._maps.append((leaf, internal))
            st = structure["learner"]["gradient_booster"]["model"]["trees"][i]
            for n in [*leaf, *internal]:
                st["split_conditions"][n] = None
            cats = {}
            for n, start, size in zip(
                t.get("categories_nodes", []), t.get("categories_segments", []), t.get("categories_sizes", []), strict=True
            ):
                cats[n] = t["categories"][start : start + size]
            li = {int(n): j for j, n in enumerate(leaf)}
            ti = {int(n): j for j, n in enumerate(internal)}
            nodes = []
            for n in range(len(values)):
                if n in li:
                    nodes.append({"node_id": n, "leaf_index": li[n], "output_group": groups[i]})
                else:
                    node = {
                        "node_id": n,
                        "feature": t["split_indices"][n],
                        "left": t["left_children"][n],
                        "right": t["right_children"][n],
                        "missing_go_to_left": bool(t["default_left"][n]),
                    }
                    if n in ti:
                        node.update(operator="<", threshold_index=ti[n])
                    elif t["split_type"][n] == 0:
                        node.update(operator="<", fixed_threshold=str(values[n]))
                    else:
                        node.update(operator="category in set -> right", categories=cats.get(n, []))
                    nodes.append(node)
            trees.append(nodes)
        info = {
            "backend": "xgboost",
            "tree_count": len(trees),
            "objective": objective,
            "feature_names": learner.get("feature_names", []),
            "feature_count": int(learner["learner_model_param"]["num_feature"]),
            "feature_types": learner.get("feature_types", []),
            "output_dimension": max(1, int(learner["learner_model_param"]["num_class"])),
            "classes": (
                list(range(max(2, int(learner["learner_model_param"]["num_class"]))))
                if objective.startswith(("binary:", "multi:"))
                else None
            ),
            "tree_output_groups": groups,
            "base_score": learner["learner_model_param"]["base_score"],
            "aggregation": (
                "sum inference leaf contributions per output group, add base margin, "
                "apply objective transform; no extra learning rate"
            ),
            "categorical": "category partitions and recoding are immutable; category labels may be unavailable",
            "training_statistics": "original fit; feature importance is not recomputed",
        }
        super().__init__(arrays, info, trees, structure)

    def _install(self):
        doc = copy.deepcopy(self._doc)
        for i, (leaf, internal) in enumerate(self._maps):
            t = doc["learner"]["gradient_booster"]["model"]["trees"][i]
            for nodes, name in ((leaf, "leaf_values"), (internal, "thresholds")):
                for node, value in zip(nodes, self._t[f"trees.{i}.{name}"].flat, strict=True):
                    t["split_conditions"][node] = float(value)
        # XGBoost's native JSON uses Infinity for reserved missing-only splits.
        # Editable arrays have already passed finite-value validation.
        payload = bytearray(json.dumps(doc).encode())
        candidate = self._xgb.Booster()
        candidate.load_model(payload)
        backup = self._model.save_raw(raw_format="ubj")
        try:
            self._model.load_model(candidate.save_raw(raw_format="ubj"))
        except Exception:
            self._model.load_model(backup)
            raise

    def save_model(self, path):
        if os.path.splitext(path)[1].lower() not in (".json", ".ubj", ".ubjson"):
            raise ValueError("XGBoost predictive exports require .json or .ubj/.ubjson")
        self._model.save_model(path)
