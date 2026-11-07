"""scikit-learn random forests, preserving native nodes and fitted statistics."""

from __future__ import annotations

import numpy as np

from .base import TreeStore, check_version


class ForestStore(TreeStore):
    def __init__(self, model):
        import sklearn
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.utils.validation import check_is_fitted

        check_version(sklearn, (1, 6), (2, 0))
        if not isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
            raise TypeError("expected RandomForestClassifier or RandomForestRegressor")
        try:
            check_is_fitted(model)
        except Exception as e:
            raise ValueError("random forest must be fitted before adapting it") from e
        if model.n_outputs_ != 1:
            raise ValueError("multi-output random forests are unsupported; use a single target")
        self._model = model
        self._classification = isinstance(model, RandomForestClassifier)
        self._maps = []
        arrays, trees = {}, []
        for i, est in enumerate(model.estimators_):
            t = est.tree_
            leaf = np.flatnonzero(t.children_left == -1)
            internal = np.flatnonzero((t.children_left != -1) & np.isfinite(t.threshold))
            values = t.value[leaf, 0, :].copy()
            totals = values.sum(axis=1, keepdims=True) if self._classification else np.ones((len(leaf), 1))
            if self._classification:
                values /= totals
            arrays[f"trees.{i}.leaf_values"] = values
            arrays[f"trees.{i}.thresholds"] = t.threshold[internal].copy()
            self._maps.append((leaf, internal, totals))
            li = {int(n): j for j, n in enumerate(leaf)}
            ti = {int(n): j for j, n in enumerate(internal)}
            nodes = []
            for n in range(t.node_count):
                if n in li:
                    nodes.append({"node_id": n, "leaf_index": li[n]})
                else:
                    node = {
                        "node_id": n,
                        "feature": int(t.feature[n]),
                        "operator": "<=",
                        "left": int(t.children_left[n]),
                        "right": int(t.children_right[n]),
                        "missing_go_to_left": bool(t.missing_go_to_left[n]),
                    }
                    if n in ti:
                        node["threshold_index"] = ti[n]
                    else:
                        node["fixed_threshold"] = str(t.threshold[n])
                    nodes.append(node)
            trees.append(nodes)
        info = {
            "backend": "random-forest",
            "tree_count": len(trees),
            "task": "classification" if self._classification else "regression",
            "output_dimension": arrays["trees.0.leaf_values"].shape[1],
            "classes": model.classes_.tolist() if self._classification else None,
            "feature_names": getattr(model, "feature_names_in_", np.arange(model.n_features_in_)).tolist(),
            "aggregation": "mean of tree probability vectors" if self._classification else "mean of tree outputs",
            "training_statistics": "original fit; feature importance and OOB metrics are not recomputed",
        }
        super().__init__(arrays, info, trees, {"info": info, "trees": trees})

    def validate(self, arrays=None):
        super().validate(arrays)
        arrays = self._t if arrays is None else arrays
        if self._classification:
            for n, a in arrays.items():
                if n.endswith("leaf_values") and (np.any(a < 0) or not np.allclose(a.sum(axis=1), 1, rtol=0, atol=1e-8)):
                    raise ValueError(f"{n}: probabilities must be nonnegative and rows must sum to one")

    def _install(self):
        backups = [(e.tree_.value.copy(), e.tree_.threshold.copy()) for e in self._model.estimators_]
        try:
            for i, (est, (leaf, internal, totals)) in enumerate(zip(self._model.estimators_, self._maps, strict=True)):
                est.tree_.value[leaf, 0, :] = self._t[f"trees.{i}.leaf_values"] * totals
                est.tree_.threshold[internal] = self._t[f"trees.{i}.thresholds"]
        except Exception:
            for est, (values, thresholds) in zip(self._model.estimators_, backups, strict=True):
                est.tree_.value[...] = values
                est.tree_.threshold[...] = thresholds
            raise

    def initialize_for_write(self):
        info = super().initialize_for_write()
        if self._classification:
            for n, a in self._t.items():
                if n.endswith("leaf_values"):
                    a.fill(1 / a.shape[1])
        return info

    def save_model(self, path):
        import joblib

        joblib.dump(self._model, path)
