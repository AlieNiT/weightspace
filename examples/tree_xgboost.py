"""Offline XGBoost margin edit. Run with weightspace[xgboost]."""

import numpy as np
from xgboost import XGBRegressor

import weightspace as wsp
from weightspace.llm import ScriptedLLM, call, reply

x = np.array([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float32)
model = XGBRegressor(n_estimators=2, max_depth=1, n_jobs=1, random_state=0).fit(x, [-3, -3, 5, 5])
before = model.predict(x)
ws = wsp.WeightSpace.from_xgboost(model)


def predictions():
    """Predictions on reference points."""
    return model.predict(x).tolist()


llm = ScriptedLLM(
    [
        reply(None, call("describe_tree", tree_id=0)),
        reply(None, call("python", code="W['trees.0.leaf_values'][:] += 1")),
        reply(None, call("forward_predictions")),
        reply(None, call("submit", answer="Added one to every contribution in the first tree.")),
    ]
)
result = wsp.improve(
    ws, "Increase every prediction by exactly one, preserving splits.", llm, probes=[predictions], forward_budget=1
)
assert result.ok
np.testing.assert_allclose(model.predict(x), before + 1, atol=1e-6)
print(result)
