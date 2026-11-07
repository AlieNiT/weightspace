"""Offline fixed-structure forest edit. Run with weightspace[sklearn]."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import weightspace as wsp
from weightspace.llm import ScriptedLLM, call, reply

x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
model = RandomForestClassifier(n_estimators=1, max_depth=1, bootstrap=False, random_state=0).fit(x, [0, 0, 1, 1])
ws = wsp.WeightSpace.from_sklearn(model)


def probabilities():
    """Probabilities on the four reference points, using the original object."""
    return model.predict_proba(x).tolist()


llm = ScriptedLLM(
    [
        reply(None, call("describe_tree", tree_id=0)),
        reply(None, call("write_tensor", name="trees.0.leaf_values", values=[[0.8, 0.2], [0.1, 0.9]])),
        reply(None, call("forward_probabilities")),
        reply(None, call("submit", answer="Set calibrated probabilities in the two existing leaves.")),
    ]
)
result = wsp.fix(
    ws, "Set the left leaf to [0.8, 0.2] and the right to [0.1, 0.9].", llm, probes=[probabilities], forward_budget=1
)
assert result.ok and result.diff.changed_params == 4
np.testing.assert_allclose(model.predict_proba(x), [[0.8, 0.2]] * 2 + [[0.1, 0.9]] * 2)
print(result)
