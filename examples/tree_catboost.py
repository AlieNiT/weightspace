"""Offline categorical CatBoost edit with preserved CTRs. Run with weightspace[catboost]."""

import numpy as np
from catboost import CatBoostClassifier, Pool

import weightspace as wsp
from weightspace.llm import ScriptedLLM, call, reply

data = [[str(i % 5), float(i % 7)] for i in range(100)]
pool = Pool(data, [int(i % 5 > 1) for i in range(100)], cat_features=[0])
model = CatBoostClassifier(
    iterations=4,
    depth=3,
    one_hot_max_size=2,
    random_seed=0,
    thread_count=1,
    verbose=False,
    allow_writing_files=False,
).fit(pool)
reference = Pool([["0", 1.0], ["3", np.nan], ["unseen", 0.0]], cat_features=[0])
before = model.predict(reference, prediction_type="RawFormulaVal")
ws = wsp.WeightSpace.from_catboost(model, pool=pool)


def margins():
    """Raw margins, including an unseen category and a missing numeric value."""
    return model.predict(reference, prediction_type="RawFormulaVal").tolist()


llm = ScriptedLLM(
    [
        reply(None, call("describe_tree", tree_id=0)),
        reply(None, call("python", code="W['trees.0.leaf_values'][:] += 0.5")),
        reply(None, call("forward_margins")),
        reply(None, call("submit", answer="Shifted raw margins by 0.5 while preserving categorical rules.")),
    ]
)
result = wsp.fix(ws, "Increase raw margins by 0.5 for all inputs.", llm, probes=[margins], forward_budget=1)
assert result.ok
np.testing.assert_allclose(model.predict(reference, prediction_type="RawFormulaVal"), before + 0.5)
print(result)
