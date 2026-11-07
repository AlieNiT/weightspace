# Concepts

## WeightSpace

`WeightSpace` wraps a set of named tensors. It loads them from:

- `WeightSpace.from_torch(module)`: any `torch.nn.Module`, so edits change the live model in place;
- `WeightSpace.from_hf(name_or_model)`: a Hugging Face model;
- `WeightSpace.from_file(path)`: `.safetensors`, `.npz` or `.pt`;
- `WeightSpace.from_numpy(dict)`: plain arrays.
- `WeightSpace.from_sklearn(forest)`, `from_xgboost(model)`, `from_catboost(model, pool=reference_pool)`: supported fitted tree ensembles.

Every write is tracked against a snapshot of the original tensor, so you can always ask what changed:

```python
ws.W["fc.weight"][3] = 0  # tracked edit
ws.diff()  # changed values, fraction, L2 and max |delta| per tensor
ws.checkpoint("a")
ws.restore("a")
ws.revert("fc.weight")
ws.save_patch("fix.npz")
other.apply_patch("fix.npz")
```

`ws.R` is a read-only view and `ws.W` a writable one. `ws.sync()` pushes edits back into the underlying model before it runs.

## Tree parameters

Tree adapters use private staging arrays, rather than native writable views. `ws.model_info` and `ws.describe_tree(tree_id, offset=0, limit=100)` expose immutable descriptions without prediction. The latter is also a read-only session tool. Tree IDs follow native order. Array positions map explicitly to native leaf IDs, numeric nodes or CatBoost shared borders. Text mode includes structure and arrays within `max_chars`.

| Array | Meaning |
|---|---|
| `trees.<id>.leaf_values` | `(leaf_count, output_dimension)`; forest regression predictions, forest class probabilities, or boosting inference contributions |
| `trees.<id>.thresholds` | Numeric internal nodes only; forest `<=` routes left, XGBoost `<` routes left |
| `features.<id>.borders` | CatBoost shared float32 numeric borders, strictly increasing; each split reference updates together |

XGBoost scalar trees contribute to one output group each. Their stored contributions already include the learning rate. The objective transforms the summed margins and preserved base score. CatBoost uses native leaf-bit order, then its preserved scale/bias and loss transformation. Forests average tree predictions. Class probabilities must be finite, nonnegative, and sum to one per leaf (absolute tolerance `1e-8`); edits are rejected rather than normalized.

Reserved infinite thresholds used for missing-only splits remain immutable and appear as `fixed_threshold` in tree descriptions. They are excluded from editable threshold arrays; ordinary editable values must be finite.

Categorical partitions, category recoding, CatBoost hashes, one-hot rules, CTR definitions/tables and missing routes are immutable. CatBoost needs `pool=` for categorical export; retain a reference pool covering the model's categories, and keep it private to the adapter. Category labels may be unavailable when only hashes or codes remain. Borders involved in CTR projections are excluded from editing. Unseen-category behavior follows the backend (including errors where XGBoost recoding rejects unseen labels).

`ws.initialize_for_write()` initializes and synchronizes values before committing a new diff baseline. Tensor stores retain zero initialization. Boosting and forest regression leaves become zero; forest classification leaves become uniform probabilities. Independent numeric thresholds become zero. CatBoost border groups with multiple entries stay intact, since setting them all to zero would break strict ordering. The return value and `ws.initialization_info` report these as `preserved_border_groups`. Base scores, scale/bias, categories, topology and missing routes remain intact. `write(..., zero_init=False)` and CLI `write --no-zero-init` skip initialization.

Validation covers the full proposed ensemble before a native update. Failed validation retains staged edits for correction or `revert()`, leaves the native model at its last valid state, and prevents probe execution or budget use. Failed final synchronization returns `stop_reason="error"`, `ok=False`, diagnostics and transcript. Unchanged sync never reloads or predicts. Models must remain under the caller's exclusive control during editing; concurrent refits or external mutation are unsupported. Feature importances, training statistics and OOB scores describe the original fit and are not recomputed.

`changed_params` counts only editable scalars. Checkpoints, restore, revert and commit use the existing tracking engine. Tree patches include versioned backend, schema, structure and baseline fingerprints; all entries and values are validated before any staging mutation. Apply them to the matching baseline. Legacy tensor patches keep their existing format.

`save(path)` exports parameter arrays to `.npz` / `.safetensors`; those arrays alone cannot reconstruct a predictive tree model. `save_model(path)` synchronizes and exports XGBoost JSON/UBJSON, CatBoost CBM or forest joblib. Reload with `WeightSpace.from_file(path, model_type=..., pool=...)`; categorical CatBoost still requires the reference pool. Forest joblib loading requires `allow_pickle=True` and trusted files because unpickling can execute code.

The support matrix and dependency ranges are in the [README](../README.md#tree-ensembles). NumPy-only installations remain usable, and each tree backend imports lazily.

## Sessions and modes

`wsp.Session(ws, llm, mode=...)` runs an agent loop where the LLM calls tools until it calls `submit`. The shortcuts `wsp.read`, `wsp.fix`, `wsp.write` and `wsp.improve` build one for you.

| Mode | Can edit | Typical task |
|---|---|---|
| `read` | no | What does this network compute? Which task was it trained on? |
| `fix` | yes | Find the cause of a known failure and repair it. |
| `write` | yes (weights zeroed first) | Program a network from scratch for a described behaviour. |
| `improve` | yes | Make the network better at a goal, measured by your probes. |

## What the LLM sees

The LLM never gets the model object. It gets tensor names, shapes and statistics, and these tools:

- `list_tensors`, `describe_tensor` (norms, quantiles, SVD), `read_tensor` (slices) and, when editing, `write_tensor`;
- `python`: a persistent sandbox with `np`, `R`, `W` (edit modes), `diff`, `checkpoint`, `restore`, `revert` and anything you add through `namespace=`;
- your probes, each costing forward budget;
- `submit`, with an answer and a confidence.

The sandbox runs in your process. It is not a security boundary (see [SECURITY.md](../SECURITY.md)). The rule against training or running the network in the sandbox comes from the prompt and is not enforced by code, so check transcripts when it matters.

## Results

`Session.run()` returns a `Result` with the answer, confidence, steps, forward passes used, token usage, the thinking setting and the weight diff. `res.save(dir)` writes `transcript.jsonl`, `result.json` and `patch.npz`.
