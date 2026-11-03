# Concepts

## WeightSpace

`WeightSpace` wraps a set of named tensors. It loads them from:

- `WeightSpace.from_torch(module)`: any `torch.nn.Module`, so edits change the live model in place;
- `WeightSpace.from_hf(name_or_model)`: a Hugging Face model;
- `WeightSpace.from_file(path)`: `.safetensors`, `.npz` or `.pt`;
- `WeightSpace.from_numpy(dict)`: plain arrays.

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
