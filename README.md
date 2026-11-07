<h1 align="center">weightspace</h1>

<p align="center"><b>Let an LLM read and edit a neural network's weights directly.</b></p>

<p align="center">
  <a href="https://pypi.org/project/weightspace/"><img alt="PyPI" src="https://img.shields.io/pypi/v/weightspace"></a>
  <a href="https://github.com/leofmarciano/weightspace/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/leofmarciano/weightspace/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-2ea44f"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776ab">
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="docs/README.md">Docs</a> ·
  <a href="#examples">Examples</a> ·
  <a href="experiments/exp8">Exp. 8 replication</a> ·
  <a href="https://rikiyat.github.io/blog/weight-space-programming/">Paper</a> ·
  <a href="https://marcianotheory.com">Marciano Theory</a>
</p>

A small Python framework for **weight-space programming**: an LLM reads and edits a neural network's weights directly, by reasoning about what each part computes, instead of training it.

Based on [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi, Teoh, Zhang, Khattab, 2026). This repo is the harness, not the paper's results: how well it works depends on the LLM you plug in.

## What you get

- **Models.** PyTorch, Hugging Face, XGBoost, CatBoost, scikit-learn random forests, or `.safetensors` / `.npz` / `.pt` files. The LLM sees arrays and immutable tree descriptions; the model stays private.
- **Four tasks.** `read` (what does this network do?), `fix` (find and fix the cause of a failure), `write` (program a network from zero), `improve` (make it better at a goal).
- **Tools for the LLM.** List, describe (norms, SVD) and read tensor slices; a persistent Python sandbox over the weights; edits, checkpoints and rollbacks.
- **Budgeted forward passes.** You decide how the network may be run (scores, rollouts, generations, attention maps only...) by writing a `probe`, and how many times.
- **Edit tracking.** Every run reports exactly how many weights changed, and exports a sparse patch.
- **Text mode.** For small networks: dump every weight into the prompt and get one reply back, no tools (like the paper's first experiments).
- **Providers.** OpenAI-compatible APIs (OpenAI, OpenRouter, vLLM, Ollama), [Open Responses](https://www.openresponses.org) (`openresponses:<model>`) and Anthropic, or your own `LLM` class.

## Install

```bash
pip install "weightspace[all]"   # or pick: [torch] [hf] [trees] [sklearn] [xgboost] [catboost] [openai] [anthropic]
export OPENAI_API_KEY=...      # or ANTHROPIC_API_KEY / OPENROUTER_API_KEY
```

## Quick start

```python
import torch
import weightspace as wsp

model = ...  # any torch.nn.Module


@wsp.probe
def accuracy():
    """Per-class accuracy on a held-out batch."""
    with torch.no_grad():
        pred = model(x_val).argmax(-1)
    return {c: float((pred[y_val == c] == c).float().mean()) for c in range(4)}


res = wsp.fix(
    model,
    "The classifier never predicts class 3. Find out why and fix it.",
    llm="openai:gpt-5",  # or "anthropic:claude-sonnet-4-5", "openrouter:..."
    context="Linear(2->16) -> ReLU -> Linear(16->4); input is a 2D point, output is its quadrant.",
    probes=[accuracy],
    forward_budget=3,
)

print(res)  # answer, confidence, forward passes used, which weights changed
res.save("runs/fix")  # transcript.jsonl, result.json, patch.npz
```

The model is edited in place. The other tasks look the same:

```python
wsp.read(model, "Which game was this policy trained on?", llm=..., context=GRID_AND_ACTIONS)
wsp.read(model, "What does this network compute?", llm=..., text_mode=True)  # weights in the prompt
wsp.write(model, "Play Breakout: move the paddle under the ball.", llm=..., context=GAME_RULES)  # zeroes first
wsp.improve(hf_model, "Stop the reasoning loops.", llm=..., probes=[generate], forward_budget=25)
```

## Tree ensembles

All four tasks also work on an **existing fitted tree structure**. Install `weightspace[trees]` or a single backend extra; Torch is optional.

| Backend | Versions | Models | Editable parameters |
|---|---|---|---|
| scikit-learn | `>=1.6,<2` | RandomForestClassifier / RandomForestRegressor | Leaf outputs and numeric thresholds |
| XGBoost | `>=3.1,<4` | Booster / XGBClassifier / XGBRegressor, `gbtree` | Inference leaf contributions and numeric thresholds |
| CatBoost | `>=1.2.8,<1.3` | Symmetric classifiers and regressors | Leaf vectors and safe shared numeric borders |

Scalar regression, binary classification and multiclass classification are supported. XGBoost and CatBoost retain native categorical rules, encodings, missing routes and output transformations. Forest inputs must be numeric or externally encoded. Ranking, multi-output targets, XGBoost linear/DART and CatBoost asymmetric/text/embedding models are rejected.

```python
ws = wsp.WeightSpace.from_xgboost(fitted_model)  # or from_sklearn(fitted_forest)
# Categorical CatBoost needs a private reference pool for reloadable JSON export:
# ws = wsp.WeightSpace.from_catboost(fitted_model, pool=reference_pool)
print(ws.model_info)
print(ws.describe_tree(0, offset=0, limit=100))
res = wsp.fix(ws, "Inspect the leaf contributions and fix the described bias.", llm=...)
ws.save_model("edited.json")  # XGBoost .json/.ubj; CatBoost .cbm; forest joblib
```

`trees.<id>.leaf_values` has shape `(leaf_count, output_dimension)` (one column per XGBoost tree, with its output group in metadata). Forest classification leaves are probability vectors in class order; invalid probabilities are rejected. Numeric thresholds use `trees.<id>.thresholds`; CatBoost shares `features.<id>.borders` across trees. Features, nodes, tree counts, category rules and output dimensions stay fixed. `write()` initializes this fitted structure; it does not fit or construct trees. Use `zero_init=False` to keep current values.

Edits synchronize into the supplied object before probes. Invalid edits keep native predictions at the last valid state and remain staged for correction. `save()` / `--save-weights` export arrays only and cannot reconstruct a tree model. Native export uses `save_model()` / `--save-model`. Native file loading requires `model_type="xgboost"`, `"catboost"` or `"random-forest"`; forest joblib loading additionally requires `allow_pickle=True` and must only be used for trusted files.

```bash
weightspace inspect model.json --model-type xgboost
weightspace inspect forest.joblib --model-type random-forest --allow-pickle
weightspace inspect model.cbm --model-type catboost --catboost-pool reference.py:make_pool
weightspace fix model.json "Correct the leaf bias" --model-type xgboost --llm openai:gpt-5 --save-model edited.json
```

See [tree semantics and initialization](docs/concepts.md#tree-parameters) and offline scripted examples for [forests](examples/tree_forest.py), [XGBoost](examples/tree_xgboost.py) and [categorical CatBoost](examples/tree_catboost.py).

## Thinking on or off

`thinking` sets how much the editing LLM reasons before it acts. Use `thinking=False` to have it edit the weights with no thinking at all:

```python
wsp.fix(model, "...", llm="openai:gpt-5.1", thinking=False)  # or "low" / "medium" / "high"; None = provider default
```

With thinking off, the provider's reasoning is disabled and the LLM is told to go straight to tool calls and edits. How that reaches each provider:

| Backend | off | on |
|---|---|---|
| OpenAI | `reasoning_effort="none"` (`OpenAIChat(..., off_effort="minimal")` for gpt-5) | `reasoning_effort=<level>` |
| Open Responses | `reasoning.effort="none"` | `reasoning.effort=<level>` |
| OpenRouter | `reasoning.enabled=false` | `reasoning.effort=<level>` |
| vLLM / SGLang (Qwen3 etc.) | `OpenAIChat(..., thinking_style="chat_template")` → `enable_thinking=false` | `enable_thinking=true` |
| Anthropic | `thinking.type="disabled"`; on Opus 5.5, where thinking is always on, `effort="low"` instead | adaptive thinking with `effort=<level>` on newer models such as Opus 5.5; older models fall back to a token budget |

CLI: `--no-thinking` or `--thinking low|medium|high|xhigh|max`. The setting is saved in `result.json`, so you can compare runs with and without thinking.

`speed="fast"` (CLI `--fast`) turns on the provider's fast mode for the editing LLM: same model, faster output, higher price. See [providers](docs/providers.md#fast-mode).

## Probes

A probe is any function. It runs after the current weights are synced into the model, and whatever it returns is what the LLM sees. Its arguments become the tool's schema.

```python
@wsp.probe(cost=1)
def generate(prompts: list, max_new_tokens: int = 256):
    """Generate answers for up to 8 prompts and report repetition."""
    ...
```

Want the LLM to see only attention patterns, not outputs? Write a probe that returns only attention patterns. That's how you set what the LLM can observe.

## Lower level

```python
ws = wsp.WeightSpace.from_hf("Qwen/Qwen2.5-0.5B-Instruct")  # or from_torch / from_file / from_numpy
ws.W["model.layers.15.mlp.down_proj.weight"][:, :64] -= ...  # tracked in-place edit
print(ws.diff())  # what changed vs. the original
ws.checkpoint("a")
ws.restore("a")
ws.revert()
ws.save_patch("fix.npz")
other.apply_patch("fix.npz")

s = wsp.Session(
    ws,
    llm,
    mode="improve",
    probes=[...],
    forward_budget=50,
    tools=[my_tool],
    namespace={"tok": tokenizer},
    max_steps=80,
    verbose=True,
)
res = s.run("...")
```

Custom tools: `wsp.tool("name", "description", x={"type": "number"})(fn)`.
Offline runs and tests: `wsp.ScriptedLLM([...])` replays a fixed script, and
`wsp.ScriptedLLM.from_transcript("runs/1/transcript.jsonl")` replays a logged run.

## Production

Budgets (`token_budget`, `cost_budget`, `time_budget`), cost tracking (`pricing=`), `stop_reason`, partial results
on errors and Ctrl-C, OpenTelemetry tracing, parallel read-only tool calls and replay are covered in
[docs/production.md](docs/production.md).

```python
from weightspace.otel import OpenTelemetryCallback

res = wsp.fix(
    model, task, llm="anthropic:claude-opus-5-5", pricing=(5, 25, 0.5, 6.25), cost_budget=5.0, callbacks=[OpenTelemetryCallback()]
)
print(res.stop_reason, res.tokens, res.cost)
```

## CLI

```bash
weightspace inspect model.safetensors --describe
weightspace read model.safetensors "What was this trained to do?" --llm openai:gpt-5 --context @notes.md
weightspace fix model.safetensors "..." --llm anthropic:claude-sonnet-4-5 --save-weights fixed.safetensors --out runs/1
weightspace fix model.safetensors "..." --llm openai:gpt-5 --probe probes.py:rollout --forward-budget 10 \
    --price 1.25,10 --max-cost 2 --events events.jsonl --json
weightspace fix model.safetensors "..." --replay runs/1/transcript.jsonl   # no tokens spent
```

## Results: Exp. 8 replication

We reran the paper's Exp. 8 with this harness: fix Qwen3.5-0.8B's math by editing its weights, no training. Test sets are MATH-500 and GSM8K test, one sample per problem, last `\boxed{}` graded (code in [`experiments/exp8`](experiments/exp8)). ± is one standard error; "cut" is the share of answers that hit the token limit.

**Thinking on** (paper protocol, 32k tokens, first 250 problems of each set):

| Edit | Editing LLM | Forward passes | Weights changed | MATH-500 | cut | GSM8K | cut |
|---|---|---|---|---|---|---|---|
| none (base) | | | 0 | 8.4 ± 1.8 | 92% | 12.4 ± 2.1 | 89% |
| paper's edit ([checkpoint](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming)) | gpt-6-astra | 50 | 131,072 | 52.4 ± 3.2 | 34% | 61.6 ± 3.1 | 10% |
| 4 MLP neurons ×6 | Opus 5.5, thinking high | 7 of 12 | 4,096 | 40.0 ± 3.1 | 60% | 53.2 ± 3.2 | 42% |
| 4 attention heads ×1.5 | Opus 5.5, thinking off, fast | 7 of 12 | 1,048,576 | 17.2 ± 2.4 | 82% | 34.0 ± 3.0 | 59% |
| 1 MLP neuron ×0.5 (unfinished run) | gpt-6-astra, thinking high | 12 budget, run lost | 1,024 | not measured yet | | | |

**Thinking off** (model card sampling, 8k tokens, full sets: 500 MATH-500, 500 GSM8K):

| Edit | MATH-500 | cut | GSM8K | cut |
|---|---|---|---|---|
| none (base) | 46.4 ± 2.2 | 31% | 56.6 ± 2.2 | 6% |
| paper's edit | 42.0 ± 2.2 | 29% | 57.2 ± 2.2 | 3% |
| Opus, thinking high | 41.8 ± 2.2 | 23% | 54.4 ± 2.2 | 2% |
| Opus, thinking off | 33.2 ± 2.1 | 20% | 42.0 ± 2.2 | 3% |
| Astra's neuron (first 250 of each set) | 46.0 ± 3.2 | 28% | 56.4 ± 3.1 | 6% |

What this says:

- Editing works. Opus found 4 neurons that push down "Wait" and amplified them; 4,096 weights out of 750M took the thinking model from 8.4% / 12.4% to 40.0% / 53.2%.
- It does not beat the paper. The paper's edit, with 50 forward passes against Opus's 7, is ahead by 8 to 12 points. Our run of the paper's checkpoint (52.4% / 61.6%) is close to what the paper reports (48.1% / 67.1%, with more samples per problem), and our base (8.4% / 12.4%) matches theirs (7.9% / 13.8%).
- Turning thinking off is a strong baseline. The base model with thinking off scores 46.4% / 56.6%, above Opus's edit, and none of the edits help in non-thinking mode. The edits mostly cut the thinking loops; they don't add math skill. Compare any edit against the thinking-off base.
- A second Opus session (up to 25 forward passes in total) is not in the table. It ran out of steps without submitting, its best variant (36.6% on practice problems against 30.5% for the ×6 edit) is within noise, and its layer-23 change wasn't saved before the GPU pod died.
- Training-set gains overstate test gains: Opus measured 8.6% → 50.8% on held-out practice problems, but a fresh practice set gave 30.5%.

## Examples

- `examples/fix_quadrants.py`: fix a sabotaged classifier (runs offline with a scripted LLM if you omit `--llm`)
- `examples/read_which_task.py`: name the boolean function an MLP learned, from its weights in the prompt
- `examples/hf_reduce_loops.py`: edit a Hugging Face LM to stop repetition loops, with generation and attention probes

## Rules and caveats

The system prompt tells the LLM not to train, fit weights to data, or run the network inside the sandbox; the only way to run it is the budgeted probes. The sandbox never holds the model object, but it is not a security boundary: code runs in your process, and an LLM could still multiply matrices by hand. Check transcripts if the rules matter for your experiment, as the paper's authors did.

## Contributing

Issues and PRs are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and conventions, [SECURITY.md](SECURITY.md) to report vulnerabilities, and [CHANGELOG.md](CHANGELOG.md) for releases.

```bash
pip install -e ".[all,dev]"
pytest && ruff check . && mypy src/weightspace
```

## Citation

If you use weightspace in research, cite the original paper and this repository ([CITATION.cff](CITATION.cff)).

## License

Apache 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

---

<p align="center">Marciano Theory SI</p>
