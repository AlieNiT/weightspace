# weightspace

A small Python framework for **weight-space programming**: an LLM reads and edits a neural network's weights directly, by reasoning about what each part computes, instead of training it.

Based on [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi, Teoh, Zhang, Khattab, 2026). This repo is the harness, not the paper's results: how well it works depends on the LLM you plug in.

## What you get

- **Any model.** PyTorch modules, Hugging Face models, or `.safetensors` / `.npz` / `.pt` files. The LLM only ever sees tensors, never the model object.
- **Four tasks.** `read` (what does this network do?), `fix` (find and fix the cause of a failure), `write` (program a network from zero), `improve` (make it better at a goal).
- **Tools for the LLM.** List, describe (norms, SVD) and read tensor slices; a persistent Python sandbox over the weights; edits, checkpoints and rollbacks.
- **Budgeted forward passes.** You decide how the network may be run (scores, rollouts, generations, attention maps only...) by writing a `probe`, and how many times.
- **Edit tracking.** Every run reports exactly how many weights changed, and exports a sparse patch.
- **Text mode.** For small networks: dump every weight into the prompt and get one reply back, no tools (like the paper's first experiments).
- **Providers.** OpenAI-compatible APIs (OpenAI, OpenRouter, vLLM, Ollama) and Anthropic, or your own `LLM` class.

## Install

```bash
pip install -e ".[all]"        # or pick: .[torch] .[hf] .[openai] .[anthropic]
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

## Thinking on or off

`thinking` sets how much the editing LLM reasons before it acts. Use `thinking=False` to have it edit the weights with no thinking at all:

```python
wsp.fix(model, "...", llm="openai:gpt-5.1", thinking=False)  # or "low" / "medium" / "high"; None = provider default
```

With thinking off, the provider's reasoning is disabled and the LLM is told to go straight to tool calls and edits. How that reaches each provider:

| Backend | off | on |
|---|---|---|
| OpenAI | `reasoning_effort="none"` (`OpenAIChat(..., off_effort="minimal")` for gpt-5) | `reasoning_effort=<level>` |
| OpenRouter | `reasoning.enabled=false` | `reasoning.effort=<level>` |
| vLLM / SGLang (Qwen3 etc.) | `OpenAIChat(..., thinking_style="chat_template")` → `enable_thinking=false` | `enable_thinking=true` |
| Anthropic | `thinking.type="disabled"` | extended thinking with a 2k / 8k / 16k token budget |

CLI: `--no-thinking` or `--thinking low|medium|high`. The setting is saved in `result.json`, so you can compare runs with and without thinking.

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
Offline runs and tests: `wsp.ScriptedLLM([...])` replays a fixed script.

## CLI

```bash
weightspace inspect model.safetensors --describe
weightspace read model.safetensors "What was this trained to do?" --llm openai:gpt-5 --context @notes.md
weightspace fix model.safetensors "..." --llm anthropic:claude-sonnet-4-5 --save-weights fixed.safetensors --out runs/1
```

## Examples

- `examples/fix_quadrants.py`: fix a sabotaged classifier (runs offline with a scripted LLM if you omit `--llm`)
- `examples/read_which_task.py`: name the boolean function an MLP learned, from its weights in the prompt
- `examples/hf_reduce_loops.py`: edit a Hugging Face LM to stop repetition loops, with generation and attention probes

## Rules and caveats

The system prompt tells the LLM not to train, fit weights to data, or run the network inside the sandbox; the only way to run it is the budgeted probes. The sandbox never holds the model object, but it is not a security boundary: code runs in your process, and an LLM could still multiply matrices by hand. Check transcripts if the rules matter for your experiment, as the paper's authors did.

## Development

```bash
pip install -e ".[all,dev]"
pytest
ruff check . && ruff format --check .
```
