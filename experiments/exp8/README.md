# Exp. 8 replication: Qwen3.5-0.8B math

A replication of the paper's Experiment 8 with `weightspace`: an LLM edits the weights of [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) to make it better at math, with thinking on, and the edit is evaluated on benchmarks the editor never sees.

- `run_edit.py`: the editing session. The LLM gets the 24-layer text model as a `WeightSpace`, practice problems (GSM8K train, MATH train, AIME) and two budgeted probes: `generate` (vLLM generation with the evaluation protocol, optionally paired original vs. edited) and `teacher_forced` (log-probs, hidden states and MLP activations). Scope guards refuse edits to the final norm, the embeddings and whole-tensor rescales.
- `evaluate.py`: MATH-500 and GSM8K test with the same protocol (chat template with thinking, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, 32,768 new tokens, last `\boxed{}` graded).
- `evaluate.py --no-thinking`: the same sets with thinking off (`enable_thinking=False`, the model card's non-thinking sampling: temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5, 8,192 new tokens). This is the baseline any edit has to beat, since turning thinking off already removes most loops.
- `table.py`: renders the result JSONs of an eval dir as a Markdown table.
- `common.py`, `gen_worker.py`: prompts, grading, loop detection, checkpoint building and the vLLM worker.

Needs one GPU (tested on an H100 80GB with vLLM 0.33), `pip install vllm datasets`, and the practice sets from [lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs) as JSONL in `--practice` (`gsm8k_train.jsonl`, `math_train.jsonl`, `aime.jsonl`).

```bash
python run_edit.py --llm anthropic:claude-opus-5-5 --budget 12 --work runs/opus
python evaluate.py base    $QWEN_SNAPSHOT        runs/eval
python evaluate.py opus    runs/opus/edit.npz    runs/eval
python evaluate.py astra   $ASTRA_SNAPSHOT       runs/eval   # the paper's public edit, as a positive control
python evaluate.py base_nothink $QWEN_SNAPSHOT  runs/eval --no-thinking
python table.py runs/eval
```

The paper's own edit is [RikiyaT/Qwen3.5-0.8B-weight-space-programming](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming).
