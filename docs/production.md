# Running in production

Everything here is optional. A plain `wsp.fix(model, task, llm=...)` still works the same way.

## Results you can act on

`Result` tells you why a run stopped and what it cost:

```python
res = wsp.fix(model, task, llm="anthropic:claude-opus-5-5", pricing=(5, 25, 0.5, 6.25))
res.stop_reason  # submitted | answered | max_steps | token_budget | cost_budget | time_budget | interrupted | error
res.ok  # True for submitted / answered
res.tokens  # Usage(input_tokens, output_tokens, cached_input_tokens, cache_write_tokens, reasoning_tokens, requests)
res.cost  # USD, when pricing= is given
res.timings  # {"llm": {"calls", "seconds"}, "tool:python": {...}, ...}
res.run_id  # also sent as the OpenAI prompt_cache_key and the trace conversation id
```

`res.usage` still holds the provider's raw counters. `tokens` is the same shape for every provider.

## Budgets

```python
wsp.improve(model, goal, llm=..., token_budget=2_000_000, cost_budget=10.0, time_budget=3600, pricing=(1.25, 10))
```

When a budget runs out the LLM gets one last turn to submit, so you still get an answer and the edits so far.
`pricing` is USD per million tokens: `(input, output[, cached_input, cache_write])`, a dict, or `wsp.Pricing`.

## Failures and Ctrl-C

- Ctrl-C returns a `Result` with `stop_reason="interrupted"`: transcript, edits and usage up to that point.
- If the LLM request fails after the SDK's own retries, `wsp.SessionError` is raised. `err.result` has the
  partial result, so nothing is lost.
- Backends take `max_retries` (default 4) and `timeout` (default 900 s), passed to the official SDKs.
- Anthropic streams long requests (`stream="auto"`: when `max_tokens` > 21333), which long Opus turns need.

## Observability

Pass `callbacks=[...]`; each gets every run event (`run_start`, `step_start`, `llm_start`, `llm_end`, `tool_start`,
`tool_end`, `message`, `run_end`; see `weightspace.events`). A callback is a function or an object with `on_event`.

```python
from weightspace.events import EventLog, LoggingCallback
from weightspace.otel import OpenTelemetryCallback  # pip install "weightspace[otel]"

wsp.fix(model, task, llm=..., callbacks=[OpenTelemetryCallback(), LoggingCallback(), EventLog("events.jsonl")])
```

- `OpenTelemetryCallback` emits GenAI semantic-convention spans: `invoke_agent weightspace` per run, `chat <model>`
  per request (token usage, cache reads/writes, finish reason) and `execute_tool <name>` per tool call. Tool
  arguments are only recorded with `capture_content=True`.
- `LoggingCallback` logs to the `weightspace` logger (milestones at INFO, the rest at DEBUG).
- `EventLog` / `TranscriptWriter` append JSONL as the run goes, so a crash keeps the log. `log_path=` uses
  `TranscriptWriter`; `verbose=True` uses `ConsoleLogger`.

## Parallel tool calls

Read-only tools (`list_tensors`, `describe_tensor`, `read_tensor`, and custom tools declared with
`wsp.tool(..., parallel=True)`) that the LLM asks for in the same turn run on a thread pool
(`parallel_tools=8` workers). Anything that writes weights, runs Python or spends forward passes runs alone and
in order, so edits never race. Results always come back in the order they were asked for.

## OpenAI

- `prompt_cache_key` is set to the run id on api.openai.com, so every step of a run lands on the same cache.
  Pass `prompt_cache_key="..."` to pick your own, or `False` to turn it off.
- Any other request field goes straight through: `get_llm("openai:gpt-5", parallel_tool_calls=True,
  safety_identifier="user-123")`.
- `openresponses:<model>` runs the Responses API stateless (`store=False`) and replays encrypted reasoning.

## Replay

Every run can be replayed offline, with no tokens spent:

```python
llm = wsp.ScriptedLLM.from_transcript("runs/1/transcript.jsonl")
wsp.fix(model, task, llm=llm, probes=[...], forward_budget=10)
```

Tool outputs are recomputed, so replaying on other weights shows how the same program behaves there.
CLI: `weightspace fix model.safetensors "..." --replay runs/1/transcript.jsonl`.

## Sandbox

- Code runs under `torch.no_grad()` (`no_grad=True`), so autograd-based training of the weights fails.
- The timeout works on any thread: SIGALRM on the main thread, a watchdog elsewhere (for example inside a web
  server worker). A single long C call, like a huge matmul, finishes before the timeout fires.
- It is still not a security boundary. See [SECURITY.md](../SECURITY.md).

## Performance

Large tensors are processed in 1M-element blocks on a small shared thread pool (NumPy releases the GIL), so
nothing makes a full float64 copy of a tensor:

- `ws.diff()`, `ws.patch()` and checkpoints compare weights bitwise (a NaN that was not edited is not a change).
  The original norm used by `rel_delta` is computed once per tensor.
- Checkpoints store a sparse delta from the originals, so checkpointing a small edit to an embedding costs
  megabytes, not gigabytes.
- `describe_tensor` gets exact quantiles from one histogram pass plus a partition of the few bins that hold the
  wanted ranks, instead of sorting the whole tensor. Its text is cached by content, so describing a touched but
  unchanged tensor is free.
- The spectrum uses an SVD for small matrices and the Gram matrix in the input precision (solved in float64)
  above 256 rows or columns.

`python benchmarks/profile_hotpaths.py` times these paths on a 546M-parameter bf16 model (CPU, 8 threads):

| | 0.3.0 | now | |
|---|---|---|---|
| `describe` embedding 151936 x 1024 | 6,475 ms, 2,376 MB | 1,241 ms, 132 MB | 5.2x |
| `describe` MLP 3584 x 1024 | 341 ms | 130 ms | 2.6x |
| `write_tensor` (+ diff over 237M touched weights) | 371 ms | 80 ms | 4.6x |
| `ws.diff()` | 381 ms | 81 ms | 4.7x |
| `ws.patch()` | 170 ms | 83 ms | 2.1x |
| `ws.checkpoint()` | 245 ms, 1,807 MB | 88 ms, 8 MB | 2.8x |
| `ws.restore()` + diff | 460 ms | 147 ms | 3.1x |
| first edit of the embedding (snapshot) | 152 ms | 29 ms | 5.3x |
| agent loop, 30 steps, 60 tool calls | 9,242 ms | 373 ms | 24.8x |
