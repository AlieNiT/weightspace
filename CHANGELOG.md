# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-11-03

### Added
- Run events and callbacks (`callbacks=`): `LoggingCallback`, `EventLog`, `TranscriptWriter`, `ConsoleLogger`, and `weightspace.otel.OpenTelemetryCallback` with GenAI semantic-convention spans (`weightspace[otel]`).
- `Usage` (tokens normalized across providers, incl. cache reads/writes and reasoning) and `Pricing`; `Result.tokens`, `cost`, `stop_reason`, `run_id`, `seconds`, `timings`, `error`, `ok`.
- `token_budget`, `cost_budget` and `time_budget`: the LLM gets a last turn to submit when one runs out.
- Ctrl-C returns a partial `Result`; API failures raise `SessionError` carrying the partial result.
- Read-only tool calls of one turn run in parallel (`parallel_tools=`, `wsp.tool(..., parallel=True)`).
- `ScriptedLLM.from_transcript` to replay logged runs offline; CLI `--replay`.
- OpenAI `prompt_cache_key` per run on api.openai.com; `max_retries` / `timeout` on all backends; Anthropic streaming for long requests.
- CLI: `--probe FILE.py:FN`, `--forward-budget`, `--price`, `--max-cost`, `--max-tokens`, `--max-time`, `--events`, `--json`, `--version`; exit code 2 when a run does not finish cleanly.
- `py.typed` marker.

### Changed
- `ws.diff()` and `ws.patch()` run in blocks (about 6x faster and 50x less memory on large tensors); `describe_tensor` is cached and uses a Gram-matrix spectrum for tall/wide matrices.
- The sandbox timeout works off the main thread and without SIGALRM; sandbox code runs under `torch.no_grad()`.
- `agent.py` split into `tools.py`, `result.py`, `events.py` and `usage.py` (same public API).

- `openresponses:<model>` backend (`OpenResponses`) for the [Open Responses](https://www.openresponses.org) spec (OpenAI Responses API), replaying encrypted reasoning between turns. `openai-responses:` is kept as an alias.
- `llms.txt` with a short usage guide for coding agents.
- `speed=` on `Session`, `wsp.read/fix/write/improve` and backends (CLI `--speed` / `--fast`) for provider fast mode: Anthropic `speed`, OpenAI `service_tier`. Saved in `result.json`; the Exp. 8 script takes `--speed fast`.
- Anthropic `thinking=False` falls back to `effort="low"` on models that can't disable thinking (Opus 5.5); `usage.thinking_tokens` is now recorded.

### Added
- Exp. 8 replication harness for Qwen3.5-0.8B in `experiments/exp8`.
- Open-source project files, docs and a wider CI matrix.

## [0.1.0] - 2026-11-03

### Added
- `WeightSpace` with NumPy, PyTorch, Hugging Face, SafeTensors and `.pt` stores; tracked edits, diffs, checkpoints and sparse patches.
- `read`, `fix`, `write` and `improve` sessions with a persistent Python sandbox, budgeted probes and text mode.
- OpenAI-compatible (OpenAI, OpenRouter, vLLM, SGLang, Ollama) and Anthropic backends.
- `thinking` switch (off / low / medium / high / xhigh / max), with adaptive thinking and prompt caching on Anthropic.
- `weightspace` CLI and examples.

[Unreleased]: https://github.com/leofmarciano/weightspace/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/leofmarciano/weightspace/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/leofmarciano/weightspace/releases/tag/v0.1.0
