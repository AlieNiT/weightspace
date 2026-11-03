# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-11-03

### Added
- `openresponses:<model>` backend (`OpenResponses`) for the [Open Responses](https://www.openresponses.org) spec (OpenAI Responses API), replaying encrypted reasoning between turns. `openai-responses:` is kept as an alias.
- `speed=` on `Session`, `wsp.read/fix/write/improve` and backends (CLI `--speed` / `--fast`) for provider fast mode: Anthropic `speed`, OpenAI `service_tier`. Saved in `result.json`.
- `usage.thinking_tokens` is recorded for Anthropic models.
- `llms.txt` with a short usage guide for coding agents.
- Exp. 8 replication harness for Qwen3.5-0.8B in `experiments/exp8`.
- Open-source project files, docs and a wider CI matrix.

### Changed
- Anthropic `thinking=False` falls back to `effort="low"` on models that can't disable thinking (Opus 5.5).

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
