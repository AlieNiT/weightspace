# Contributing

Thanks for helping out. Bug reports, new adapters, providers and experiments are all welcome.

## Setup

```bash
git clone https://github.com/leofmarciano/weightspace && cd weightspace
pip install -e ".[all,dev]"
pre-commit install   # optional: runs ruff on every commit
```

## Before opening a PR

```bash
ruff check . && ruff format --check .
pytest
```

Tests run offline: use `wsp.ScriptedLLM` or a fake client instead of real API calls. Don't commit API keys, model weights or large run artifacts (`runs/` is ignored).

## Conventions

- Commit messages and PR titles follow [Conventional Commits](https://www.conventionalcommits.org/) in English: `feat(adapters): ...`, `fix: ...`, `docs: ...`, `test: ...`, `ci: ...`.
- Keep PRs small and focused, and add a test for any behaviour change.
- New LLM providers subclass `weightspace.llm.LLM`; new weight formats implement the `Store` interface in `space.py`.
- Add a line to `CHANGELOG.md` under "Unreleased".

## Releases

Maintainers bump `version` in `pyproject.toml`, move the changelog entries under the new version, and push a `vX.Y.Z` tag. The `publish` workflow uploads to PyPI with Trusted Publishing.
