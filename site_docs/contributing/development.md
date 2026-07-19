---
title: "AnyCode Development Workflow for Contributors"
description: "Set up AnyCode, create focused branches, implement code and docs together, run every local quality gate, and submit a pull request for maintainer review."
keywords: AnyCode contributing, Python development workflow, AnyCode pull request, uv pytest ruff pyright, topic branch
---

# Development Workflow

AnyCode uses short-lived topic branches and pull requests into `main`. A change is ready for review when its behavior, tests, documentation, compatibility impact, and changelog entry tell the same story.

The repository-level [`CONTRIBUTING.md`](https://github.com/Quantlix/anycode/blob/main/CONTRIBUTING.md) is the current contribution policy. This page provides the same workflow in the versioned documentation.

## Choose the right path

| Change | Start with | Why |
| --- | --- | --- |
| Reproducible bug with a narrow fix | Issue or focused pull request | The expected behavior is already defined |
| Documentation correction | Focused pull request | The code or current contract decides the answer |
| New public feature or provider | Issue or design discussion | New APIs create a long-term support obligation |
| Breaking API or behavior change | Design discussion | Maintainers must agree on compatibility, migration, and version impact |
| Persisted format or config change | Design discussion | Readers, writers, migration, and rollback need a coordinated contract |
| Security vulnerability | Private vulnerability report | Public details can expose users before a fix exists |

Search open issues and pull requests first. For a feature, describe the user problem and proposed contract before writing a large implementation.

## Set up the repository

AnyCode requires Python 3.12 or newer and uses `uv` for dependency and environment management.

```bash
git clone https://github.com/Quantlix/anycode.git
cd anycode
uv sync --locked --group dev
```

Run a small baseline before editing:

```bash
uv run python -m pytest tests/test_compatibility.py
uv run python -m ruff check .
```

If the task starts from a reported failure, run that exact failing test or command instead. A clean baseline makes regressions attributable to the change.

## Create a topic branch

Start from the current upstream branch:

```bash
git fetch origin
git switch --create fix/short-description origin/main
```

Use one of these prefixes:

| Prefix | Purpose |
| --- | --- |
| `feat/` | Backward-compatible user-facing capability |
| `fix/` | Bug fix |
| `docs/` | Documentation-only correction or guide |
| `refactor/` | Internal behavior-preserving change |
| `test/` | Test or evaluation-harness change |
| `chore/` | Dependency, CI, packaging, or repository maintenance |

Keep a branch focused on one outcome. You may rewrite your own topic-branch history with `git push --force-with-lease`; never rewrite `main`, release tags, or shared maintenance branches.

## Implement the complete change

Use the nearest implementation and test as the starting point. Follow the architecture and style rules in [`AGENTS.md`](https://github.com/Quantlix/anycode/blob/main/AGENTS.md), especially these contracts:

- Agent execution, provider calls, and tool calls are async.
- Pydantic models are frozen; create a replacement instead of mutating an instance.
- `LLMAdapter`, storage protocols, and the tool registry are extension boundaries.
- Public examples import from `anycode`, not internal modules.
- External data is validated at the boundary and errors do not expose credentials.

The pull request should update each affected surface:

| Change surface | Required companion work |
| --- | --- |
| Behavior or bug fix | Regression test that fails without the fix |
| Public signature, default, exception, or result | Docstring and API/reference documentation |
| User workflow | Relevant guide or tutorial and a runnable example when useful |
| Public export | `anycode.__all__`, compatibility review, and generated API inventory verification |
| YAML/TOML field | Loader validation, configuration reference, example config, and format-version review |
| Checkpoint or durable data | Old-reader fixture, future-version rejection test, migration, and rollback guidance |
| User-visible change | Entry under `[Unreleased]` in `CHANGELOG.md` |

Do not commit `site/`, `dist/`, local virtual environments, secrets, or generated run artifacts.

## Validate while you work

Run the smallest test that can disprove your current implementation first:

```bash
uv run python -m pytest tests/test_target.py -k test_name
```

Before requesting final review, run the full local gate:

```bash
uv run python scripts/check_versions.py
uv run python -m ruff check .
uv run python -m ruff format --check src/
uv run python -m pyright
uv run python -m pytest
uv run python -m mkdocs build --strict
uv run python scripts/check_docs.py
```

Package, dependency, CLI, and release changes also require:

```bash
uv run python -m build --no-isolation
uv run python -m twine check --strict dist/*
```

The [testing and CI guide](testing.md) explains the cross-platform matrix, optional-extra isolation, Docker-backed integration suite, and built-wheel checks.

## Write the commit and pull request

Commit subjects follow Conventional Commits:

```text
fix: preserve tool results during handoff
feat: add provider capacity limits
docs: document checkpoint migration
```

Explain the reason for the change when it is not obvious from the subject. Avoid mixing formatting or unrelated cleanup into the same commit.

The pull request template asks for:

- The problem and observable result.
- Public-contract and version impact.
- Focused and full validation commands.
- Documentation and changelog updates.
- Migration, rollback, persisted-data, and security considerations.

Use a draft pull request while the contract or implementation is still changing. Mark it ready after the branch is scoped, tested, documented, and free of unrelated edits.

## Review and merge

Reviewers check behavior, edge cases, failure handling, typing, compatibility, documentation, release notes, and test evidence. Respond to each review thread and request another review after a material rewrite.

Required checks must pass before merge. Ordinary pull requests are squash-merged to keep `main` readable, and the source branch is deleted afterward. Maintainers handle backports through a separate pull request against the maintained release line.

## Next steps

- [Testing and continuous integration](testing.md)
- [Documentation contributor guide](docs-guide.md)
- [Compatibility and versioning](../reference/compatibility.md)
- [Maintainer governance](maintainers.md)
