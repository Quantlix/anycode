# Contributing to AnyCode

AnyCode accepts focused bug fixes, documentation improvements, tests, and discussed enhancements. A contribution is ready to merge when its behavior, tests, documentation, compatibility impact, and changelog entry agree.

## Before you start

- Search existing issues and pull requests before opening a duplicate.
- Bug fixes may start with a pull request when the problem and expected behavior are clear.
- Discuss new features, public API changes, persisted-format changes, large dependencies, and cross-cutting architecture changes in an issue before implementation.
- Report vulnerabilities through the private process in [SECURITY.md](SECURITY.md), not a public issue.

Maintainers may close an unsolicited feature pull request when the project has not agreed to its contract or long-term maintenance cost.

## Development setup

AnyCode requires Python 3.12 or newer and uses `uv` for dependency management.

```bash
git clone https://github.com/Quantlix/anycode.git
cd anycode
uv sync --locked --group dev
```

Create a short-lived branch from the current `main` branch:

```bash
git fetch origin
git switch --create fix/short-description origin/main
```

Use `feat/`, `fix/`, `docs/`, `refactor/`, `test/`, or `chore/` followed by a short lowercase description. Keep one coherent outcome per branch.

## Make the change

Follow the repository conventions in [AGENTS.md](AGENTS.md):

- Keep agent execution, provider calls, and tools async.
- Keep Pydantic models frozen and create replacements instead of mutating them.
- Import public APIs from the package root in examples and user documentation.
- Preserve provider, tool, and storage protocol boundaries.
- Add tests for behavior changes and regressions.
- Update docstrings, guides, examples, and reference pages in the same pull request as user-visible behavior.
- Add a user-facing entry under `[Unreleased]` in [CHANGELOG.md](CHANGELOG.md) for notable changes.

Commit subjects use Conventional Commit prefixes such as `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, or `chore:`.

## Compatibility review

The following are public contracts:

- Names exported through `anycode.__all__`.
- Documented function signatures, defaults, exceptions, and result semantics.
- CLI commands and documented output intended for automation.
- Declarative YAML and TOML configuration.
- Checkpoints, durable run records, transcripts, and turn checkpoints.

Read the [compatibility and versioning policy](https://quantlix.github.io/anycode/latest/reference/compatibility/) before changing one of these surfaces. A breaking change needs prior design agreement, the correct version increment, migration and rollback notes, compatibility tests, documentation, and a prominent changelog entry.

## Run tests

Run a focused test while iterating, then run the complete local gate before marking the pull request ready:

```bash
uv run python scripts/check_versions.py
uv run python -m ruff check .
uv run python -m ruff format --check src/
uv run python -m pyright
uv run python -m pytest
uv run python -m mkdocs build --strict
```

The default pytest configuration excludes service-backed integration tests. To run the same Redis and ChromaDB suite used in CI:

```bash
docker compose up -d --wait redis chromadb
uv run python -m pytest tests/integration -m integration --strict-markers --tb=short
docker compose down
```

Packaging, dependency, CLI, and release changes also require:

```bash
uv run python -m build --no-isolation
uv run python -m twine check --strict dist/*
```

CI runs the complete non-integration suite on Python 3.12 and 3.13 across Linux, Windows, and macOS. It separately tests every optional extra, service integrations, documentation, and installed wheel behavior.

## Documentation changes

Documentation source lives in `site_docs/` and builds with MkDocs Material. Preview it with:

```bash
uv run python -m mkdocs serve
```

Every page must be registered in `mkdocs.yml`, use one clear H1, and describe current behavior. Keep `site_docs/llms.txt` current when an important page is added, renamed, or moved. The [documentation contributor guide](https://quantlix.github.io/anycode/latest/contributing/docs-guide/) covers page types, frontmatter, API reference generation, and versioned deployment.

## Open a pull request

Complete the pull request template with:

- The problem and observable outcome.
- Compatibility and version impact.
- Tests and exact validation commands.
- Documentation and changelog impact.
- Migration, rollback, persistence, or security considerations.

Use a draft pull request for work in progress. Mark it ready only when the change is scoped, tested, documented, and free of unrelated edits. Maintainers normally squash-merge approved pull requests after required checks pass and review conversations are resolved.

## Review and conduct

Review may request changes to behavior, tests, API shape, documentation, or scope. Keep discussion technical and respectful, assume good intent, and focus on the effect of the change. Maintainer decisions follow [MAINTAINERS.md](MAINTAINERS.md).
