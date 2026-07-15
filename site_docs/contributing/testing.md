---
title: "Testing and Continuous Integration"
description: "Run AnyCode quality checks, cross-platform tests, Docker-backed integration tests, optional-extra isolation, and package validation locally and in CI."
keywords: AnyCode tests, pytest, GitHub Actions, CI matrix, optional dependencies, Redis integration, ChromaDB integration
---

# Testing and Continuous Integration

AnyCode separates source quality, portable behavior, optional dependency packaging, service-backed behavior, and built distributions into distinct CI jobs. This keeps each failure attributable while testing the supported Python and operating-system surface.

## Local quality gate

Install the locked development environment, then run the same deterministic checks used by CI:

```bash
uv sync --locked --group dev
uv run python scripts/check_versions.py
uv run python -m ruff check .
uv run python -m ruff format --check src/
uv run python -m pyright
uv run python scripts/generate_contract_schemas.py --check
uv run python -m pytest
uv run python examples/36_runtime_baseline.py
uv run python examples/37_semantic_contract.py
uv run python -m mkdocs build --strict
uv run python scripts/check_docs.py
```

The default pytest configuration excludes tests marked `integration`. Unit, runtime, MCP subprocess, CLI, provider-adapter, and persistence tests still run without live provider credentials. Deterministic provider tests use `FakeAdapter`.

## Service integration tests

Redis and ChromaDB tests use the host ports declared in `docker-compose.yml`. Start only the services the current integration suite consumes:

```bash
docker compose up -d --wait redis chromadb
uv run python -m pytest tests/integration -m integration --strict-markers --tb=short
docker compose down
```

An explicit `-m integration` overrides the default non-integration selection. Service fixtures skip when their endpoint is unavailable, while the CI job also checks both endpoints before pytest starts so missing infrastructure cannot produce a false green build.

## CI matrix

| Job | Environment | Contract |
| --- | --- | --- |
| `Quality` | Ubuntu, Python 3.12 | Version consistency, locked dependencies, Ruff, formatting, Pyright, semantic schemas, and reproducible contract evidence |
| `Tests` | Ubuntu, Windows, and macOS; Python 3.12 and 3.13 | Complete non-integration suite on every supported platform/runtime pair |
| `Optional extra` | Ubuntu, Python 3.12 | Core-only install plus an isolated install and import smoke test for every declared extra |
| `Integration services` | Ubuntu, Python 3.12, Redis, and ChromaDB | All tests marked `integration`, with healthy external endpoints required |
| `Package Validation` | Ubuntu, Python 3.12 and 3.13 | Wheel/sdist build, strict metadata checks, core-only wheel import, and CLI-extra smoke test |
| `Documentation` | Ubuntu, Python 3.12 | Strict site build, generated public API coverage, tool/example claims, metadata, and curated agent links |

Source quality, test, optional-extra, integration, and build environments are resolved from `uv.lock` with `--locked`. A pull request whose dependency metadata requires a lockfile update fails instead of silently resolving a different environment. Fresh wheel smoke tests use `pip` so they also validate the dependency ranges published in wheel metadata.

## Adding an optional extra

When adding or renaming an entry under `[project.optional-dependencies]`:

1. Update `pyproject.toml` and regenerate `uv.lock`.
2. Add an isolated entry to the `optional-extras` matrix in `.github/workflows/ci.yml`.
3. Import both the third-party dependency and the AnyCode module that owns the feature.
4. Run `uv run python -m pytest tests/test_ci.py`.

The CI contract test compares the workflow matrix with package metadata. Missing and stale extra entries fail locally and in CI.

## Built-package checks

Source-tree imports can pass while a wheel omits a module or dependency. `Package Validation` therefore installs the built wheel into a fresh virtual environment without extras and imports `AnyCode` before installing the same wheel with `[cli]` and invoking `anycode version`. Release workflows retain their own full gate before publishing.

## Documentation contract checks

The strict MkDocs build validates navigation, internal links, snippets, and generated API directives. `scripts/check_docs.py` runs after the build and verifies claims derived from the repository:

- Every package-root export has generated API coverage.
- The README lists exactly the registered built-in tools.
- Numbered example files are contiguous and documented counts are current.
- Every Markdown page has title and description metadata.
- Every versioned documentation link in `site_docs/llms.txt` maps to a source page.

Final release tags deploy docs. Pull requests, `main`, and manual dispatches validate docs without modifying published release versions.

## Contract suites and fault evidence

The [contract test conventions](contract-tests.md) define reusable suites, immutable golden fixtures, state-machine properties, real process-death tests, and failure injection around durable boundaries. Changes to a public or persisted contract also use the [ADR process](architecture-decisions.md) and its [copyable template](adr-template.md).

The deterministic runtime baseline is not a pass/fail latency benchmark on shared CI. Its schema, workload sizes, event counts, checkpoint production, and monotonic context growth are contract-tested; elapsed observations remain comparable only on controlled, like-for-like runners. See [runtime contracts and baseline](../reference/runtime-contracts.md#reproduce-the-baseline).
