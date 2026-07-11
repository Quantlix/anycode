---
title: CLI Reference — AnyCode Python Framework
description: "AnyCode CLI reference: scaffold projects, run YAML and TOML team configs, inspect tools and providers, run eval suites, and manage durable run stores."
keywords: AnyCode CLI, anycode command, anycode run, anycode init, anycode inspect, anycode eval, anycode runs, CLI reference
---

# CLI Reference

The `anycode` command-line interface scaffolds projects, runs team configs, inspects your runtime, executes eval suites, and manages durable run stores. This page documents every command, its purpose, and its common invocations.

| Command | Purpose |
| --- | --- |
| [`anycode version`](#anycode-version) | Print package, Python, and provider dependency information. |
| [`anycode init`](#anycode-init) | Scaffold a starter project with config, code, and tooling. |
| [`anycode run`](#anycode-run) | Run a YAML or TOML team config. |
| [`anycode inspect`](#anycode-inspect) | Inspect tools, providers, plugins, config, and team. |
| [`anycode eval`](#anycode-eval) | Run evaluation scenarios from a suite file. |
| [`anycode runs`](#anycode-runs) | Inspect durable run stores (list, show, tail, audit, sweep). |

Install the CLI extra before using the `anycode` command:

```bash
uv add "anycode-py[cli]"
```

Inside this repository, run commands through `uv`:

```bash
uv run anycode version
```

## `anycode version`

Print package, Python, and provider dependency information.

```bash
uv run anycode version
```

## `anycode init`

Create a starter project with `team.yaml`, `main.py`, `.env.example`, a `tools/` package, and `.gitignore`.

```bash
uv run anycode init my-project
```

Use `--force` to overwrite scaffolded files in an existing directory.

!!! tip "First run"
    After scaffolding, copy `.env.example` to `.env`, add a provider API key, then run the project's config with [`anycode run`](#anycode-run).

## `anycode run`

Run a YAML or TOML team config.

```bash
uv run anycode run team.yaml
```

The config can include agents, tasks, cost controls, routing, RAG, reflection, verification, and context engineering blocks. See the [YAML config guide](../guides/yaml-config.md) for the full schema.

## `anycode inspect`

Inspect runtime configuration and available components.

Common forms:

```bash
uv run anycode inspect tools
uv run anycode inspect providers
uv run anycode inspect plugins
uv run anycode inspect config team.yaml
uv run anycode inspect team team.yaml
```

!!! note "Validate before you run"
    `inspect config` and `inspect team` resolve a config without executing it — a quick way to catch schema or wiring errors before a full run.

## `anycode eval`

Run evaluation scenarios from a suite file. Deterministic fake responses can be used for CI and harness tests without provider credentials.

```bash
uv run anycode eval tests/fixtures/eval/runtime_reliability_deterministic.yaml
```

## `anycode runs`

Inspect durable run stores:

```bash
uv run anycode runs list --root .anycode/runs
uv run anycode runs show <run-id> --root .anycode/runs
uv run anycode runs tail <run-id> --root .anycode/runs
uv run anycode runs audit <run-id> --root .anycode/runs
uv run anycode runs sweep --root .anycode/runs
uv run anycode runs sweep --root .anycode/runs --retention-days 30 --max-runs 1000
```

Use these commands when workflows checkpoint turns, pause for wake conditions, or need an audit trail. Retention is disabled unless `--retention-days` or `--max-runs` is supplied; only terminal runs are eligible. See [Production controls](../guides/production-controls.md) for durable-run configuration.

## See also

- [Quickstart](../getting-started/quickstart.md) — run your first agent and team.
- [Installation](../getting-started/installation.md) — install AnyCode and the CLI extra.
- [YAML config guide](../guides/yaml-config.md) — the schema behind `anycode run`.
- [Public API reference](public-api.md) — drive the same runtime from Python.
