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
| [`anycode api`](#anycode-api) | Print the public API surface as a table or as JSON. |
| [`anycode eval`](#anycode-eval) | Run (`run`) or compare (`compare`) evaluation suites. |
| [`anycode runs`](#anycode-runs) | Inspect durable run stores (list, show, tail, audit, sweep). |
| [`anycode harness`](#anycode-harness) | Emit a harness manifest (`manifest`) or run an experimental evolution sweep (`evolve`). |

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

## `anycode api`

Print the public API surface. Written for humans skimming and for AI coding agents that
should not have to read the source tree to learn the API.

```bash
uv run anycode api --core        # the 15 symbols covering most use, with live signatures
uv run anycode api Agent         # one symbol, full signature, kind, module, and summary
uv run anycode api --compact     # every symbol, names and summaries only
uv run anycode api --json        # machine-readable, stable key order
uv run anycode api --kind model  # filter by class, model, protocol, function, type, constant, module
```

| Option | Effect |
| --- | --- |
| *(no argument)* | Every public symbol, grouped by module |
| `<Symbol>` | One symbol in full |
| `--core` | Only `anycode.CORE_SURFACE` |
| `--compact` | Drop signatures |
| `--json` | Emit JSON instead of a table |
| `--kind <kind>` | Filter by symbol kind |

Approximate output sizes: `--core` 4.6 KB, `--compact` 29 KB, full JSON 139 KB. Start with
`--core`; it is the artifact meant to go into a prompt.

The same data is available in Python through `anycode.describe()`, which returns an
`ApiMap`, or `anycode.describe("Agent")`, which returns a single `ApiEntry`.

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

Run and compare evaluation suites. The command has two subcommands, `run` and `compare`. Deterministic fake responses can be used for CI and harness tests without provider credentials.

### `anycode eval run`

Execute every scenario in a suite file against the configured provider and write a JSON report.

```bash
uv run anycode eval run tests/fixtures/eval/runtime_reliability_deterministic.yaml
```

| Option | Default | Effect |
| --- | --- | --- |
| `--variant`, `-v` | `baseline` | Label for this harness variant, recorded in the report. |
| `--output`, `-o` | `artifacts/eval/report.json` | Where to write the JSON report. |
| `--name`, `-n` | `default` | Suite name recorded in the report. |
| `--markdown`, `-m` | off | Also print a markdown summary table. |

The command exits non-zero if any scenario fails, so it doubles as a CI gate.

### `anycode eval compare`

Diff two reports and surface regressions, improvements, and newly added scenarios.

```bash
uv run anycode eval compare artifacts/eval/baseline.json artifacts/eval/candidate.json
```

By default it exits non-zero when the candidate regresses against the baseline. Pass `--no-fail-on-regression` to report differences without failing.

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

## `anycode harness`

Inspect and — experimentally — evolve the harness itself. This is an advanced surface used by evaluation tooling; most projects never need it. See the [adaptive harness section of the Public API](public-api.md#adaptive-harness-advanced).

### `anycode harness manifest`

Emit a deterministic manifest of every editable harness component for a configured team.

```bash
uv run anycode harness manifest --config team.yaml
```

| Option | Default | Effect |
| --- | --- | --- |
| `--config`, `-c` | *(required)* | AnyCode config file to build the manifest from. |
| `--output`, `-o` | `artifacts/harness/manifest.json` | Where to write the manifest JSON. |
| `--notes` | *(none)* | Optional note attached to the manifest. |
| `--pretty` | off | Pretty-print the (secret-redacted) manifest to stdout. |

### `anycode harness evolve`

Run a controlled, **dry-run-by-default** evolution sweep over an eval suite. It never writes back to your repository; accepted changes are emitted as reviewable JSON patches.

```bash
uv run anycode harness evolve tests/fixtures/eval/runtime_reliability_deterministic.yaml
```

| Option | Default | Effect |
| --- | --- | --- |
| `--max-iterations` | `3` | Maximum proposal cycles. |
| `--dry-run` / `--apply` | `--dry-run` | `--apply` is reserved for experimental tooling and currently exits early. |
| `--patch-dir` | `artifacts/harness/patches` | Where to emit reviewable patches. |

!!! warning "Experimental"
    Harness evolution is experimental and intended for evaluation environments, not production runs. Keep it in dry-run mode and review every emitted patch by hand.

## See also

- [Quickstart](../getting-started/quickstart.md) — run your first agent and team.
- [Installation](../getting-started/installation.md) — install AnyCode and the CLI extra.
- [YAML config guide](../guides/yaml-config.md) — the schema behind `anycode run`.
- [Public API reference](public-api.md) — drive the same runtime from Python.
