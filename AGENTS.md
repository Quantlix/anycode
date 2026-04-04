# Repository Guidelines

- **Repo**: [github.com/Quantlix/anycode](https://github.com/Quantlix/anycode)
- In chat replies, file references must be repo-root relative only (e.g. `src/anycode/core/agent.py:42`); never absolute paths or `~/…`.

---

## Project Overview

AnyCode is a multi-agent AI orchestration framework for Python. It composes autonomous AI agents into collaborative teams with inter-agent messaging, dependency-aware task scheduling, and provider-agnostic LLM integration (Anthropic, OpenAI).

| Detail          | Value                                    |
|-----------------|------------------------------------------|
| **Package**     | `anycode-py` (PyPI)                      |
| **License**     | MIT                                      |
| **Python**      | ≥ 3.12                                   |
| **Build**       | hatchling                                |
| **Repo**        | `github.com/Quantlix/anycode`            |

---

## Project Structure & Module Organization

Source code lives under `src/anycode/` and follows a layered architecture:

```
src/anycode/
├── core/           # Orchestration engine, agents, runner, pool, scheduler
├── collaboration/  # Team coordination, message bus, shared memory, KV store
├── tasks/          # Task queue, dependency resolution (topological sort)
├── tools/          # Tool registry, executor, built-in tools (bash, file_*, grep)
├── providers/      # LLM adapters (Anthropic, OpenAI) via Protocol interface
├── telemetry/      # OpenTelemetry tracing, metrics, structured lifecycle events
├── guardrails/     # Budget enforcement, content validators, lifecycle hooks
├── structured/     # Schema-constrained LLM output via Pydantic models
├── helpers/        # Concurrency gate (semaphore), token usage tracking
└── types.py        # All Pydantic models — frozen (immutable) by default
```

- **Tests**: `tests/` — colocated as `tests/test_<module>.py`.
- **Examples**: `examples/` — numbered, self-contained scripts (`01_solo_worker.py`, …).
- **Scripts**: `scripts/` — setup, lint, test automation.

### Architecture Boundaries

- **Provider Protocol**: `LLMAdapter` is a `Protocol` — swap LLM providers without touching business logic.
- **Immutable models**: Every Pydantic model uses `frozen=True`. Never mutate; always create new instances.
- **Import boundaries**: Public API is exported from `src/anycode/__init__.py`. Internal modules should not be imported directly by consumers.
- **Tool boundary**: Built-in tools live under `src/anycode/tools/`. User-defined tools are registered at runtime via the `ToolRegistry`, not by patching internal modules.

### Data Flow

```
AnyCode (orchestrator)
  → Team (agent coordination + messaging)
    → AgentPool (bounded concurrency via Semaphore)
      → AgentRunner (LLM ↔ tool conversation loop)
        → LLMAdapter (Protocol: AnthropicAdapter | OpenAIAdapter)
        → ToolExecutor (validates inputs via Pydantic, runs tools)
```

### Key Patterns

- **Wavefront execution**: Tasks execute in waves; each wave is all tasks whose dependencies are satisfied.
- **Topological sort**: `get_task_dependency_order()` uses Kahn's algorithm for dependency resolution.
- **Cascading failure**: `TaskQueue._propagate_failure()` marks downstream dependents as blocked.
- **4 scheduling strategies**: `round-robin`, `least-busy`, `capability-match`, `dependency-first`.

---

## Build, Test & Development Commands

Runtime baseline: **Python 3.12+** — always use **`uv`** as the package manager (never `pip`).

```bash
# Dependencies
uv sync                                    # install all deps
uv add <pkg>                               # add runtime dependency
uv add --group dev <pkg>                   # add dev dependency
uv add --optional telemetry <pkg>          # add to optional extras group

# Tests
uv run pytest                              # run full suite
uv run pytest tests/                       # explicit test directory
uv run pytest tests/test_foo.py            # single file
uv run pytest -k "test_name"               # single test by name

# Lint & format
uv run ruff check src/                     # lint
uv run ruff format src/                    # auto-format
uv run ruff format --check src/            # verify formatting (CI)

# Type check
uv run pyright
```

### Pre-push gate

Run the full verification before pushing to `main`:

```bash
uv run ruff check src/ && uv run ruff format --check src/ && uv run pyright && uv run pytest
```

If deps are missing (`ModuleNotFoundError`, `command not found`), run `uv sync` first, then retry the exact command once. If it still fails, report the command and first actionable error.

---

## Coding Style & Naming Conventions

| Convention       | Standard                                                                               |
|------------------|----------------------------------------------------------------------------------------|
| **Line length**  | 150 (configured in `pyproject.toml` via ruff)                                          |
| **Linter**       | ruff — `select = ["E", "F", "I", "UP"]`                                                |
| **Type checker** | pyright — `typeCheckingMode = "standard"`                                               |
| **Imports**      | Sorted by ruff (isort rules via `"I"` selector)                                        |
| **Async**        | All agent execution, tool calls, and LLM interactions are async                         |
| **Immutability** | All Pydantic models: `model_config = ConfigDict(frozen=True)` — create new, never mutate |

### Naming

| Element         | Convention          | Example                                |
|-----------------|---------------------|----------------------------------------|
| Classes         | `PascalCase`        | `AgentRunner`, `ToolExecutor`          |
| Functions       | `snake_case`        | `create_task`, `run_agent`             |
| Constants       | `UPPER_SNAKE_CASE`  | `BUILT_IN_TOOLS`, `EMPTY_USAGE`        |
| Private symbols | Leading underscore  | `_propagate_failure`, `_EventBus`      |
| Protocols       | Descriptive nouns   | `LLMAdapter`, `MemoryStore`            |

### Code Quality Rules

- **No magic numbers** — extract numeric literals into named `UPPER_SNAKE_CASE` constants at module top or shared constants location.
- **No planning references in code** — never add comments referencing backlogs, phases, sprints, or planning docs (e.g. `# Phase 1 features`). Comments explain *why*, not project management.
- **Clean comments** — only where logic is non-obvious. No redundant restating of what code does.
- **Constants organization** — group related constants together with a clear section; use descriptive names.
- **Prefer `Result`-style outcomes** — use explicit return types for recoverable errors rather than raising bare exceptions in library code.
- **Discriminated unions** — prefer when parameter shape changes runtime behavior.
- **Avoid `Any`** — prefer real types, `Protocol`, or `Unknown` except at true FFI boundaries.

### Module Organization

- Keep files focused: 200–400 lines typical, 800 max.
- One primary class per module (e.g. `agent.py` → `Agent`, `runner.py` → `AgentRunner`).
- Related helpers may coexist (e.g. `task.py` has `create_task`, `is_task_ready`, `get_task_dependency_order`).
- Export public API from package `__init__.py`.
- Extract helpers instead of creating "V2" module copies.

---

## Testing Guidelines

| Setting             | Value                                          |
|---------------------|------------------------------------------------|
| **Framework**       | pytest + pytest-asyncio                        |
| **Async mode**      | `asyncio_mode = "auto"` — no manual decoration |
| **Test directory**  | `tests/`                                       |
| **Coverage target** | 80%+ lines/branches                            |
| **Run command**     | `uv run pytest`                                |

### Test Conventions

- **File naming**: `tests/test_<module>.py` — match source module names.
- **Isolation**: No shared mutable state between tests.
- **Fixtures**: Use `pytest.fixture` for reusable test setup.
- **LLM mocking**: Mock LLM responses for deterministic provider tests — no real API calls in unit tests.
- **Pydantic edge cases**: Validate frozen model construction and rejection of mutation.
- **Cleanup**: Tests must clean up timers, env vars, globals, mocks, temp dirs, and module state.
- **Scoped tests prove the change** — `uv run pytest` remains the default landing bar; scoped tests do not replace full-suite gates by default.

---

## Security & Configuration

- **API keys**: NEVER hardcode in source — use environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`).
- **No secrets in AGENTS.md / CLAUDE.md** — use `.env` files (git-ignored) for local API keys.
- **Subprocess safety**: `tools/bash.py` executes shell commands — always validate/sanitize inputs at system boundaries.
- **Provider adapters**: API keys passed at runtime via `create_adapter(provider, api_key=...)` — keys must come from env.
- **Dependencies**: Pin major versions; review new deps for supply chain risk.
- **Input validation**: Validate all external data (API responses, user input, file content) at system boundaries using Pydantic schemas.
- **Error messages**: Must not leak sensitive data (API keys, internal paths, stack traces to end users).

---

## Git Workflow & Commit Conventions

- **Branch strategy**: Feature branches off `main`.
- **Commit format**: Conventional commits — `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`.
- **Commit messages**: Concise, action-oriented (e.g. `feat: add wavefront task scheduler`).
- **Group related changes** — avoid bundling unrelated refactors.
- **Pre-push**: Run the full verification gate (lint, format, type-check, tests).

---

## Examples

Example files live in `examples/` with numbered, descriptive names:

| File                        | Purpose                                                      |
|-----------------------------|--------------------------------------------------------------|
| `01_solo_worker.py`         | One-shot, streaming, and conversational agent usage          |
| `02_crew_workflow.py`       | Multi-agent collaboration with coordinated crew              |
| `03_staged_pipeline.py`     | Dependency graph with wavefront task execution               |
| `04_hybrid_tooling.py`      | Hybrid crew with user-defined tool functions                 |
| `05_production_features.py` | Observability, guardrails, structured output with live calls |

### Example Conventions

- **No planning references** in filenames or headers.
- **End-to-end by default** — examples make real LLM API calls using keys from `.env` via `python-dotenv`.
- **Provider auto-detection** — detect `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` from `.env` and select provider/model accordingly.
- **Self-contained** — each example must be runnable standalone: `uv run python examples/<file>.py`.

---

## Collaboration & Safety Notes

- When answering questions, respond with high-confidence answers only — verify in code; do not guess.
- For file references in responses, use repo-root-relative paths only.
- Never add comments referencing internal project management (backlogs, sprints, phase names).
- Agents must not modify baseline, snapshot, or expected-failure files to silence checks without explicit approval.
- Pure test additions/fixes generally do **not** need a changelog entry unless they alter user-facing behavior.

---