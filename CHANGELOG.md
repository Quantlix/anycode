# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-04-05

### Added

- **Pluggable Memory module** (`src/anycode/memory/`) — layered memory system with persistent KV stores and semantic vector search.
  - `SQLiteStore` — async SQLite-backed `MemoryStore` with WAL mode, metadata tracking, and `created_at`/`updated_at` timestamps.
  - `RedisStore` — Redis-backed `MemoryStore` for distributed deployments (optional `[redis]` extra).
  - `InMemoryVectorStore` — TF-IDF + cosine similarity vector search with zero external dependencies.
  - `ChromaDBVectorStore` — embedding-backed vector search via ChromaDB (optional `[vector]` extra).
  - `CompositeMemory` — unified interface querying both KV and vector stores with auto-indexing support.
  - `create_memory_store()` factory for config-driven backend creation from `MemoryConfig`.
- **Workflow Checkpointing module** (`src/anycode/checkpoint/`) — crash recovery for long-running DAG-based agent workflows.
  - `CheckpointManager` — automatic checkpoint creation after each execution wave, spec-change detection via SHA-256 hash, and configurable auto-pruning.
  - `FilesystemCheckpointStore` — human-readable JSON checkpoint files with atomic writes (tmp → rename).
  - `SQLiteCheckpointStore` — WAL-mode SQLite backend for high-concurrency checkpoint storage.
  - `serialize_checkpoint()` / `deserialize_checkpoint()` — deterministic round-trip serialization supporting all LLM message content types (`TextBlock`, `ToolUseBlock`, `ToolResultBlock`, `ImageBlock`).
- **Human-in-the-Loop module** (`src/anycode/hitl/`) — approval gates for enterprise-grade agent workflows.
  - `ApprovalManager` — config-driven approval enforcement with tool/task filtering and audit history tracking.
  - `CallbackApprovalGate` — programmatic approval via user-provided async callable.
  - `StdinApprovalGate` — interactive console approval with box-formatted prompts for CLI workflows.
  - `WebhookApprovalGate` — HTTP webhook + polling approval for async and remote approval flows.
  - `format_approval_request()` — box-formatted console output for approval prompts.
- New Pydantic types (all `frozen=True`): `VectorSearchResult`, `VectorStore` Protocol, `MemoryConfig`, `CheckpointConfig`, `CheckpointData`, `CheckpointStore` Protocol, `ApprovalConfig`, `ApprovalRequest`, `ApprovalResponse`, `ApprovalGate` Protocol.
- Optional dependency groups in `pyproject.toml`: `persistence` (`aiosqlite>=0.20`), `redis` (`redis[hiredis]>=5.0`), `vector` (`chromadb>=0.5`).
- **Examples**: `examples/06_pluggable_memory.py` (SQLite, Redis, vector search, composite memory, SharedMemory DI), `examples/07_checkpointing.py` (filesystem/SQLite stores, serialization, spec-change detection, crash/resume), `examples/08_hitl_approval.py` (callback/stdin/webhook gates, config enforcement, timeouts, audit trail).
- **Test suites** for all Phase 2 modules — unit tests (`test_memory.py`, `test_checkpoint.py`, `test_hitl.py`) and integration tests (`test_checkpoint_stores.py`, `test_composite_memory.py`, `test_full_pipeline.py`).

### Changed

- **Orchestrator** — `AnyCode` now saves checkpoints automatically after each execution wave via `CheckpointManager`, supports `resume_from` parameter (accepts `"latest"` or a specific checkpoint ID) for crash recovery, and enforces task-level approval gates via `ApprovalManager` before execution.
- **SharedMemory** — accepts any `MemoryStore` backend via constructor injection; defaults to `InMemoryStore` for full backward compatibility.
- **TeamConfig** — new optional `memory_store` parameter for pluggable team memory backends.
- **Types** — expanded `types.py` with 11 new Pydantic models for memory, checkpoint, and approval subsystems. All models remain frozen (immutable).

## [0.2.0] - 2025-04-05

### Added

- **Telemetry module** — OpenTelemetry-integrated tracing with `Tracer`, `Span`, and `ConsoleExporter` for full lifecycle visibility across agent runs. Includes `MetricsCollector` with `Timer` for latency tracking and `EventEmitter` for structured telemetry events.
- **Guardrails module** — Runtime safety layer with `BudgetTracker` for token/cost budget enforcement, `HookRunner` with `LoggingHook` for turn-level lifecycle hooks, and composable content validators (`MaxLengthValidator`, `ContainsValidator`, `BlocklistValidator`).
- **Structured output module** — Schema-constrained LLM responses via Pydantic models. Includes `schema_to_tool_def` and `schema_to_openai_response_format` for cross-provider schema conversion, `parse_structured_output` for validated extraction, and `build_retry_prompt` for automatic recovery on malformed responses.
- **Production features example** (`examples/05_production_features.py`) demonstrating telemetry, guardrails, and structured output working together in a real workflow.
- **Test suite** — Initial tests for guardrails, structured output, and telemetry modules.
- **Dev scripts** — `scripts/setup.sh`, `scripts/lint.sh`, `scripts/test.sh` for reproducible local development.
- `TraceConfig`, `SpanAttributes`, `GuardrailConfig`, `BudgetStatus`, `ValidationResult`, `OutputValidator`, `TurnHook`, `StructuredOutputConfig`, `StructuredRunResult`, and `StructuredAgentResult` types.
- Optional `telemetry` dependency group for OpenTelemetry packages.

### Changed

- **Runner** — Expanded `AgentRunner` with guardrail integration, structured output support, trace context propagation, and improved turn-level error handling. Significant internal refactor for extensibility.
- **Orchestrator** — `AnyCode` orchestrator now supports telemetry hooks, budget-aware scheduling, and structured task results. Task execution flow refactored for better observability.
- **Agent** — `Agent` class extended with guardrail config, trace context, and structured output options. Agent state management improved.
- **Scheduler** — Enhanced scheduling strategies with budget-aware task prioritization.
- **Providers** — `AnthropicAdapter` and `OpenAIAdapter` updated with streaming improvements, better error propagation, and structured output pass-through.
- **Tools** — `ToolRegistry` and `ToolExecutor` refined for safer execution, improved validation, and better error messages. `bash`, `file_write`, and `grep` tools hardened.
- **Collaboration** — `Team`, `MessageBus`, and `SharedMemory` refined with tighter type contracts and improved concurrency safety.
- **Types** — Expanded `types.py` with all new Pydantic models. All models remain frozen (immutable).

### Fixed

- Pool concurrency edge case in `AgentPool` under high parallelism.
- Task dependency validation now catches circular references earlier.

## [0.1.0] - 2025-03-20

### Added

- Initial release of the AnyCode Python orchestration framework.
- Core agent system with `Agent`, `AgentRunner`, `AgentPool`, and `Scheduler`.
- `AnyCode` high-level orchestrator with `TaskSpec` declarative API.
- Provider-agnostic LLM integration via `LLMAdapter` protocol with Anthropic and OpenAI adapters.
- Team collaboration primitives: `Team`, `MessageBus`, `SharedMemory`, `InMemoryStore`.
- Dependency-aware task scheduling with topological sort.
- Built-in tool system: `bash`, `file_read`, `file_edit`, `file_write`, `grep`.
- `ToolRegistry` with `define_tool` for runtime tool registration.
- `Semaphore`-based concurrency gating.
- Token usage tracking with `merge_usage`.
- Four examples: solo worker, crew workflow, staged pipeline, hybrid tooling.
- Pydantic-based immutable type system (`frozen=True` on all models).

[Unreleased]: https://github.com/Quantlix/anycode/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Quantlix/anycode/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Quantlix/anycode/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Quantlix/anycode/releases/tag/v0.1.0
