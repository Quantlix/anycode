# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Runtime telemetry & cancellation** — `AgentRunner.stream` now catches `asyncio.CancelledError`, emits a terminal `cancelled` lifecycle phase with a `user_cancelled` `StopReason`, and records final `phase`/`stop_reason`/`recoverable` attributes on a dedicated `anycode.agent.{name}.terminal` span before re-raising.
- **Adaptive context lifecycle** — `ContextPolicy` gains `provider_overrides`, `preserved_task_state`, and `preserved_verification_failures`. `ContextManager` accepts a `provider=` kwarg, resolves the matching override via `ContextPolicy.for_provider()`, and emits a `ContextManifest` that includes the resolved provider plus preserved state/failure sections during compaction.
- **Declarative quality gates** — new `anycode.verification.registry` exposes `register_sensor_factory`, `build_sensor`, and `build_sensors`. Built-in factories cover `ruff`, `pyright`, `pytest`, and a pure-Python `regex` sensor. `AgentConfig.verification` and `RunnerOptions.verification` plumb `VerificationSensorConfig` tuples; the runner instantiates a `QualityGate`, evaluates it on the `after_task` phase, and translates `block`/`escalate` outcomes into a `verification_failed` stop reason while `retry` outcomes feed sensor feedback back into the agent loop. The YAML config loader reads top-level and per-agent `verification:` blocks.
- **Deterministic evaluation suite** — new `anycode.providers.fake.FakeAdapter` (and `FakeResponse`) replays a scripted reply sequence with no LLM credentials. `EvalScenario` gains `deterministic`, `fake_responses`, and `fake_tool_failures` fields; `run_scenario` now branches into a deterministic harness when the flag is set. `EvalScenarioResult` and `EvalReport` aggregate `cost_usd`, `retries`, and `verification_failures` so CI can track the new metrics.
- New examples: `examples/22_deterministic_eval.py`, `examples/23_context_pressure.py`, `examples/24_verification_gates.py`, `examples/25_runtime_cancellation.py`.
- New deterministic eval fixture `tests/fixtures/eval/runtime_reliability_deterministic.yaml`.
- New tests in `tests/test_phase6_runtime.py` covering provider overrides, the sensor registry, the deterministic suite, and runner cancellation telemetry.

## [0.5.0] - 2026-05-06

### Added

- **Phase 3.2 — Agent Handoff (orchestrator integration)** — `Team` workflows now route handoff requests through `HandoffExecutor`, validate handoff targets against team membership, support policy-driven handoffs via `OrchestratorConfig.handoff_policy`, and emit `Handoff` records on `TeamRunResult.handoffs`.
- **Phase 3.4 — Intelligent Routing (orchestrator integration)** — `Router.route()` decisions are applied per task before execution; the resolved model/provider override is layered onto the agent config without mutating the original. `TeamRunResult.route_decisions` exposes the decision trail.
- **Phase 4.1 — CLI Toolkit** — new `anycode` CLI (built on `typer` + `rich`):
  - `anycode init <dir>` scaffolds a project (`team.yaml`, `main.py`, `.env.example`, `tools/`, `.gitignore`).
  - `anycode run <config.yaml>` loads a team config and runs it end-to-end.
  - `anycode inspect tools|providers|team <path>|config <path>` introspects the runtime.
  - `anycode version` prints package + Python info.
  - Available via `pip install anycode-py[cli]`.
- **Phase 4.2 — Declarative YAML/TOML config** (`src/anycode/config/`) — `load_config(path)` + `validate_config(path)` parse `.yaml`/`.yml`/`.toml` files into typed `LoadedConfig` with `${ENV_VAR}` substitution. New `AnyCode.from_config(path)` and `engine.run_team_from_config(goal=...)` classmethod/method.
- **Phase 4.3 — Examples cookbook** — five new end-to-end examples (`13_cost_tracking.py`, `14_self_reflection.py`, `15_rag_memory.py`, `16_dag_visualization.py`, `17_yaml_config.py`).
- **Phase 5.1 — Self-Reflection / Critic Loop** (`src/anycode/reflection/`) — `LLMCritic`, `parse_critic_json`, `ReflectionLoop` with `self`/`peer`/`custom` modes. Configured via `OrchestratorConfig.reflection = ReflectionConfig(...)`. Tracks `reflections_count` and `quality_score` on `AgentRunResult`.
- **Phase 5.2 — Cost-Aware Execution Engine** (`src/anycode/cost/`) — `CostTracker`, `build_cost_report`, `DEFAULT_PRICING`, `find_pricing` (with wildcard fallback), `calculate_cost`. Configured via `OrchestratorConfig.cost = CostConfig(budget_usd=..., on_budget_exceeded="stop"|"warn"|"continue")`. Emits cost-alert events at the configured threshold and stops execution when the budget is exhausted (when `on_budget_exceeded="stop"`). `TeamRunResult.cost_report` exposes per-agent and per-model breakdown.
- **Phase 5.3 — DAG Visualization** (`src/anycode/viz/`) — `render_dag(queue, format="mermaid"|"dot"|"json"|"ascii", show_status=True)` and `render_timeline(team_result, width=40)`. Mermaid output includes `classDef` styling per task status.
- **Phase 5.4 — RAG Memory** (`src/anycode/memory/rag.py`, `src/anycode/memory/indexer.py`) — `RAGRetriever` (dedup, namespace filtering, relevance/token caps) and `RAGIndexer` (paragraph-aware chunking, optional tool-result indexing). Configured via `OrchestratorConfig.rag = RAGConfig(...)`. RAG context is auto-injected into every task prompt and outputs are auto-indexed to the configured `VectorStore` (defaults to `InMemoryVectorStore`).

### Changed

- `AgentRunResult` gained `handoff_request`, `reflections_count`, and `quality_score` fields (all optional / defaulted).
- `RunResult` gained `handoff_request` field (optional).
- `TeamRunResult` gained `handoffs`, `route_decisions`, and `cost_report` fields (all optional).
- `OrchestratorConfig` gained `cost`, `reflection`, and `rag` fields (all optional).
- Public exports added to `anycode.__init__`: `CostTracker`, `build_cost_report`, `DEFAULT_PRICING`, `calculate_cost`, `find_pricing`, `render_dag`, `render_timeline`, `LLMCritic`, `ReflectionLoop`, `parse_critic_json`, `RAGRetriever`, `RAGIndexer`, plus the new types `ModelPricing`, `CostConfig`, `CostBreakdown`, `CostReport`, `CriticResult`, `Critic`, `ReflectionConfig`, `RAGConfig`, `RAGContext`, `RAGEntry`.

### Tests

- 43 new tests across `tests/test_cost.py`, `tests/test_viz.py`, `tests/test_config.py`, `tests/test_reflection.py`, `tests/test_rag.py`, `tests/test_cli.py`. Total suite: 343 passing.

## [0.4.0] - 2026-06-10

### Added

- **Additional LLM Providers** — 4 new provider adapters implementing the `LLMAdapter` Protocol.
  - `GeminiAdapter` — Google Gemini via `google-genai` SDK with function calling and streaming support.
  - `OllamaAdapter` — Local Ollama models via HTTP (`httpx`), zero external SDK dependencies, OpenAI-compatible tool format.
  - `BedrockAdapter` — AWS Bedrock for Claude models via `boto3`, Anthropic message format, streaming via response streams.
  - `AzureOpenAIAdapter` — Azure OpenAI via the official `openai` SDK with Azure-specific auth and deployment configuration.
  - `_openai_compat` shared helper module — extracted common OpenAI mapping logic (messages, tools, stop reasons) for reuse across OpenAI, Azure, and Ollama adapters.
  - Extended `create_adapter()` factory to resolve all 6 providers with lazy imports.
- **MCP Integration module** (`src/anycode/mcp/`) — Model Context Protocol support for external tool servers.
  - `MCPClient` — manages connection lifecycle (stdio, SSE, streamable-http transports) via the official `mcp` SDK, with tool discovery and tool execution.
  - `schema_to_pydantic_model()` — dynamic Pydantic model generation from JSON Schema for MCP tool inputs.
  - `mcp_tool_to_definition()` — converts MCP tools into AnyCode `ToolDefinition` with prefixed naming (`mcp_{server}_{tool}`).
  - `discover_and_register()` — batch discovery and registration of MCP tools into the `ToolRegistry`.
  - `validate_server_config()` — transport-aware configuration validation.
  - `ToolRegistry.register_from_mcp()` and `ToolRegistry.deregister_prefix()` for MCP tool lifecycle management.
- **Agent Handoff module** (`src/anycode/handoff/`) — context-preserving agent-to-agent task delegation.
  - `HANDOFF_TOOL_DEF` — built-in sentinel tool that agents call to request a handoff (returns `__HANDOFF__:to:summary:reason`).
  - `HandoffExecutor` — orchestrates context transfer with conversation trimming, system/user prompt generation, and configurable depth limiting.
  - `trim_context()`, `build_handoff_system_prompt()`, `build_handoff_user_message()` — protocol helpers for handoff payloads.
  - Runner integration: `AgentRunner` detects handoff sentinels in tool results and yields `StreamEvent(type="handoff")`.
- **Intelligent Routing module** (`src/anycode/routing/`) — zero-cost heuristic task routing.
  - `classify_task()` — microsecond complexity classification (5 levels: trivial, simple, moderate, complex, expert) based on description length and dependency count.
  - `match_rule()` / `evaluate_rules()` — declarative rule engine supporting complexity conditions, keyword-in checks, and regex patterns with priority ordering.
  - `DefaultRouter` — `Router` Protocol implementation combining classifier + rules engine with default model fallback.
  - Orchestrator integration: routing decisions applied before task wave execution.
- New Pydantic types (all `frozen=True`): `MCPServerConfig`, `MCPToolInfo`, `HandoffRequest`, `Handoff`, `HandoffPolicy` Protocol, `ComplexityLevel`, `RoutingRule`, `RoutingConfig`, `RouteDecision`, `Router` Protocol.
- Extended `AgentConfig.provider` literal to include `"google" | "ollama" | "bedrock" | "azure"`.
- Extended `OrchestratorConfig` with `mcp_servers`, `handoff_policy`, `max_handoff_depth`, `routing` fields.
- Extended `TeamRunResult` with `handoffs` field.
- Optional dependency groups in `pyproject.toml`: `google` (`google-generativeai>=0.8`), `bedrock` (`boto3>=1.34`), `azure` (`openai>=1.50`), `mcp` (`mcp>=1.0`).
- **Examples**: `examples/09_multi_provider.py`, `examples/10_mcp_tools.py`, `examples/11_agent_handoff.py`, `examples/12_intelligent_routing.py`.
- **Test suites**: `tests/test_providers.py` (35 tests), `tests/test_mcp.py` (24 tests), `tests/test_handoff.py` (19 tests), `tests/test_routing.py` (18 tests).

### Changed

- **Orchestrator** — `AnyCode` now manages MCP client lifecycles (connect/disconnect) as an async context manager, registers the handoff tool for agents that opt in, injects per-agent MCP tools into the tool registry, and applies routing decisions before task wave execution.
- **AgentRunner** — detects handoff sentinel results in the tool loop; on detection, yields a `StreamEvent(type="handoff")` and terminates the turn.
- **ToolRegistry** — added `register_from_mcp()` for batch MCP tool registration and `deregister_prefix()` for cleanup on server disconnect.
- **providers/openai.py** — refactored to import shared mapping logic from `_openai_compat.py` (no behavior change).

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

[Unreleased]: https://github.com/Quantlix/anycode/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/Quantlix/anycode/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Quantlix/anycode/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Quantlix/anycode/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Quantlix/anycode/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Quantlix/anycode/releases/tag/v0.1.0
