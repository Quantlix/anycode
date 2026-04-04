# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/Quantlix/anycode/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Quantlix/anycode/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Quantlix/anycode/releases/tag/v0.1.0
