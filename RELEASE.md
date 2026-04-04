# Release v0.2.0

**Production-Ready Foundations** — Telemetry, guardrails, and structured output land in AnyCode.

---

## Highlights

This release transforms AnyCode from a capable orchestration engine into a production-grade framework. Three new modules give you the observability, safety, and data extraction tooling needed to run multi-agent systems with confidence.

### OpenTelemetry Tracing

Every agent run, tool call, and task execution is now traceable. The new `Tracer` integrates with OpenTelemetry so you can pipe spans directly into Jaeger, Datadog, or any OTLP-compatible backend. For local development, `ConsoleExporter` prints structured traces to stdout. `MetricsCollector` tracks latency distributions and `EventEmitter` publishes lifecycle events for custom dashboards.

### Guardrails & Safety

`BudgetTracker` enforces token and cost limits at the run level — agents stop before they overspend. `HookRunner` gives you turn-level lifecycle hooks for logging, auditing, or injecting custom logic between LLM calls. Content validators (`MaxLengthValidator`, `ContainsValidator`, `BlocklistValidator`) compose together to screen agent output before it reaches downstream consumers.

### Structured Output

Extract typed, validated data from LLM responses using Pydantic schemas. `schema_to_tool_def` converts your models into provider-native tool definitions, `parse_structured_output` validates the result, and `build_retry_prompt` automatically recovers when the model returns malformed JSON. Works with both Anthropic and OpenAI.

## What Changed

- **Runner** — Refactored for guardrail integration, structured output, and trace propagation.
- **Orchestrator** — Budget-aware scheduling, telemetry hooks, structured task results.
- **Agent** — Extended with guardrail config and trace context.
- **Providers** — Streaming improvements and structured output pass-through for both adapters.
- **Tools** — Hardened execution with better validation and error messages.
- **Collaboration** — Tighter type contracts and improved concurrency safety.

## Getting Started

```bash
pip install anycode-py
```

For telemetry support:
```bash
pip install anycode-py[telemetry]
```

See `examples/05_production_features.py` for a working demo of all three new modules.

## Full Changelog

See [CHANGELOG.md](CHANGELOG.md) for the complete list of changes.
