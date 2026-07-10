---
title: "Trace and Measure AnyCode Runs with Telemetry"
description: "Turn on AnyCode tracing with TraceConfig or env vars, export spans to the console or OTLP, and collect metrics and events with MetricsCollector and EventEmitter."
keywords: anycode telemetry, tracing, TraceConfig, Tracer, opentelemetry otlp, MetricsCollector, EventEmitter, observability, spans, ANYCODE_TRACE_ENABLED
---

# Observe Runs with Telemetry

When an agent misbehaves in production you need to see *what it did* — the turns, tool calls, and decisions. AnyCode ships a tracer that records spans through a run, plus a metrics collector and an event emitter you can drive yourself. Everything is off by default and zero-overhead until enabled. This guide shows how to turn each on.

## Enable tracing

The `Tracer` is the one telemetry component wired into the run pipeline automatically. Pass an enabled `Tracer` to an `Agent`, or flip it on with environment variables — no code change needed.

=== "In code"

    ```python title="tracing.py"
    from anycode import Agent, Tracer
    from anycode.types import TraceConfig

    tracer = Tracer(TraceConfig(enabled=True, exporter="console"))
    agent = Agent(config, tool_registry, tool_executor, tracer=tracer)
    ```

=== "Via environment"

    ```bash
    export ANYCODE_TRACE_ENABLED=true
    export ANYCODE_TRACE_EXPORTER=console   # or otlp
    export ANYCODE_TRACE_ENDPOINT=http://localhost:4317
    ```

| `TraceConfig` field | Default | Effect |
| --- | --- | --- |
| `enabled` | `False` | Master switch |
| `service_name` | `"anycode"` | Service name on emitted spans |
| `exporter` | `"console"` | `console`, `otlp`, or `none` |
| `endpoint` | `None` | OTLP collector endpoint |
| `sample_rate` | `1.0` | Fraction of traces to record |

Spans capture per-turn phases, sensor evaluations, and attributes like `stop_reason` and token counts. When tracing is disabled, span calls return a shared no-op object, so instrumentation costs nothing.

!!! warning "OTLP export needs the extra"
    The `otlp` exporter lazily loads the OpenTelemetry SDK and **silently does nothing** if it isn't installed. Install `anycode-py[telemetry]` and point `endpoint` at your collector to actually ship spans.

## Collect metrics

`MetricsCollector` aggregates counters and histograms — tokens, cost, latency, errors. It is a standalone utility: construct it with `enabled=True` and record from your own code or hooks; the framework does not push into it automatically.

```python title="metrics.py"
from anycode import MetricsCollector, Timer

metrics = MetricsCollector(enabled=True)
metrics.record_token_usage("worker", "claude-haiku-4-5", input_tokens=1200, output_tokens=340)

with Timer(metrics, "tool.grep"):
    ...   # timed block records latency on exit

print(metrics.get_summary())
```

## Emit events

`EventEmitter` records a timestamped stream of named events — agent start/complete, turn boundaries, tool calls, budget warnings — with typed helpers for each. Like the metrics collector, you build and drive it yourself.

```python title="events.py"
from anycode import EventEmitter

events = EventEmitter(enabled=True)
events.agent_start("worker", "claude-haiku-4-5")
events.turn_complete("worker", turn_number=1, token_input=1200, token_output=340)

for event in events.events:
    print(event.to_dict())
```

!!! note "Metrics and events are opt-in and manual"
    Only the `Tracer` is auto-instrumented. `MetricsCollector` and `EventEmitter` are yours to wire into hooks or surrounding code — nothing is collected unless you enable the object and call it.

## Next steps

- [Visualize runs](visualization.md) — render the task graph and per-agent activity.
- [Track and cap cost](cost-tracking.md) — the cost numbers you'll feed into metrics.
- [Production controls](production-controls.md) — telemetry as part of hardening a run.
- [Public API](../reference/public-api.md) — `Tracer`, `MetricsCollector`, and `EventEmitter` signatures.
