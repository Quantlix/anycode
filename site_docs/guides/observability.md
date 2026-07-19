---
title: "Trace and Measure AnyCode Runs with Telemetry"
description: "Trace AnyCode runs with correlated spans, bounded metrics and events, redacted JSONL logs, or failure-isolated OTLP export for production observability."
keywords: anycode telemetry, tracing, TraceConfig, structured JSONL logs, opentelemetry otlp, runtime metrics, correlation IDs, observability, alerts
---

# Observe Runs with Telemetry

An enabled AnyCode `Tracer` automatically records correlated spans and derives operational metrics and completion events from them. Use the console exporter for local work, JSONL for container log collectors, or OTLP for a tracing backend. Telemetry is disabled by default.

## Enable telemetry

Pass an enabled `Tracer` to an `Agent`, or configure the default tracer with environment variables.

=== "In code"

    ```python title="tracing.py"
    from anycode import Agent, Tracer
    from anycode.types import TraceConfig

    tracer = Tracer(TraceConfig(enabled=True, exporter="jsonl"))
    agent = Agent(config, tool_registry, tool_executor, tracer=tracer)
    ```

=== "Via environment"

    ```bash
    export ANYCODE_TRACE_ENABLED=true
    export ANYCODE_TRACE_EXPORTER=jsonl
    export ANYCODE_TRACE_SAMPLE_RATE=1.0
    ```

| `TraceConfig` field | Default | Effect |
| --- | --- | --- |
| `enabled` | `False` | Enables spans, metrics, and completion events |
| `service_name` | `"anycode"` | Service name used by OTLP |
| `exporter` | `"console"` | `console`, `jsonl`, `otlp`, or `none` |
| `endpoint` | `None` | OTLP collector endpoint |
| `sample_rate` | `1.0` | Probability of retaining and exporting each root trace |
| `redact_sensitive_data` | `True` | Scrubs recognized credentials from built-in exports |
| `max_recorded_spans` | `10_000` | Retained sampled spans per tracer |
| `max_recorded_events` | `10_000` | Retained completion events per tracer |
| `max_metric_series` | `1_000` | Unique in-memory counter and histogram series |
| `max_histogram_samples` | `1_000` | Recent values retained per histogram series |

The corresponding retention environment variables are `ANYCODE_TRACE_MAX_RECORDED_SPANS`, `ANYCODE_TRACE_MAX_RECORDED_EVENTS`, `ANYCODE_TRACE_MAX_METRIC_SERIES`, and `ANYCODE_TRACE_MAX_HISTOGRAM_SAMPLES`. `ANYCODE_TRACE_SERVICE_NAME`, `ANYCODE_TRACE_ENDPOINT`, and `ANYCODE_TRACE_REDACT_SENSITIVE_DATA` configure the remaining exporter options.

## Correlate a run

Every instrumented agent run has a stable `run_id`, which is persisted when durability is enabled. Its spans share an AnyCode `trace_id`; each span also has a `span_id` and `parent_span_id`. Completion events inherit `run_id`, agent, model, provider, task, and tool dimensions from their parent span, so child LLM and tool records remain searchable without repeating attributes at every call site.

```python
result = await agent.run("Investigate the failed deployment")
run_id = result.lifecycle_events[0].run_id

for event in tracer.events.events:
    record = event.to_dict()
    if record["attributes"].get("run_id") == run_id:
        print(record)
```

Sampling is decided once per root and inherited by its children. A sampled-out trace is omitted from `tracer.spans` and from console, JSONL, and OTLP export. Its operational metrics and in-memory completion events are still recorded. Use `sample_rate=1.0` when an external dashboard derives exact run counts only from exported spans.

## Emit structured JSONL logs

The `jsonl` exporter writes one redacted JSON object for every sampled completed span. Records include wall-clock timestamps, duration, status, `trace_id`, `span_id`, `parent_span_id`, `run_id`, and effective runtime dimensions.

```json
{"duration_ms":412.7,"operation":"anycode.llm.chat","parent_span_id":"af82c1e3b47048aa","run_id":"7ed62e65-86be-44ba-b825-96fdfab73231","span_id":"626d9eac8ea843f4","status":"ok","timestamp":"2026-07-11T04:15:22.481923+00:00","trace_id":"7ed62e6586be44bab82596fdfab73231","type":"anycode.span.completed"}
```

Send stdout to your platform's normal collector, such as Fluent Bit, Vector, CloudWatch, or a Kubernetes logging agent. Export failures are logged as warnings and do not change agent results or cancellation behavior.

Credential-like values are redacted when a span or event is serialized or sent through a built-in exporter. The live in-memory object retains its original values. Set `redact_sensitive_data=False` only when the sink has an independently enforced data-protection boundary.

!!! warning "Redaction is defense in depth"
    Pattern and key-based redaction reduces accidental credential leakage but cannot identify every sensitive business value or personal identifier. Keep secrets out of telemetry attributes and error messages, then enforce access controls and retention at the collector.

## Use automatic metrics

Completed spans feed `tracer.metrics` even when trace sampling excludes them. The collector exposes counters and bounded histogram samples.

| Metric | Type | Dimensions | Meaning |
| --- | --- | --- | --- |
| `anycode.runs` | Counter | `outcome`, `stop_reason` | Terminal run outcomes |
| `anycode.latency.ms` | Histogram | `operation` | LLM, tool, turn, and terminal latency |
| `anycode.llm.first_token.ms` | Histogram | `model` | Streaming time to first token when available |
| `anycode.tokens.input` | Counter | `agent`, `model` | Provider-reported input tokens |
| `anycode.tokens.output` | Counter | `agent`, `model` | Provider-reported output tokens |
| `anycode.tokens.total` | Counter | `agent`, `model` | Combined token usage |
| `anycode.cost.usd` | Counter | `agent`, `model` | Estimated model cost |
| `anycode.retries` | Histogram | `agent`, `model` | Aggregate retries per terminal run |
| `anycode.errors` | Counter | `operation`, `error_type` | Operation and non-success terminal errors |

```python
summary = tracer.metrics.get_summary()
run_count = tracer.metrics.get_counter(
    "anycode.runs",
    {"outcome": "completed", "stop_reason": "success"},
)
```

`MetricsCollector` is in-memory. `get_summary()` reports counter values, retained histogram sizes, and drop counts; use `get_histogram()` when a metrics bridge needs the retained values. You can also derive backend metrics from JSONL or OTLP span attributes. AnyCode does not start an HTTP metrics endpoint.

## Inspect events and retention health

`tracer.events` retains an `anycode.span.completed` event for every completed span, including sampled-out spans. Each event carries the effective correlation and runtime attributes used by metrics and structured logs.

All in-memory telemetry is bounded. Old spans, events, and histogram values roll out when their configured limit is reached; new metric label combinations are rejected after `max_metric_series`.

```python
health = {
    "dropped_spans": tracer.dropped_spans,
    "dropped_events": tracer.events.dropped_events,
    "dropped_metric_series": tracer.metrics.dropped_series,
    "dropped_histogram_samples": tracer.metrics.dropped_histogram_samples,
}
```

The first three counters indicate that retention or label-cardinality limits are undersized for the observation window. Histogram sample drops indicate normal rolling-window eviction, but a fast increase can justify a larger sample limit when percentile stability matters.

You can also construct `MetricsCollector` and `EventEmitter` directly for application-specific instrumentation. Their typed helpers remain available, and their serializers redact sensitive values by default.

## Export spans with OTLP

Install the telemetry extra and configure the collector:

=== "In code"

    ```python
    from anycode import TraceConfig, Tracer

    tracer = Tracer(
        TraceConfig(
            enabled=True,
            exporter="otlp",
            endpoint="http://localhost:4317",
            service_name="deployment-worker",
        )
    )
    ```

=== "Via environment"

    ```bash
    pip install "anycode-py[telemetry]"
    export ANYCODE_TRACE_ENABLED=true
    export ANYCODE_TRACE_EXPORTER=otlp
    export ANYCODE_TRACE_ENDPOINT=http://localhost:4317
    ```

OTLP spans preserve AnyCode's original start and end time. The exporter attaches `anycode.trace_id`, `anycode.span_id`, and `anycode.parent_span_id` as explicit attributes, together with `run_id`. Use those application attributes for cross-record correlation; the backend's native OpenTelemetry IDs are assigned by its SDK and are not the AnyCode IDs.

The OTLP dependency is lazy. If the telemetry extra is absent, the exporter does not ship spans. Verify collector ingestion during deployment readiness checks.

`AnyCode.close()` flushes the configured OTLP provider during normal engine shutdown. When you own a standalone `Tracer`, flush and close it before a short-lived process exits:

```python
tracer.force_flush(timeout_millis=30_000)
tracer.shutdown()
```

## Build dashboards and alerts

A production dashboard should show:

- Run volume and outcome ratio by `stop_reason`.
- P50, P95, and P99 latency for LLM, tool, turn, and terminal operations.
- Streaming first-token latency by model.
- Input, output, and total tokens by agent and model.
- Estimated cost by agent and model.
- Retry distribution and provider/tool error counts.
- Telemetry drop counters and JSONL or OTLP ingestion health.

Start alert thresholds from measured baselines and service objectives. Useful first alerts are:

- Non-success run ratio above 5% for 10 minutes, excluding expected user cancellation.
- P95 LLM latency or first-token latency above twice the seven-day baseline for 10 minutes.
- P95 retries at or above 2, or any sustained `provider_unavailable` stop reason.
- Any `side_effect_unknown` stop reason, which requires operator reconciliation before retry.
- Cost above 80% of the configured budget, with a second alert at exhaustion.
- Growth in dropped spans, events, or metric series, or missing JSONL/OTLP ingestion for five minutes while runs are active.

Tune these examples to traffic volume. A low-volume service should alert on absolute failures as well as percentages.

## Map GenAI operations safely

`GenAITelemetryMapper` maps AnyCode model, tool, policy, verification, artifact, and A2A operations to the pinned OpenTelemetry GenAI semantic-convention snapshot reported by `OTEL_GENAI_CONVENTION_SNAPSHOT`. The mapping is versioned independently so a collector upgrade cannot silently change emitted attribute meaning.

```python
from anycode import GenAITelemetryConfig, GenAITelemetryMapper

mapper = GenAITelemetryMapper(
    GenAITelemetryConfig(profile="metadata", max_string_length=512)
)
record = mapper.map(
    "model.chat",
    {"provider": "openai", "model": "configured-model", "input_tokens": 120},
)
```

Capture profiles are `off`, `metadata`, `redacted`, and `full`. Metadata excludes prompt, response, tool arguments, artifact bodies, and credential-like values. Redacted permits content only after key/value scrubbing; full still applies credential rejection, length limits, hashing rules, and cardinality bounds. Use full only with an independently protected telemetry destination and explicit data governance.

`BoundedTelemetryBuffer` isolates exporter outages from run state and exposes dropped-record counts. Identity fields carried by `ExecutionContext` flow into runtime spans as audit metadata, while credential references and raw credentials are excluded.

See [Map GenAI telemetry safely](genai-telemetry.md) for capture profiles, attribute mapping, hashing, cardinality limits, buffering, and exporter-failure behavior.

## The complete, runnable program

The snippets above are fragments. Here is one self-contained file that enables a tracer, produces the spans an instrumented agent run would emit, and then reads back the correlated events, the derived metrics, and the retention counters. It runs fully offline with no API key: telemetry is synchronous, and the spans stand in for a live run so the output is deterministic.

```python title="telemetry_demo.py"
from anycode import Tracer
from anycode.types import SpanAttributes, TraceConfig


def simulate_instrumented_run(tracer: Tracer, run_id: str) -> None:
    """Open the spans an agent run would emit, driving metrics and events.

    With the JSONL exporter, each completed sampled span prints one redacted
    JSON line as it closes.
    """
    with tracer.span("anycode.run_agent.deployer") as root:
        root.set_attributes(
            SpanAttributes(run_id=run_id, agent_name="deployer", model="claude-haiku-4-5")
        )

        with tracer.span("anycode.llm.chat", parent=root) as llm_span:
            llm_span.set_attributes(
                SpanAttributes(
                    model="claude-haiku-4-5",
                    provider="anthropic",
                    token_input=350,
                    token_output=120,
                    cost_usd=0.0012,
                )
            )

        with tracer.span("anycode.tool.file_read", parent=root) as tool_span:
            tool_span.set_attributes(SpanAttributes(tool_name="file_read"))

        with tracer.span("anycode.run_agent.deployer.terminal", parent=root) as terminal:
            terminal.set_attributes(
                SpanAttributes(phase="completed", stop_reason="success", retry_count=0)
            )


def main() -> None:
    tracer = Tracer(TraceConfig(enabled=True, exporter="jsonl"))
    run_id = "7ed62e65-86be-44ba-b825-96fdfab73231"

    print("--- JSONL span logs ---")
    simulate_instrumented_run(tracer, run_id)

    # Correlate: every completion event that belongs to this run.
    print(f"\n--- Completion events for run {run_id} ---")
    for event in tracer.events.events:
        attributes = event.to_dict()["attributes"]
        if attributes.get("run_id") == run_id:
            print(f"  {attributes['operation']}  status={attributes['status']}")

    # Automatic metrics derived from the completed spans.
    print("\n--- Metric counters ---")
    for key, value in tracer.metrics.get_summary()["counters"].items():
        print(f"  {key}: {value}")
    runs = tracer.metrics.get_counter(
        "anycode.runs", {"outcome": "completed", "stop_reason": "success"}
    )
    print(f"  completed-success runs: {runs}")

    # Retention health — all in-memory telemetry is bounded.
    print("\n--- Retention health ---")
    health = {
        "dropped_spans": tracer.dropped_spans,
        "dropped_events": tracer.events.dropped_events,
        "dropped_metric_series": tracer.metrics.dropped_series,
        "dropped_histogram_samples": tracer.metrics.dropped_histogram_samples,
    }
    print(f"  {health}")

    # Flush and close before a short-lived process exits (required for OTLP).
    tracer.force_flush(timeout_millis=30_000)
    tracer.shutdown()


if __name__ == "__main__":
    main()
```

Run it from the project root:

```bash
uv run python telemetry_demo.py
```

!!! tip "Prefer a tested copy? Use the examples directory"
    The repository ships a runnable, CI-tested version of this pattern. See [`examples/05_production_features.py`](https://github.com/Quantlix/anycode/blob/main/examples/05_production_features.py), whose observability section drives the console exporter, `MetricsCollector`, `Timer`, and `EventEmitter` end to end.

## Next steps

- [Map GenAI telemetry safely](genai-telemetry.md) - configure capture, redaction, buffering, and export.

- [Visualize runs](visualization.md) - render the task graph and per-agent activity.
- [Track and cap cost](cost-tracking.md) - enforce the budgets represented in telemetry.
- [Production controls](production-controls.md) - combine telemetry with durability and safety controls.
- [Configuration](../reference/configuration.md) - review every `TraceConfig` field and default.
