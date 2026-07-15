---
title: "Map AnyCode GenAI Telemetry Safely"
description: Map AnyCode runtime events to pinned OpenTelemetry GenAI attributes with capture profiles, redaction, bounded buffering, and failure-isolated export.
keywords: AnyCode OpenTelemetry, GenAI semantic conventions, AI telemetry redaction, LLM tracing, bounded telemetry buffer
---

# Map GenAI telemetry safely

`GenAITelemetryMapper` converts AnyCode runtime events into records aligned with a pinned OpenTelemetry GenAI semantic-convention snapshot. Capture policy is explicit: turn telemetry off, keep metadata only, redact content, or retain permitted content after centralized secret filtering. `BoundedTelemetryBuffer` keeps exporter failures outside the runtime state path.

## Choose a capture profile

```python
from anycode import GenAITelemetryConfig, GenAITelemetryMapper

config = GenAITelemetryConfig(
    profile="redacted",
    max_string_length=2_048,
    max_attributes=96,
    hash_fields=("user_id",),
    buffer_capacity=500,
)
mapper = GenAITelemetryMapper(config)
```

| Profile | Captured behavior |
| --- | --- |
| `off` | `map()` returns `None` |
| `metadata` | Content-like fields are removed; bounded operational metadata remains |
| `redacted` | Central redaction runs before truncation, hashing, and attribute limits |
| `full` | Permitted content may remain, but credential-like fields are still removed and redaction still runs |

`full` never means raw credentials. Keys matching API keys, authorization, credentials, passwords, private keys, secrets, and token patterns are removed under every enabled profile.

## Map a model event

```python
from anycode import ExecutionContext

context = ExecutionContext(
    principal="service:release-review",
    tenant_scope="tenant:example",
    classification="internal",
    trace_id="4f3c2a1b0e9d8c7b6a5f4e3d2c1b0a99",
)

record = mapper.map(
    "model.completed",
    {
        "provider": "configured-provider",
        "request_model": "review-model",
        "response_model": "review-model-2026-07",
        "input_tokens": 1_250,
        "output_tokens": 320,
        "prompt": "Review the release notes",
        "api_key": "must-never-be-exported",
        "user_id": "user-42",
    },
    context=context,
    span_id="8a7b6c5d4e3f2a10",
)

if record is not None:
    assert "api_key" not in record.attributes
    print(record.name, record.attributes)
```

The mapper adds the event type, AnyCode telemetry schema version, pinned convention version, and operation name. It maps provider, request model, response model, token usage, bounded execution identity audit fields, and trace identifiers when available.

## Know the event operation mapping

| Event prefix | `gen_ai.operation.name` |
| --- | --- |
| `run` | `invoke_workflow` |
| `task` | `invoke_agent` |
| `model` | `chat` |
| `tool` | `execute_tool` |
| `policy` | `policy` |
| `verification` | `evaluation` |
| `artifact` | `artifact` |
| `a2a` | `a2a` |

Unknown prefixes remain visible as their own operation name. This preserves forward compatibility without pretending that an unknown event has standard GenAI semantics.

## Buffer records without blocking runtime state

```python
from anycode import BoundedTelemetryBuffer

buffer = BoundedTelemetryBuffer(capacity=config.buffer_capacity)
buffer.append(record)


async def export_batch(records):
    await otel_exporter.export(records)


exported = await buffer.flush(export_batch, max_batch_size=100)
if not exported:
    logger.warning("telemetry export failed", extra={"failures": buffer.export_failures})
```

The buffer drops the oldest record when full and increments `dropped`. If the exporter raises, `flush()` returns `False`, increments `export_failures`, and retains the batch for a later retry. A successful flush removes only the exported records.

Telemetry must not become a durability dependency. Bound retries and monitor drops and export failures; do not fail admitted work only because an observability service is unavailable.

## Control data volume and cardinality

- Use `metadata` for workloads where prompts and outputs must never leave the execution boundary.
- Use `hash_fields` for stable correlation without exporting the original field value.
- Keep `max_string_length` and `max_attributes` bounded to control payload size.
- Pass only validated JSON values in event payloads.
- Keep credential references out of general payload fields even though credential-like keys are removed.
- Pin dashboards and processors to `config.convention_snapshot` and review changes before updating it.

## Connect to existing tracing

The mapper produces provider-neutral records rather than owning an OTLP pipeline. Send buffered records through the exporter and collector selected by the host. Existing `Tracer`, `OTLPExporter`, metrics, and runtime events remain available for span-level and application-level observability.

Use `ExecutionContext.trace_id` to correlate identity-aware runtime records with an incoming distributed trace. Keep policy-engine decision logs and durability events as their own authoritative records; telemetry is an operational projection, not the system of record.

## Next steps

- [Configure observability](observability.md)
- [Propagate execution identity and policy](execution-identity.md)
- [Host AnyCode services](hosting-services.md)
- [Review the security boundary](../reference/security.md)
