"""Versioned AnyCode-to-OpenTelemetry GenAI mapping and capture policy."""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Literal

from pydantic import Field, JsonValue

from anycode.contracts.models import ContractModel, utc_now
from anycode.identity.context import ExecutionContext
from anycode.security.redaction import redact_sensitive

OTEL_GENAI_CONVENTION_SNAPSHOT = "1.43.0"
TelemetryCaptureProfile = Literal["off", "metadata", "redacted", "full"]

_CONTENT_KEY_PARTS = (
    "artifact_body",
    "completion",
    "content",
    "input_messages",
    "output_messages",
    "prompt",
    "response",
    "system_instructions",
    "tool_argument",
    "tool_input",
    "tool_output",
)
_CREDENTIAL_KEY_PARTS = ("api_key", "authorization", "credential", "password", "private_key", "secret")
_OPERATION_NAMES = {
    "run": "invoke_workflow",
    "task": "invoke_agent",
    "model": "chat",
    "tool": "execute_tool",
    "policy": "policy",
    "verification": "evaluation",
    "artifact": "artifact",
    "a2a": "a2a",
}


class GenAITelemetryConfig(ContractModel):
    profile: TelemetryCaptureProfile = "redacted"
    convention_snapshot: str = OTEL_GENAI_CONVENTION_SNAPSHOT
    max_string_length: int = Field(default=4_096, ge=16)
    max_attributes: int = Field(default=128, ge=1)
    hash_fields: tuple[str, ...] = ()
    buffer_capacity: int = Field(default=1_000, ge=1)


class GenAITelemetryRecord(ContractModel):
    name: str = Field(min_length=1)
    attributes: dict[str, JsonValue] = Field(default_factory=dict)
    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    observed_at: datetime = Field(default_factory=utc_now)


def _normalized_key(key: str) -> str:
    return key.casefold().replace("-", "_").replace(".", "_")


def _matches(key: str, parts: tuple[str, ...]) -> bool:
    normalized = _normalized_key(key)
    return any(part in normalized for part in parts)


def _is_credential_key(key: str) -> bool:
    normalized = _normalized_key(key)
    if _matches(normalized, _CREDENTIAL_KEY_PARTS):
        return True
    return normalized == "token" or normalized.endswith("_token") or normalized.startswith("token_")


def _sanitize_value(value: JsonValue, config: GenAITelemetryConfig, *, key: str = "") -> JsonValue:
    if isinstance(value, dict):
        items: list[tuple[str, JsonValue]] = []
        for child_key in sorted(value):
            if _is_credential_key(child_key):
                continue
            if config.profile == "metadata" and _matches(child_key, _CONTENT_KEY_PARTS):
                continue
            sanitized = _sanitize_value(value[child_key], config, key=child_key)
            if child_key in config.hash_fields:
                digest = hashlib.sha256(str(sanitized).encode()).hexdigest()
                sanitized = f"sha256:{digest}"
            items.append((child_key, sanitized))
            if len(items) >= config.max_attributes:
                break
        return dict(items)
    if isinstance(value, list):
        return [_sanitize_value(item, config, key=key) for item in value[: config.max_attributes]]
    if isinstance(value, str) and len(value) > config.max_string_length:
        return value[: config.max_string_length]
    return value


def sanitize_telemetry_attributes(attributes: dict[str, JsonValue], config: GenAITelemetryConfig) -> dict[str, JsonValue]:
    """Apply capture, credential, truncation, hashing, and cardinality policy."""
    if config.profile == "off":
        return {}
    source = redact_sensitive(attributes) if config.profile in ("redacted", "full") else attributes
    sanitized = _sanitize_value(source, config)
    return sanitized if isinstance(sanitized, dict) else {}


class GenAITelemetryMapper:
    """Maps runtime events to one pinned OpenTelemetry GenAI convention snapshot."""

    def __init__(self, config: GenAITelemetryConfig | None = None) -> None:
        self.config = config or GenAITelemetryConfig()

    def map(
        self,
        event_type: str,
        payload: dict[str, JsonValue],
        *,
        context: ExecutionContext | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> GenAITelemetryRecord | None:
        if self.config.profile == "off":
            return None
        category = event_type.split(".", maxsplit=1)[0]
        operation = _OPERATION_NAMES.get(category, category)
        attributes: dict[str, JsonValue] = {
            "anycode.event.type": event_type,
            "anycode.telemetry.schema_version": "1.0",
            "otel.semconv.version": self.config.convention_snapshot,
            "gen_ai.operation.name": operation,
            **payload,
        }
        if context is not None:
            attributes.update({f"anycode.identity.{key}": value for key, value in context.audit_attributes().items()})
        provider = payload.get("provider")
        request_model = payload.get("request_model", payload.get("model"))
        response_model = payload.get("response_model")
        if isinstance(provider, str):
            attributes["gen_ai.provider.name"] = provider
        if isinstance(request_model, str):
            attributes["gen_ai.request.model"] = request_model
        if isinstance(response_model, str):
            attributes["gen_ai.response.model"] = response_model
        if isinstance(payload.get("input_tokens"), int):
            attributes["gen_ai.usage.input_tokens"] = payload["input_tokens"]
        if isinstance(payload.get("output_tokens"), int):
            attributes["gen_ai.usage.output_tokens"] = payload["output_tokens"]
        return GenAITelemetryRecord(
            name=f"{operation} {payload.get('model', category)}",
            attributes=sanitize_telemetry_attributes(attributes, self.config),
            trace_id=trace_id or (context.trace_id if context else None),
            span_id=span_id,
            parent_span_id=parent_span_id,
        )


TelemetryBatchExporter = Callable[[tuple[GenAITelemetryRecord, ...]], Awaitable[None]]


class BoundedTelemetryBuffer:
    """Failure-isolated bounded queue for async telemetry exporters."""

    def __init__(self, capacity: int = 1_000) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._records: deque[GenAITelemetryRecord] = deque(maxlen=capacity)
        self._dropped = 0
        self._export_failures = 0

    @property
    def records(self) -> tuple[GenAITelemetryRecord, ...]:
        return tuple(self._records)

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def export_failures(self) -> int:
        return self._export_failures

    def append(self, record: GenAITelemetryRecord | None) -> None:
        if record is None:
            return
        if len(self._records) == self._records.maxlen:
            self._dropped += 1
        self._records.append(record)

    async def flush(self, exporter: TelemetryBatchExporter, *, max_batch_size: int | None = None) -> bool:
        size = len(self._records) if max_batch_size is None else min(len(self._records), max_batch_size)
        if size == 0:
            return True
        batch = tuple(list(self._records)[:size])
        try:
            await exporter(batch)
        except Exception:
            self._export_failures += 1
            return False
        for _ in range(size):
            self._records.popleft()
        return True
