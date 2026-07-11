"""Span lifecycle management with OpenTelemetry or no-op fallback."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from collections import deque
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from anycode.constants import MS_PER_SECOND
from anycode.security.redaction import redact_sensitive, safe_exception_message
from anycode.telemetry.events import EventEmitter, TelemetryEvent
from anycode.telemetry.metrics import MetricsCollector
from anycode.types import SpanAttributes, TraceConfig

logger = logging.getLogger(__name__)

_INHERITED_ATTRIBUTE_KEYS = ("run_id", "agent_name", "task_id", "model", "provider", "tool_name")


def _resolve_config(config: TraceConfig | None) -> TraceConfig:
    """Resolve TraceConfig from explicit config or environment variables."""
    if config is not None:
        return config
    enabled = os.environ.get("ANYCODE_TRACE_ENABLED", "").lower() in ("true", "1", "yes")
    if not enabled:
        return TraceConfig(enabled=False)
    return TraceConfig(
        enabled=True,
        service_name=os.environ.get("ANYCODE_TRACE_SERVICE_NAME", "anycode"),
        exporter=os.environ.get("ANYCODE_TRACE_EXPORTER", "console"),  # type: ignore[arg-type]
        endpoint=os.environ.get("ANYCODE_TRACE_ENDPOINT"),
        sample_rate=float(os.environ.get("ANYCODE_TRACE_SAMPLE_RATE", "1.0")),
        redact_sensitive_data=os.environ.get("ANYCODE_TRACE_REDACT_SENSITIVE_DATA", "true").lower() in ("true", "1", "yes"),
        max_recorded_spans=int(os.environ.get("ANYCODE_TRACE_MAX_RECORDED_SPANS", "10000")),
        max_recorded_events=int(os.environ.get("ANYCODE_TRACE_MAX_RECORDED_EVENTS", "10000")),
        max_metric_series=int(os.environ.get("ANYCODE_TRACE_MAX_METRIC_SERIES", "1000")),
        max_histogram_samples=int(os.environ.get("ANYCODE_TRACE_MAX_HISTOGRAM_SAMPLES", "1000")),
    )


class Span:
    """Represents a single trace span with timing and attributes."""

    def __init__(self, name: str, parent: Span | None = None, *, sampled: bool = True, trace_id: str | None = None) -> None:
        self.name = name
        self.parent = parent
        self.sampled = parent.sampled if parent else sampled
        self.trace_id = parent.trace_id if parent else (trace_id or uuid.uuid4().hex)
        self.span_id = uuid.uuid4().hex[:16]
        self.attributes: dict[str, Any] = {}
        self.events: list[dict[str, Any]] = []
        self.status: str = "ok"
        self.error: str | None = None
        self._started_at = datetime.now(UTC)
        self._ended_at: datetime | None = None
        self._start_time: float = time.monotonic()
        self._end_time: float | None = None

    def set_attributes(self, attrs: SpanAttributes) -> None:
        for key, value in attrs.model_dump(exclude_none=True).items():
            self.attributes[key] = value

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append({"name": name, "attributes": attributes or {}, "timestamp": time.monotonic()})

    def set_error(self, error: str) -> None:
        self.status = "error"
        self.error = error

    def end(self) -> None:
        self._ended_at = datetime.now(UTC)
        self._end_time = time.monotonic()

    @property
    def duration_ms(self) -> float:
        end = self._end_time or time.monotonic()
        return (end - self._start_time) * MS_PER_SECOND

    @property
    def started_at_ns(self) -> int:
        return int(self._started_at.timestamp() * 1_000_000_000)

    @property
    def ended_at_ns(self) -> int:
        ended_at = self._ended_at or datetime.now(UTC)
        return int(ended_at.timestamp() * 1_000_000_000)

    def to_dict(self, *, redact_sensitive_data: bool = True) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent.span_id if self.parent else None,
            "parent": self.parent.name if self.parent else None,
            "sampled": self.sampled,
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "started_at": self._started_at.isoformat(),
            "ended_at": self._ended_at.isoformat() if self._ended_at else None,
        }
        return redact_sensitive(payload) if redact_sensitive_data else payload


class _NoOpSpan(Span):
    """Zero-overhead span when tracing is disabled."""

    def __init__(self) -> None:
        super().__init__("noop")

    def set_attributes(self, attrs: SpanAttributes) -> None:
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        pass

    def set_error(self, error: str) -> None:
        pass

    def end(self) -> None:
        pass


_NOOP_SPAN = _NoOpSpan()


class SpanExporter:
    """Base exporter interface."""

    def export(self, span: Span) -> None:
        pass

    def export_completion(self, span: Span, event: TelemetryEvent) -> None:
        self.export(span)

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True

    def shutdown(self) -> None:
        pass


class ConsoleExporter(SpanExporter):
    """Prints span data to stdout for local development."""

    def __init__(self, *, redact_sensitive_data: bool = True) -> None:
        self._redact_sensitive_data = redact_sensitive_data

    def export(self, span: Span) -> None:
        data = span.to_dict(redact_sensitive_data=self._redact_sensitive_data)
        indent = "  " if span.parent else ""
        status_icon = "x" if data["status"] == "error" else "v"
        print(f"{indent}[{status_icon}] {data['name']} ({data['duration_ms']:.1f}ms)")  # noqa: T201
        for key, value in data["attributes"].items():
            if value:
                print(f"{indent}    {key}: {value}")  # noqa: T201
        if data["error"]:
            print(f"{indent}    ERROR: {data['error']}")  # noqa: T201
        for event in data["events"]:
            print(f"{indent}    event: {event['name']}")  # noqa: T201


class JSONLExporter(SpanExporter):
    """Writes one redacted JSON object per completed span to stdout."""

    def __init__(self, *, redact_sensitive_data: bool = True) -> None:
        self._redact_sensitive_data = redact_sensitive_data

    def export(self, span: Span) -> None:
        payload = {"type": "anycode.span", **span.to_dict(redact_sensitive_data=self._redact_sensitive_data)}
        print(json.dumps(payload, default=str, separators=(",", ":"), sort_keys=True))  # noqa: T201

    def export_completion(self, span: Span, event: TelemetryEvent) -> None:
        data = event.to_dict(redact_sensitive_data=self._redact_sensitive_data)
        payload = {**data["attributes"], "type": data["name"], "timestamp": data["observed_at"]}
        print(json.dumps(payload, default=str, separators=(",", ":"), sort_keys=True))  # noqa: T201


class OTLPExporter(SpanExporter):
    """Exports spans via OpenTelemetry SDK (lazy-loaded)."""

    def __init__(self, endpoint: str | None = None, service_name: str = "anycode", *, redact_sensitive_data: bool = True) -> None:
        self._endpoint = endpoint
        self._service_name = service_name
        self._redact_sensitive_data = redact_sensitive_data
        self._tracer: Any = None
        self._provider: Any = None

    def _init_tracer(self) -> None:
        if self._tracer is not None:
            return
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter  # type: ignore[import-not-found]
            from opentelemetry.sdk.resources import Resource  # type: ignore[import-not-found]
            from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-not-found]
            from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore[import-not-found]

            resource = Resource.create({"service.name": self._service_name})
            provider = TracerProvider(resource=resource)
            exporter = OTLPSpanExporter(endpoint=self._endpoint) if self._endpoint else OTLPSpanExporter()
            provider.add_span_processor(BatchSpanProcessor(exporter))
            self._provider = provider
            self._tracer = provider.get_tracer(self._service_name)
        except ImportError:
            pass

    def export(self, span: Span) -> None:
        data = span.to_dict(redact_sensitive_data=self._redact_sensitive_data)
        attributes = {
            **data["attributes"],
            "anycode.trace_id": data["trace_id"],
            "anycode.span_id": data["span_id"],
            "anycode.parent_span_id": data["parent_span_id"] or "",
        }
        self._export_span(span, attributes)

    def export_completion(self, span: Span, event: TelemetryEvent) -> None:
        data = event.to_dict(redact_sensitive_data=self._redact_sensitive_data)
        attributes = {
            **data["attributes"],
            "anycode.trace_id": span.trace_id,
            "anycode.span_id": span.span_id,
            "anycode.parent_span_id": span.parent.span_id if span.parent else "",
        }
        self._export_span(span, attributes)

    def _export_span(self, span: Span, attributes: dict[str, Any]) -> None:
        self._init_tracer()
        if self._tracer is None:
            return
        from opentelemetry import trace  # type: ignore[import-not-found]

        events = span.to_dict(redact_sensitive_data=self._redact_sensitive_data)["events"]
        otel_span = self._tracer.start_span(span.name, start_time=span.started_at_ns)
        try:
            for key, value in attributes.items():
                if isinstance(value, (str, int, float, bool)):
                    otel_span.set_attribute(key, value)
            for event in events:
                otel_span.add_event(event["name"], event.get("attributes", {}))
            if span.status == "error":
                otel_span.set_status(trace.StatusCode.ERROR, span.error or "")
        finally:
            otel_span.end(end_time=span.ended_at_ns)

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        if self._provider is None:
            return True
        try:
            return bool(self._provider.force_flush(timeout_millis=timeout_millis))
        except Exception as exc:
            logger.warning("OTLP flush failed: %s", safe_exception_message(exc))
            return False

    def shutdown(self) -> None:
        if self._provider is None:
            return
        provider = self._provider
        self._provider = None
        self._tracer = None
        try:
            provider.shutdown()
        except Exception as exc:
            logger.warning("OTLP shutdown failed: %s", safe_exception_message(exc))


class Tracer:
    """Manages span lifecycle and exports completed spans."""

    def __init__(self, config: TraceConfig | None = None) -> None:
        self._config = _resolve_config(config)
        self._enabled = self._config.enabled
        self._exporter = self._build_exporter()
        self._spans: deque[Span] = deque(maxlen=self._config.max_recorded_spans)
        self._dropped_spans = 0
        self._current_span: ContextVar[Span | None] = ContextVar(f"anycode_current_span_{id(self)}", default=None)
        self._metrics = MetricsCollector(
            enabled=self._enabled,
            max_series=self._config.max_metric_series,
            max_histogram_samples=self._config.max_histogram_samples,
        )
        self._events = EventEmitter(enabled=self._enabled, max_events=self._config.max_recorded_events)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def spans(self) -> list[Span]:
        return list(self._spans)

    @property
    def metrics(self) -> MetricsCollector:
        return self._metrics

    @property
    def events(self) -> EventEmitter:
        return self._events

    @property
    def dropped_spans(self) -> int:
        return self._dropped_spans

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        if self._exporter is None:
            return True
        return self._exporter.force_flush(timeout_millis)

    def shutdown(self) -> None:
        if self._exporter is not None:
            self._exporter.shutdown()

    def _build_exporter(self) -> SpanExporter | None:
        if not self._enabled:
            return None
        if self._config.exporter == "console":
            return ConsoleExporter(redact_sensitive_data=self._config.redact_sensitive_data)
        if self._config.exporter == "jsonl":
            return JSONLExporter(redact_sensitive_data=self._config.redact_sensitive_data)
        if self._config.exporter == "otlp":
            return OTLPExporter(
                endpoint=self._config.endpoint,
                service_name=self._config.service_name,
                redact_sensitive_data=self._config.redact_sensitive_data,
            )
        return None

    def start_span(self, name: str, parent: Span | None = None, *, trace_id: str | None = None) -> Span:
        if not self._enabled:
            return _NOOP_SPAN
        resolved_parent = parent or self._current_span.get()
        resolved_trace_id = resolved_parent.trace_id if resolved_parent else (trace_id or uuid.uuid4().hex)
        sampled = resolved_parent.sampled if resolved_parent else self._should_sample(resolved_trace_id)
        span = Span(name, parent=resolved_parent, sampled=sampled, trace_id=resolved_trace_id)
        self._current_span.set(span)
        return span

    def _should_sample(self, trace_id: str) -> bool:
        if self._config.sample_rate <= 0.0:
            return False
        if self._config.sample_rate >= 1.0:
            return True
        sample = int.from_bytes(hashlib.sha256(trace_id.encode()).digest()[:8], "big") / 2**64
        return sample < self._config.sample_rate

    def end_span(self, span: Span) -> None:
        if not self._enabled or isinstance(span, _NoOpSpan):
            return
        span.end()
        completion_event = self._record_observability(span)
        if span.sampled:
            if len(self._spans) == self._spans.maxlen:
                self._dropped_spans += 1
            self._spans.append(span)
        if span.sampled and self._exporter and completion_event is not None:
            try:
                self._exporter.export_completion(span, completion_event)
            except Exception as exc:
                logger.warning("Telemetry exporter failed for span %s: %s", span.name, safe_exception_message(exc))
        if self._current_span.get() is span:
            self._current_span.set(span.parent)

    def _record_observability(self, span: Span) -> TelemetryEvent | None:
        self._metrics.record_latency(span.name, span.duration_ms)
        attributes = self._effective_attributes(span)
        agent_name = attributes.get("agent_name")
        model = attributes.get("model")
        input_tokens = attributes.get("token_input", 0)
        output_tokens = attributes.get("token_output", 0)
        if isinstance(agent_name, str) and isinstance(model, str) and isinstance(input_tokens, int) and isinstance(output_tokens, int):
            if input_tokens or output_tokens:
                self._metrics.record_token_usage(agent_name, model, input_tokens, output_tokens)
            cost_usd = attributes.get("cost_usd", 0.0)
            if isinstance(cost_usd, (int, float)) and cost_usd:
                self._metrics.record_cost(agent_name, model, float(cost_usd))
        first_token_ms = attributes.get("llm_first_token_ms")
        if isinstance(first_token_ms, (int, float)):
            self._metrics.record("anycode.llm.first_token.ms", float(first_token_ms), {"model": str(model or "unknown")})
        phase = attributes.get("phase")
        stop_reason = attributes.get("stop_reason")
        if isinstance(phase, str) and isinstance(stop_reason, str) and span.name.endswith(".terminal"):
            self._metrics.record_run(phase, stop_reason)
            retry_count = attributes.get("retry_count", 0)
            if isinstance(retry_count, int):
                self._metrics.record_retries(retry_count, {"agent": str(agent_name or "unknown"), "model": str(model or "unknown")})
            if phase != "completed":
                self._metrics.record_error(span.name, stop_reason)
        if span.status == "error":
            self._metrics.record_error(span.name, "error")
        return self._events.emit(
            "anycode.span.completed",
            {
                "trace_id": span.trace_id,
                "span_id": span.span_id,
                "parent_span_id": span.parent.span_id if span.parent else None,
                "run_id": attributes.get("run_id"),
                "operation": span.name,
                "status": span.status,
                "duration_ms": span.duration_ms,
                "started_at": span._started_at.isoformat(),
                "ended_at": span._ended_at.isoformat() if span._ended_at else None,
                **attributes,
                **({"error": span.error} if span.error else {}),
            },
        )

    @staticmethod
    def _effective_attributes(span: Span) -> dict[str, Any]:
        inherited: dict[str, Any] = {}
        ancestors: list[Span] = []
        parent = span.parent
        while parent is not None:
            ancestors.append(parent)
            parent = parent.parent
        for ancestor in reversed(ancestors):
            for key in _INHERITED_ATTRIBUTE_KEYS:
                if key in ancestor.attributes:
                    inherited[key] = ancestor.attributes[key]
        return {**inherited, **span.attributes}

    @contextmanager
    def span(self, name: str, parent: Span | None = None, *, trace_id: str | None = None) -> Generator[Span, None, None]:
        s = self.start_span(name, parent=parent, trace_id=trace_id)
        try:
            yield s
        except Exception as e:
            s.set_error(safe_exception_message(e))
            raise
        finally:
            self.end_span(s)

    @asynccontextmanager
    async def async_span(self, name: str, parent: Span | None = None, *, trace_id: str | None = None) -> AsyncGenerator[Span, None]:
        s = self.start_span(name, parent=parent, trace_id=trace_id)
        try:
            yield s
        except Exception as e:
            s.set_error(safe_exception_message(e))
            raise
        finally:
            self.end_span(s)

    def get_noop_span(self) -> Span:
        return _NOOP_SPAN
