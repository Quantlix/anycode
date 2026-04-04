"""Span lifecycle management with OpenTelemetry or no-op fallback."""

from __future__ import annotations

import os
import time
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

from anycode.types import SpanAttributes, TraceConfig

_MS_PER_SECOND = 1000


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
    )


class Span:
    """Represents a single trace span with timing and attributes."""

    def __init__(self, name: str, parent: Span | None = None) -> None:
        self.name = name
        self.parent = parent
        self.attributes: dict[str, Any] = {}
        self.events: list[dict[str, Any]] = []
        self.status: str = "ok"
        self.error: str | None = None
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
        self._end_time = time.monotonic()

    @property
    def duration_ms(self) -> float:
        end = self._end_time or time.monotonic()
        return (end - self._start_time) * _MS_PER_SECOND

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parent": self.parent.name if self.parent else None,
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


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


class ConsoleExporter(SpanExporter):
    """Prints span data to stdout for local development."""

    def export(self, span: Span) -> None:
        data = span.to_dict()
        indent = "  " if span.parent else ""
        status_icon = "x" if data["status"] == "error" else "v"
        print(f"{indent}[{status_icon}] {data['name']} ({data['duration_ms']:.1f}ms)")
        for key, value in data["attributes"].items():
            if value:
                print(f"{indent}    {key}: {value}")
        if data["error"]:
            print(f"{indent}    ERROR: {data['error']}")
        for event in data["events"]:
            print(f"{indent}    event: {event['name']}")


class OTLPExporter(SpanExporter):
    """Exports spans via OpenTelemetry SDK (lazy-loaded)."""

    def __init__(self, endpoint: str | None = None, service_name: str = "anycode") -> None:
        self._endpoint = endpoint
        self._service_name = service_name
        self._tracer: Any = None

    def _init_tracer(self) -> None:
        if self._tracer is not None:
            return
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            resource = Resource.create({"service.name": self._service_name})
            provider = TracerProvider(resource=resource)
            exporter = OTLPSpanExporter(endpoint=self._endpoint) if self._endpoint else OTLPSpanExporter()
            provider.add_span_processor(BatchSpanProcessor(exporter))
            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer(self._service_name)
        except ImportError:
            pass

    def export(self, span: Span) -> None:
        self._init_tracer()
        if self._tracer is None:
            return
        from opentelemetry import trace

        with self._tracer.start_as_current_span(span.name) as otel_span:
            for key, value in span.attributes.items():
                if isinstance(value, (str, int, float, bool)):
                    otel_span.set_attribute(key, value)
            for event in span.events:
                otel_span.add_event(event["name"], event.get("attributes", {}))
            if span.status == "error":
                otel_span.set_status(trace.StatusCode.ERROR, span.error or "")


class Tracer:
    """Manages span lifecycle and exports completed spans."""

    def __init__(self, config: TraceConfig | None = None) -> None:
        self._config = _resolve_config(config)
        self._enabled = self._config.enabled
        self._exporter = self._build_exporter()
        self._spans: list[Span] = []
        self._current_span: Span | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def spans(self) -> list[Span]:
        return list(self._spans)

    def _build_exporter(self) -> SpanExporter | None:
        if not self._enabled:
            return None
        if self._config.exporter == "console":
            return ConsoleExporter()
        if self._config.exporter == "otlp":
            return OTLPExporter(endpoint=self._config.endpoint, service_name=self._config.service_name)
        return None

    def start_span(self, name: str, parent: Span | None = None) -> Span:
        if not self._enabled:
            return _NOOP_SPAN
        span = Span(name, parent=parent or self._current_span)
        self._current_span = span
        return span

    def end_span(self, span: Span) -> None:
        if not self._enabled or isinstance(span, _NoOpSpan):
            return
        span.end()
        self._spans.append(span)
        if self._exporter:
            self._exporter.export(span)
        if self._current_span is span:
            self._current_span = span.parent

    @contextmanager
    def span(self, name: str, parent: Span | None = None) -> Generator[Span, None, None]:
        s = self.start_span(name, parent=parent)
        try:
            yield s
        except Exception as e:
            s.set_error(str(e))
            raise
        finally:
            self.end_span(s)

    @asynccontextmanager
    async def async_span(self, name: str, parent: Span | None = None) -> AsyncGenerator[Span, None]:
        s = self.start_span(name, parent=parent)
        try:
            yield s
        except Exception as e:
            s.set_error(str(e))
            raise
        finally:
            self.end_span(s)

    def get_noop_span(self) -> Span:
        return _NOOP_SPAN
