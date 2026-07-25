"""Tracing, metrics, and event emission.

Exports resolve lazily so the optional OpenTelemetry SDK is imported only when used."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from anycode._lazy import build_export_map, lazy_getattr

_EXPORTS = build_export_map(
    {
        "anycode.telemetry.events": (
            "EventEmitter",
            "TelemetryEvent",
        ),
        "anycode.telemetry.genai": (
            "OTEL_GENAI_CONVENTION_SNAPSHOT",
            "BoundedTelemetryBuffer",
            "GenAITelemetryConfig",
            "GenAITelemetryMapper",
            "GenAITelemetryRecord",
            "TelemetryCaptureProfile",
            "sanitize_telemetry_attributes",
        ),
        "anycode.telemetry.metrics": (
            "MetricsCollector",
            "Timer",
        ),
        "anycode.telemetry.tracer": (
            "ConsoleExporter",
            "JSONLExporter",
            "OTLPExporter",
            "Span",
            "SpanExporter",
            "Tracer",
        ),
    },
)


def __getattr__(name: str) -> Any:
    return lazy_getattr(__name__, name, _EXPORTS, globals())


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    from anycode.telemetry.events import EventEmitter, TelemetryEvent
    from anycode.telemetry.genai import (
        OTEL_GENAI_CONVENTION_SNAPSHOT,
        BoundedTelemetryBuffer,
        GenAITelemetryConfig,
        GenAITelemetryMapper,
        GenAITelemetryRecord,
        TelemetryCaptureProfile,
        sanitize_telemetry_attributes,
    )
    from anycode.telemetry.metrics import MetricsCollector, Timer
    from anycode.telemetry.tracer import ConsoleExporter, JSONLExporter, OTLPExporter, Span, SpanExporter, Tracer

__all__ = [
    "ConsoleExporter",
    "BoundedTelemetryBuffer",
    "EventEmitter",
    "JSONLExporter",
    "GenAITelemetryConfig",
    "GenAITelemetryMapper",
    "GenAITelemetryRecord",
    "MetricsCollector",
    "OTLPExporter",
    "OTEL_GENAI_CONVENTION_SNAPSHOT",
    "Span",
    "SpanExporter",
    "TelemetryEvent",
    "TelemetryCaptureProfile",
    "Timer",
    "Tracer",
    "sanitize_telemetry_attributes",
]
