"""Tests for the telemetry module: tracer, metrics, and events."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from anycode.telemetry.events import EventEmitter, TelemetryEvent
from anycode.telemetry.metrics import MetricsCollector, Timer
from anycode.telemetry.tracer import Span, Tracer, _NoOpSpan
from anycode.types import SpanAttributes, TraceConfig

# -- Tracer tests --


class TestTracer:
    def test_disabled_tracer_returns_noop_span(self) -> None:
        tracer = Tracer(TraceConfig(enabled=False))
        span = tracer.start_span("test")
        assert isinstance(span, _NoOpSpan)

    def test_enabled_tracer_creates_real_span(self) -> None:
        tracer = Tracer(TraceConfig(enabled=True, exporter="none"))
        span = tracer.start_span("test.operation")
        assert isinstance(span, Span)
        assert not isinstance(span, _NoOpSpan)
        assert span.name == "test.operation"
        tracer.end_span(span)

    def test_span_parent_child_relationship(self) -> None:
        tracer = Tracer(TraceConfig(enabled=True, exporter="none"))
        parent = tracer.start_span("parent")
        child = tracer.start_span("child", parent=parent)
        assert child.parent is parent
        tracer.end_span(child)
        tracer.end_span(parent)

    def test_span_auto_parent_via_current_span(self) -> None:
        tracer = Tracer(TraceConfig(enabled=True, exporter="none"))
        parent = tracer.start_span("parent")
        # _current_span should be parent, so child auto-parents
        child = tracer.start_span("child")
        assert child.parent is parent
        tracer.end_span(child)
        tracer.end_span(parent)

    def test_span_attributes(self) -> None:
        tracer = Tracer(TraceConfig(enabled=True, exporter="none"))
        span = tracer.start_span("test")
        span.set_attributes(SpanAttributes(agent_name="planner", model="claude-sonnet-4-6", token_input=100, token_output=50))
        assert span.attributes["agent_name"] == "planner"
        assert span.attributes["model"] == "claude-sonnet-4-6"
        assert span.attributes["token_input"] == 100
        assert span.attributes["token_output"] == 50
        tracer.end_span(span)

    def test_span_single_attribute(self) -> None:
        tracer = Tracer(TraceConfig(enabled=True, exporter="none"))
        span = tracer.start_span("test")
        span.set_attribute("custom_key", "custom_value")
        assert span.attributes["custom_key"] == "custom_value"
        tracer.end_span(span)

    def test_span_events(self) -> None:
        tracer = Tracer(TraceConfig(enabled=True, exporter="none"))
        span = tracer.start_span("test")
        span.add_event("state_change", {"from": "idle", "to": "running"})
        assert len(span.events) == 1
        assert span.events[0]["name"] == "state_change"
        tracer.end_span(span)

    def test_span_error(self) -> None:
        tracer = Tracer(TraceConfig(enabled=True, exporter="none"))
        span = tracer.start_span("test")
        span.set_error("Something failed")
        assert span.status == "error"
        assert span.error == "Something failed"
        tracer.end_span(span)

    def test_span_duration(self) -> None:
        tracer = Tracer(TraceConfig(enabled=True, exporter="none"))
        span = tracer.start_span("test")
        # Duration should be > 0 even without ending
        assert span.duration_ms >= 0
        span.end()
        assert span.duration_ms >= 0
        tracer.end_span(span)

    def test_span_to_dict(self) -> None:
        tracer = Tracer(TraceConfig(enabled=True, exporter="none"))
        span = tracer.start_span("test.op")
        span.set_attribute("key", "value")
        span.add_event("ev1")
        tracer.end_span(span)
        d = span.to_dict()
        assert d["name"] == "test.op"
        assert d["attributes"]["key"] == "value"
        assert len(d["events"]) == 1

    def test_noop_span_has_zero_overhead(self) -> None:
        noop = _NoOpSpan()
        noop.set_attributes(SpanAttributes(agent_name="x"))
        noop.set_attribute("k", "v")
        noop.add_event("e")
        noop.set_error("err")
        noop.end()
        assert noop.attributes == {}
        assert noop.events == []
        assert noop.status == "ok"

    def test_context_manager_span(self) -> None:
        tracer = Tracer(TraceConfig(enabled=True, exporter="none"))
        with tracer.span("ctx.op") as span:
            span.set_attribute("inside", True)
        assert len(tracer.spans) == 1
        assert tracer.spans[0].attributes.get("inside") is True

    def test_context_manager_records_error_on_exception(self) -> None:
        tracer = Tracer(TraceConfig(enabled=True, exporter="none"))
        with pytest.raises(ValueError, match="boom"):
            with tracer.span("failing.op") as _span:
                raise ValueError("boom")
        assert len(tracer.spans) == 1
        assert tracer.spans[0].status == "error"
        assert tracer.spans[0].error == "boom"

    @pytest.mark.asyncio
    async def test_async_span(self) -> None:
        tracer = Tracer(TraceConfig(enabled=True, exporter="none"))
        async with tracer.async_span("async.op") as span:
            span.set_attribute("async", True)
        assert len(tracer.spans) == 1

    @pytest.mark.asyncio
    async def test_async_span_error(self) -> None:
        tracer = Tracer(TraceConfig(enabled=True, exporter="none"))
        with pytest.raises(RuntimeError, match="async fail"):
            async with tracer.async_span("async.fail") as _span:
                raise RuntimeError("async fail")
        assert tracer.spans[0].status == "error"

    def test_console_exporter(self, capsys: pytest.CaptureFixture[str]) -> None:
        tracer = Tracer(TraceConfig(enabled=True, exporter="console"))
        with tracer.span("console.test") as span:
            span.set_attribute("key", "val")
        output = capsys.readouterr().out
        assert "console.test" in output

    def test_env_variable_config(self) -> None:
        env = {
            "ANYCODE_TRACE_ENABLED": "true",
            "ANYCODE_TRACE_EXPORTER": "none",
            "ANYCODE_TRACE_SERVICE_NAME": "test-svc",
            "ANYCODE_TRACE_SAMPLE_RATE": "0.5",
        }
        with patch.dict(os.environ, env, clear=False):
            tracer = Tracer()
            assert tracer.enabled is True

    def test_env_variable_disabled(self) -> None:
        with patch.dict(os.environ, {"ANYCODE_TRACE_ENABLED": "false"}, clear=False):
            tracer = Tracer()
            assert tracer.enabled is False

    def test_spans_collected(self) -> None:
        tracer = Tracer(TraceConfig(enabled=True, exporter="none"))
        tracer.start_span("a")
        span_a = tracer.start_span("a")
        tracer.end_span(span_a)
        span_b = tracer.start_span("b")
        tracer.end_span(span_b)
        assert len(tracer.spans) == 2


# -- Metrics tests --


class TestMetrics:
    def test_disabled_metrics_no_recording(self) -> None:
        collector = MetricsCollector(enabled=False)
        collector.increment("test.counter", 5)
        assert collector.get_counter("test.counter") == 0.0

    def test_increment_counter(self) -> None:
        collector = MetricsCollector(enabled=True)
        collector.increment("ops", 1)
        collector.increment("ops", 2)
        assert collector.get_counter("ops") == 3.0

    def test_counter_with_labels(self) -> None:
        collector = MetricsCollector(enabled=True)
        collector.increment("tokens", 100, {"agent": "planner"})
        collector.increment("tokens", 50, {"agent": "builder"})
        assert collector.get_counter("tokens", {"agent": "planner"}) == 100.0
        assert collector.get_counter("tokens", {"agent": "builder"}) == 50.0

    def test_record_histogram(self) -> None:
        collector = MetricsCollector(enabled=True)
        collector.record("latency", 10.0)
        collector.record("latency", 20.0)
        collector.record("latency", 30.0)
        values = collector.get_histogram("latency")
        assert values == [10.0, 20.0, 30.0]

    def test_record_token_usage(self) -> None:
        collector = MetricsCollector(enabled=True)
        collector.record_token_usage("agent1", "claude-sonnet-4-6", 100, 50)
        assert collector.get_counter("anycode.tokens.input", {"agent": "agent1", "model": "claude-sonnet-4-6"}) == 100.0
        assert collector.get_counter("anycode.tokens.output", {"agent": "agent1", "model": "claude-sonnet-4-6"}) == 50.0
        assert collector.get_counter("anycode.tokens.total", {"agent": "agent1", "model": "claude-sonnet-4-6"}) == 150.0

    def test_record_cost(self) -> None:
        collector = MetricsCollector(enabled=True)
        collector.record_cost("agent1", "claude-sonnet-4-6", 0.005)
        assert collector.get_counter("anycode.cost.usd", {"agent": "agent1", "model": "claude-sonnet-4-6"}) == 0.005

    def test_record_latency(self) -> None:
        collector = MetricsCollector(enabled=True)
        collector.record_latency("llm.chat", 150.0, {"model": "gpt-4o"})
        values = collector.get_histogram("anycode.latency.ms", {"model": "gpt-4o", "operation": "llm.chat"})
        assert values == [150.0]

    def test_record_error(self) -> None:
        collector = MetricsCollector(enabled=True)
        collector.record_error("tool.bash", "timeout")
        assert collector.get_counter("anycode.errors", {"operation": "tool.bash", "error_type": "timeout"}) == 1.0

    def test_get_summary(self) -> None:
        collector = MetricsCollector(enabled=True)
        collector.increment("c1", 10)
        collector.record("h1", 1.0)
        summary = collector.get_summary()
        assert "c1" in summary["counters"]
        assert "h1" in summary["histograms"]

    def test_reset(self) -> None:
        collector = MetricsCollector(enabled=True)
        collector.increment("c1", 10)
        collector.record("h1", 1.0)
        collector.reset()
        assert collector.get_counter("c1") == 0.0
        assert collector.get_histogram("h1") == []

    def test_timer_records_latency(self) -> None:
        collector = MetricsCollector(enabled=True)
        with Timer(collector, "test.op"):
            _ = sum(range(100))
        values = collector.get_histogram("anycode.latency.ms", {"operation": "test.op"})
        assert len(values) == 1
        assert values[0] >= 0


# -- Events tests --


class TestEvents:
    def test_disabled_emitter(self) -> None:
        emitter = EventEmitter(enabled=False)
        emitter.emit("test")
        assert len(emitter.events) == 0

    def test_emit_custom_event(self) -> None:
        emitter = EventEmitter(enabled=True)
        emitter.emit("custom.event", {"key": "value"})
        assert len(emitter.events) == 1
        assert emitter.events[0].name == "custom.event"
        assert emitter.events[0].attributes["key"] == "value"

    def test_agent_lifecycle_events(self) -> None:
        emitter = EventEmitter(enabled=True)
        emitter.agent_start("planner", "claude-sonnet-4-6")
        emitter.agent_complete("planner", turns=3, tokens_used=500)
        emitter.agent_error("planner", "timeout")
        assert len(emitter.events) == 3
        assert emitter.events[0].name == "agent.start"
        assert emitter.events[1].name == "agent.complete"
        assert emitter.events[2].name == "agent.error"

    def test_turn_events(self) -> None:
        emitter = EventEmitter(enabled=True)
        emitter.turn_start("agent1", 1)
        emitter.turn_complete("agent1", 1, token_input=100, token_output=50)
        assert len(emitter.events) == 2
        assert emitter.events[0].attributes["turn_number"] == 1

    def test_tool_events(self) -> None:
        emitter = EventEmitter(enabled=True)
        emitter.tool_start("agent1", "bash")
        emitter.tool_complete("agent1", "bash", duration_ms=150.0, is_error=False)
        assert len(emitter.events) == 2
        assert emitter.events[1].attributes["duration_ms"] == 150.0

    def test_llm_call_events(self) -> None:
        emitter = EventEmitter(enabled=True)
        emitter.llm_call_start("agent1", "claude-sonnet-4-6")
        emitter.llm_call_complete("agent1", "claude-sonnet-4-6", input_tokens=100, output_tokens=50, duration_ms=200.0)
        assert len(emitter.events) == 2

    def test_budget_events(self) -> None:
        emitter = EventEmitter(enabled=True)
        emitter.budget_warning("agent1", "tokens", used=45000, limit=50000)
        emitter.budget_exhausted("agent1", "tokens")
        assert len(emitter.events) == 2
        assert emitter.events[0].name == "budget.warning"
        assert emitter.events[1].name == "budget.exhausted"

    def test_clear(self) -> None:
        emitter = EventEmitter(enabled=True)
        emitter.emit("e1")
        emitter.emit("e2")
        emitter.clear()
        assert len(emitter.events) == 0

    def test_event_to_dict(self) -> None:
        event = TelemetryEvent("test.event", {"k": "v"})
        d = event.to_dict()
        assert d["name"] == "test.event"
        assert d["attributes"]["k"] == "v"
        assert "timestamp" in d
