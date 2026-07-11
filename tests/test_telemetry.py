"""Tests for the telemetry module: tracer, metrics, and events."""

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import patch

import pytest

from anycode import OTLPExporter
from anycode.telemetry.events import EventEmitter, TelemetryEvent
from anycode.telemetry.metrics import MetricsCollector, Timer
from anycode.telemetry.tracer import Span, Tracer, _NoOpSpan
from anycode.types import SpanAttributes, TraceConfig

# -- Tracer tests --


class TestTracer:
    def test_otlp_exporter_is_public(self) -> None:
        assert OTLPExporter.__name__ == "OTLPExporter"

    def test_otlp_export_preserves_timing_correlation_and_redaction(self) -> None:
        class _ExportedSpan:
            def __init__(self) -> None:
                self.attributes: dict[str, object] = {}
                self.events: list[tuple[str, dict[str, object]]] = []
                self.ended_at: int | None = None

            def set_attribute(self, key: str, value: object) -> None:
                self.attributes[key] = value

            def add_event(self, name: str, attributes: dict[str, object]) -> None:
                self.events.append((name, attributes))

            def set_status(self, status: object, description: str) -> None:
                pass

            def end(self, *, end_time: int) -> None:
                self.ended_at = end_time

        class _ExporterTracer:
            def __init__(self) -> None:
                self.span = _ExportedSpan()
                self.started_at: int | None = None

            def start_span(self, name: str, *, start_time: int) -> _ExportedSpan:
                assert name == "otlp.test"
                self.started_at = start_time
                return self.span

        exporter = OTLPExporter()
        exporter_tracer = _ExporterTracer()
        exporter._tracer = exporter_tracer
        span = Span("otlp.test", trace_id="a" * 32)
        span.set_attribute("run_id", "run-123")
        span.add_event("request", {"authorization": "Bearer abcdefghijklmnop"})
        span.end()
        event = TelemetryEvent(
            "anycode.span.completed",
            {
                "trace_id": span.trace_id,
                "span_id": span.span_id,
                "parent_span_id": None,
                "run_id": "run-123",
            },
        )

        exporter.export_completion(span, event)

        assert exporter_tracer.started_at == span.started_at_ns
        assert exporter_tracer.span.ended_at == span.ended_at_ns
        assert exporter_tracer.span.attributes["anycode.trace_id"] == span.trace_id
        assert exporter_tracer.span.attributes["anycode.span_id"] == span.span_id
        assert exporter_tracer.span.attributes["run_id"] == "run-123"
        assert exporter_tracer.span.events == [("request", {"authorization": "<redacted-secret>"})]

    def test_otlp_exporter_flushes_and_shuts_down_provider(self) -> None:
        class _Provider:
            def __init__(self) -> None:
                self.flush_timeout: int | None = None
                self.shutdown_called = False

            def force_flush(self, *, timeout_millis: int) -> bool:
                self.flush_timeout = timeout_millis
                return True

            def shutdown(self) -> None:
                self.shutdown_called = True

        exporter = OTLPExporter()
        provider = _Provider()
        exporter._provider = provider

        assert exporter.force_flush(timeout_millis=1234) is True
        exporter.shutdown()

        assert provider.flush_timeout == 1234
        assert provider.shutdown_called is True
        assert exporter._provider is None

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

    def test_span_exports_redact_secrets_by_default(self) -> None:
        tracer = Tracer(TraceConfig(enabled=True, exporter="none"))
        span = tracer.start_span("secret.op")
        span.set_attribute("api_key", "plain-value")
        span.set_error("provider rejected sk-1234567890abcdef1234567890")

        exported = span.to_dict()

        assert exported["attributes"]["api_key"] == "<redacted-secret>"
        assert "sk-" not in exported["error"]
        assert span.to_dict(redact_sensitive_data=False)["attributes"]["api_key"] == "plain-value"

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
        assert (
            tracer.metrics.get_counter(
                "anycode.errors",
                {"error_type": "error", "operation": "failing.op"},
            )
            == 1
        )

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

    @pytest.mark.asyncio
    async def test_concurrent_span_contexts_do_not_cross_parent(self) -> None:
        tracer = Tracer(TraceConfig(enabled=True, exporter="none"))
        both_started = asyncio.Event()
        started = 0

        async def _branch(name: str) -> Span:
            nonlocal started
            async with tracer.async_span(f"{name}.parent") as parent:
                started += 1
                if started == 2:
                    both_started.set()
                await both_started.wait()
                async with tracer.async_span(f"{name}.child") as child:
                    assert child.parent is parent
                    return child

        first, second = await asyncio.gather(_branch("first"), _branch("second"))

        assert first.trace_id != second.trace_id

    def test_console_exporter(self, capsys: pytest.CaptureFixture[str]) -> None:
        tracer = Tracer(TraceConfig(enabled=True, exporter="console"))
        with tracer.span("console.test") as span:
            span.set_attribute("key", "val")
        output = capsys.readouterr().out
        assert "console.test" in output

    def test_jsonl_exporter_emits_redacted_correlated_record(self, capsys: pytest.CaptureFixture[str]) -> None:
        tracer = Tracer(TraceConfig(enabled=True, exporter="jsonl"))
        with tracer.span("json.test") as span:
            span.set_attribute("run_id", "run-123")
            span.set_attribute("authorization", "Bearer abcdefghijklmnop")
            with tracer.span("json.child") as child:
                pass

        payloads = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        payload = next(item for item in payloads if item["operation"] == "json.test")
        child_payload = next(item for item in payloads if item["operation"] == "json.child")
        assert payload["type"] == "anycode.span.completed"
        assert payload["trace_id"] == span.trace_id
        assert payload["span_id"] == span.span_id
        assert payload["parent_span_id"] is None
        assert payload["run_id"] == "run-123"
        assert payload["authorization"] == "<redacted-secret>"
        assert payload["started_at"].endswith("+00:00")
        assert payload["ended_at"].endswith("+00:00")
        assert child_payload["trace_id"] == span.trace_id
        assert child_payload["parent_span_id"] == span.span_id
        assert child_payload["span_id"] == child.span_id
        assert child_payload["run_id"] == "run-123"

    def test_exporter_failure_does_not_change_run_behavior(self) -> None:
        tracer = Tracer(TraceConfig(enabled=True, exporter="jsonl"))
        with patch("builtins.print", side_effect=OSError("sink unavailable")):
            with tracer.span("sink.failure"):
                pass

        assert [span.name for span in tracer.spans] == ["sink.failure"]
        assert tracer.metrics.get_histogram("anycode.latency.ms", {"operation": "sink.failure"})

    def test_env_variable_config(self) -> None:
        env = {
            "ANYCODE_TRACE_ENABLED": "true",
            "ANYCODE_TRACE_EXPORTER": "none",
            "ANYCODE_TRACE_SERVICE_NAME": "test-svc",
            "ANYCODE_TRACE_SAMPLE_RATE": "1.0",
            "ANYCODE_TRACE_MAX_RECORDED_SPANS": "1",
        }
        with patch.dict(os.environ, env, clear=False):
            tracer = Tracer()
            assert tracer.enabled is True
            with tracer.span("first"):
                pass
            with tracer.span("second"):
                pass
            assert [span.name for span in tracer.spans] == ["second"]
            assert tracer.dropped_spans == 1

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

    def test_sampled_out_trace_keeps_metrics_but_skips_span_storage(self) -> None:
        tracer = Tracer(TraceConfig(enabled=True, exporter="none", sample_rate=0.0))
        with tracer.span("sampled.out") as parent:
            with tracer.span("sampled.out.child") as child:
                assert parent.sampled is False
                assert child.sampled is False

        assert tracer.spans == []
        assert tracer.metrics.get_histogram("anycode.latency.ms", {"operation": "sampled.out"})
        assert tracer.metrics.get_histogram("anycode.latency.ms", {"operation": "sampled.out.child"})

    def test_separate_roots_with_same_trace_id_share_sampling_decision(self) -> None:
        tracer = Tracer(TraceConfig(enabled=True, exporter="none", sample_rate=0.5))
        roots: list[Span] = []
        for name in ("turn.1", "turn.2", "terminal"):
            with tracer.span(name, trace_id="a" * 32) as span:
                roots.append(span)

        assert len({span.sampled for span in roots}) == 1
        assert len({span.trace_id for span in roots}) == 1

    def test_in_memory_telemetry_is_bounded_and_reports_drops(self) -> None:
        tracer = Tracer(
            TraceConfig(
                enabled=True,
                exporter="none",
                max_recorded_spans=2,
                max_recorded_events=2,
                max_metric_series=2,
                max_histogram_samples=2,
            )
        )
        for index in range(3):
            with tracer.span(f"operation.{index}"):
                pass

        assert [span.name for span in tracer.spans] == ["operation.1", "operation.2"]
        assert tracer.dropped_spans == 1
        assert len(tracer.events.events) == 2
        assert tracer.events.dropped_events == 1
        assert tracer.metrics.dropped_series == 1

        metrics = MetricsCollector(enabled=True, max_histogram_samples=2)
        for value in (1.0, 2.0, 3.0):
            metrics.record("bounded", value)
        assert metrics.get_histogram("bounded") == [2.0, 3.0]
        assert metrics.dropped_histogram_samples == 1
        assert metrics.get_summary()["dropped_histogram_samples"] == 1

    def test_completed_spans_feed_correlated_metrics_and_events(self) -> None:
        tracer = Tracer(TraceConfig(enabled=True, exporter="none"))
        parent = tracer.start_span("anycode.agent.run")
        child = tracer.start_span("anycode.llm.chat", parent=parent)
        child.set_attributes(
            SpanAttributes(
                agent_name="planner",
                model="fake-model",
                token_input=12,
                token_output=4,
            )
        )
        child.set_error("provider timeout")
        tracer.end_span(child)
        tracer.end_span(parent)

        assert child.trace_id == parent.trace_id
        assert child.span_id != parent.span_id
        labels = {"agent": "planner", "model": "fake-model"}
        assert tracer.metrics.get_counter("anycode.tokens.input", labels) == 12
        assert tracer.metrics.get_counter("anycode.tokens.output", labels) == 4
        assert tracer.metrics.get_histogram("anycode.latency.ms", {"operation": "anycode.llm.chat"})
        assert (
            tracer.metrics.get_counter(
                "anycode.errors",
                {"error_type": "error", "operation": "anycode.llm.chat"},
            )
            == 1
        )
        event = tracer.events.events[-2].to_dict()
        assert event["name"] == "anycode.span.completed"
        assert event["attributes"]["trace_id"] == child.trace_id
        assert event["attributes"]["span_id"] == child.span_id
        assert event["attributes"]["parent_span_id"] == parent.span_id

    @pytest.mark.asyncio
    async def test_runner_emits_one_correlated_trace_and_runtime_metrics(self) -> None:
        from pydantic import BaseModel

        from anycode.core.runner import AgentRunner
        from anycode.providers.fake import FakeAdapter, FakeResponse
        from anycode.tools.executor import ToolExecutor
        from anycode.tools.registry import ToolRegistry, define_tool
        from anycode.types import LLMMessage, RunnerOptions, TextBlock, ToolResult, ToolUseContext

        class _EchoInput(BaseModel):
            value: str

        async def _echo(tool_input: _EchoInput, _context: ToolUseContext) -> ToolResult:
            return ToolResult(data=tool_input.value)

        registry = ToolRegistry()
        registry.register(define_tool(name="echo", description="echo", input_model=_EchoInput, execute=_echo))
        tracer = Tracer(TraceConfig(enabled=True, exporter="none"))
        runner = AgentRunner(
            FakeAdapter(
                responses=[
                    FakeResponse(tool_calls=(("echo", {"value": "hello"}),)),
                    FakeResponse(text="done"),
                ]
            ),
            registry,
            ToolExecutor(registry),
            RunnerOptions(model="fake-model", agent_name="telemetry-agent", max_turns=3),
            tracer=tracer,
        )

        result = await runner.run([LLMMessage(role="user", content=[TextBlock(text="go")])])

        assert result.stop_reason is not None and result.stop_reason.code == "success"
        assert len({span.trace_id for span in tracer.spans}) == 1
        run_ids = {span.attributes["run_id"] for span in tracer.spans if "run_id" in span.attributes}
        assert len(run_ids) == 1
        token_labels = {"agent": "telemetry-agent", "model": "fake-model"}
        assert tracer.metrics.get_counter("anycode.tokens.input", token_labels) == 10
        assert tracer.metrics.get_counter("anycode.tokens.output", token_labels) == 10
        assert tracer.metrics.get_counter("anycode.cost.usd", token_labels) > 0
        assert tracer.metrics.get_counter("anycode.runs", {"outcome": "completed", "stop_reason": "success"}) == 1
        assert tracer.metrics.get_histogram("anycode.retries", token_labels) == [0.0]
        llm_event = next(event.to_dict() for event in tracer.events.events if event.attributes["operation"] == "anycode.llm.chat")
        assert llm_event["attributes"]["run_id"] in run_ids

    @pytest.mark.asyncio
    async def test_handoff_emits_correlated_terminal_metrics(self) -> None:
        from anycode.core.runner import AgentRunner
        from anycode.handoff.tool import HANDOFF_TOOL_DEF
        from anycode.providers.fake import FakeAdapter, FakeResponse
        from anycode.tools.executor import ToolExecutor
        from anycode.tools.registry import ToolRegistry
        from anycode.types import LLMMessage, RunnerOptions, TextBlock

        registry = ToolRegistry()
        registry.register(HANDOFF_TOOL_DEF)
        tracer = Tracer(TraceConfig(enabled=True, exporter="none"))
        runner = AgentRunner(
            FakeAdapter(
                responses=[
                    FakeResponse(
                        tool_calls=(
                            (
                                "handoff",
                                {"to_agent": "reviewer", "summary": "Ready for review", "reason": "Needs review"},
                            ),
                        )
                    )
                ]
            ),
            registry,
            ToolExecutor(registry),
            RunnerOptions(model="fake-model", agent_name="handoff-agent"),
            tracer=tracer,
        )

        result = await runner.run([LLMMessage(role="user", content=[TextBlock(text="handoff")])])

        assert result.handoff_request is not None
        terminal = next(span for span in tracer.spans if span.name.endswith(".terminal"))
        assert terminal.attributes["phase"] == "completed"
        assert terminal.attributes["stop_reason"] == "success"
        assert len({span.trace_id for span in tracer.spans}) == 1
        assert tracer.metrics.get_counter("anycode.runs", {"outcome": "completed", "stop_reason": "success"}) == 1

    @pytest.mark.asyncio
    async def test_tool_error_result_marks_span_and_error_metric(self) -> None:
        from pydantic import BaseModel

        from anycode.core.runner import AgentRunner
        from anycode.providers.fake import FakeAdapter, FakeResponse
        from anycode.tools.executor import ToolExecutor
        from anycode.tools.registry import ToolRegistry, define_tool
        from anycode.types import LLMMessage, RunnerOptions, TextBlock, ToolResult, ToolUseContext

        class _FailInput(BaseModel):
            reason: str

        async def _fail(tool_input: _FailInput, _context: ToolUseContext) -> ToolResult:
            return ToolResult(data=tool_input.reason, is_error=True)

        registry = ToolRegistry()
        registry.register(define_tool(name="fail", description="fail", input_model=_FailInput, execute=_fail))
        tracer = Tracer(TraceConfig(enabled=True, exporter="none"))
        runner = AgentRunner(
            FakeAdapter(
                responses=[
                    FakeResponse(tool_calls=(("fail", {"reason": "expected failure"}),)),
                    FakeResponse(text="recovered"),
                ]
            ),
            registry,
            ToolExecutor(registry),
            RunnerOptions(model="fake-model", agent_name="tool-agent"),
            tracer=tracer,
        )

        result = await runner.run([LLMMessage(role="user", content=[TextBlock(text="run tool")])])

        assert result.stop_reason is not None and result.stop_reason.code == "success"
        tool_span = next(span for span in tracer.spans if span.name == "anycode.tool.fail")
        assert tool_span.status == "error"
        assert (
            tracer.metrics.get_counter(
                "anycode.errors",
                {"error_type": "error", "operation": "anycode.tool.fail"},
            )
            == 1
        )


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

    def test_event_export_redacts_sensitive_attributes(self) -> None:
        event = TelemetryEvent("provider.error", {"authorization": "Bearer abcdefghijklmnop", "input_tokens": 12})

        exported = event.to_dict()

        assert exported["attributes"] == {"authorization": "<redacted-secret>", "input_tokens": 12}

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
