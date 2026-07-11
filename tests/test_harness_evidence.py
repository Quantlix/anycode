"""Tests for the trajectory evidence corpus."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anycode.harness.distill import _scrub_secrets, distill_evidence, replay_raw_trace
from anycode.harness.evidence import EvidenceCollector, EvidenceStore, write_evidence_bundle
from anycode.harness.failure_taxonomy import (
    DEFAULT_TAXONOMY_VERSION,
    categorize_event,
    categorize_run,
    categorize_stop_reason,
)
from anycode.types import (
    AgentRunResult,
    FailureCategory,
    LifecycleEvent,
    LLMMessage,
    StopReason,
    TextBlock,
    TokenUsage,
    ToolCallRecord,
    VerificationResult,
)


def _success_run(text: str = "ok") -> AgentRunResult:
    return AgentRunResult(
        success=True,
        output=text,
        messages=[LLMMessage(role="assistant", content=[TextBlock(text=text)])],
        token_usage=TokenUsage(input_tokens=5, output_tokens=5),
        tool_calls=[],
        stop_reason=StopReason(code="success", message="ok"),
    )


def _failed_run() -> AgentRunResult:
    return AgentRunResult(
        success=False,
        output="something went wrong",
        messages=[LLMMessage(role="assistant", content=[TextBlock(text="oops")])],
        token_usage=TokenUsage(),
        tool_calls=[
            ToolCallRecord(tool_name="bash", input={"cmd": "ls"}, output="error: command failed", duration=0.1),
        ],
        verification_results=[
            VerificationResult(
                sensor_name="ruff",
                kind="computational",
                passed=False,
                severity="error",
                message="lint failure",
            )
        ],
        lifecycle_events=[
            LifecycleEvent(
                run_id="r1",
                agent_name="alice",
                phase="failed",
                stop_reason=StopReason(code="tool_error", message="failed"),
            )
        ],
        stop_reason=StopReason(code="tool_error", message="bash failed"),
    )


# -- failure taxonomy --


def test_taxonomy_version_is_defined() -> None:
    assert DEFAULT_TAXONOMY_VERSION


def test_categorize_stop_reason_maps_known_codes() -> None:
    assert categorize_stop_reason(StopReason(code="max_turns", message="x")) == FailureCategory.EARLY_STOPPING
    assert categorize_stop_reason(StopReason(code="budget_exceeded", message="x")) == FailureCategory.BUDGET_EXCEEDED
    assert categorize_stop_reason(None) == FailureCategory.UNKNOWN


def test_categorize_event_handles_tool_record() -> None:
    failing = ToolCallRecord(tool_name="bash", input={}, output="ValidationError: missing required field", duration=0.0)
    assert categorize_event(failing) == FailureCategory.TOOL_ARGUMENT_ERROR

    runtime = ToolCallRecord(tool_name="bash", input={}, output="exception: oops", duration=0.0)
    assert categorize_event(runtime) == FailureCategory.TOOL_RUNTIME_ERROR

    success = ToolCallRecord(tool_name="bash", input={}, output="hello", duration=0.0)
    assert categorize_event(success) == FailureCategory.SUCCESS


def test_categorize_event_handles_verification() -> None:
    vr = VerificationResult(
        sensor_name="ruff",
        kind="computational",
        passed=False,
        severity="error",
        message="lint",
    )
    assert categorize_event(vr) == FailureCategory.VERIFICATION_FAILURE


def test_categorize_run_prioritizes_stop_reason() -> None:
    assert categorize_run(_failed_run()) == FailureCategory.TOOL_RUNTIME_ERROR
    assert categorize_run(_success_run()) == FailureCategory.SUCCESS
    assert categorize_run(None) == FailureCategory.UNKNOWN


# -- distillation --


def test_distill_evidence_success_yields_summary_packet() -> None:
    ev = distill_evidence(_success_run(), task="say-hi")
    assert ev.run_summary.outcome == "pass"
    assert any(p.category == FailureCategory.SUCCESS for p in ev.evidence_packets)


def test_distill_evidence_failure_emits_packets_for_each_failure() -> None:
    ev = distill_evidence(_failed_run(), run_id="r1", task="failure-case")
    categories = {p.category for p in ev.evidence_packets}
    assert FailureCategory.TOOL_RUNTIME_ERROR in categories
    assert FailureCategory.VERIFICATION_FAILURE in categories
    # failure map should reflect counts
    assert any(entry.count >= 1 for entry in ev.failure_map)
    # decision timeline references events
    assert ev.decision_timeline


def test_distill_evidence_uses_provided_run_id_and_manifest() -> None:
    ev = distill_evidence(
        _success_run(),
        run_id="custom-run",
        task="t",
        manifest_checksum="abc",
    )
    assert ev.run_summary.run_id == "custom-run"
    assert ev.manifest_checksum == "abc"


def test_distill_packets_link_to_event_ids() -> None:
    ev = distill_evidence(_failed_run(), task="t")
    event_ids = {event.id for event in ev.decision_timeline}
    for packet in ev.evidence_packets:
        for eid in packet.event_ids:
            if eid:
                assert eid in event_ids


# -- redaction --


def test_secret_redaction_in_tool_output() -> None:
    result = AgentRunResult(
        success=False,
        output="",
        messages=[],
        token_usage=TokenUsage(),
        tool_calls=[
            ToolCallRecord(
                tool_name="curl",
                input={},
                output="Authorization: Bearer abc123def456ghi789jkl",
                duration=0.0,
            )
        ],
        stop_reason=StopReason(code="tool_error", message="x"),
    )
    ev = distill_evidence(result, task="t")
    for packet in ev.evidence_packets:
        assert "Bearer abc123def456ghi789jkl" not in packet.summary


def test_scrub_secrets_replaces_known_token_patterns() -> None:
    text = "key=sk-1234567890abcdef1234567890 AKIAABCDEFGHIJKLMNOP github=ghp_abcdefghijklmnopqrst"
    cleaned = _scrub_secrets(text)
    assert "sk-1234567890abcdef1234567890" not in cleaned
    assert "AKIAABCDEFGHIJKLMNOP" not in cleaned
    assert "ghp_abcdefghijklmnopqrst" not in cleaned


# -- persistence --


def test_evidence_store_round_trip(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "bundles")
    ev = distill_evidence(_failed_run(), run_id="r-store", task="t")
    target = store.write_bundle(ev, raw_events=[{"name": "tool_start", "tool": "bash"}])
    assert target.exists()
    assert (target / "summary.json").exists()
    assert (target / "packets.json").exists()
    assert (target / "raw_trace.jsonl").exists()
    bundles = store.list_runs()
    assert "r-store" in bundles
    restored = store.load_bundle("r-store")
    assert restored.run_summary.run_id == "r-store"


def test_evidence_store_redacts_structured_sensitive_values(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "bundles")
    ev = distill_evidence(_failed_run(), run_id="r-secret", task="t")

    target = store.write_bundle(ev, raw_events=[{"api_key": "plain-value", "input_tokens": 12}])
    persisted = (target / "raw_trace.jsonl").read_text(encoding="utf-8")

    assert "plain-value" not in persisted
    assert '"input_tokens": 12' in persisted


def test_evidence_store_load_missing_raises(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "empty")
    with pytest.raises(FileNotFoundError):
        store.load_bundle("nope")


def test_evidence_collector_finalizes_to_evidence(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "bundles")
    collector = EvidenceCollector(run_id="c-1", task="task", store=store)
    collector.record({"event": "tool_call", "name": "bash"})
    collector.record({"event": "lifecycle", "phase": "executing"})
    ev = collector.finalize(_success_run())
    assert ev.run_summary.run_id == "c-1"
    assert ev.raw_trace_path is not None
    assert Path(ev.raw_trace_path).exists()
    assert len(collector.raw_events) == 2


def test_write_evidence_bundle_helper_writes_files(tmp_path: Path) -> None:
    ev = distill_evidence(_success_run(), run_id="h-1", task="t")
    target = write_evidence_bundle(ev, tmp_path / "bundles")
    assert (target / "summary.json").exists()


def test_replay_raw_trace_reads_jsonl(tmp_path: Path) -> None:
    target = tmp_path / "trace.jsonl"
    target.write_text(
        json.dumps({"name": "tool_start"}) + "\n" + json.dumps({"name": "tool_complete"}) + "\n",
        encoding="utf-8",
    )
    records = replay_raw_trace(target)
    assert len(records) == 2
    assert records[0]["name"] == "tool_start"


def test_replay_raw_trace_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        replay_raw_trace(tmp_path / "missing.jsonl")


# -- additional taxonomy coverage --


def test_categorize_event_handles_passed_verification() -> None:
    vr = VerificationResult(
        sensor_name="ruff",
        kind="computational",
        passed=True,
        severity="info",
        message="ok",
    )
    assert categorize_event(vr) == FailureCategory.SUCCESS


def test_categorize_event_handles_empty_tool_output() -> None:
    empty = ToolCallRecord(tool_name="bash", input={}, output="", duration=0.0)
    assert categorize_event(empty) == FailureCategory.TOOL_RUNTIME_ERROR


def test_categorize_event_handles_lifecycle_with_stop_reason() -> None:
    event = LifecycleEvent(
        run_id="r",
        agent_name="a",
        phase="failed",
        stop_reason=StopReason(code="budget_exceeded", message="b"),
    )
    assert categorize_event(event) == FailureCategory.BUDGET_EXCEEDED


def test_categorize_event_handles_lifecycle_failed_phase_without_stop() -> None:
    event = LifecycleEvent(run_id="r", agent_name="a", phase="failed")
    assert categorize_event(event) == FailureCategory.UNKNOWN


def test_categorize_event_handles_lifecycle_running_phase() -> None:
    event = LifecycleEvent(run_id="r", agent_name="a", phase="executing")
    assert categorize_event(event) == FailureCategory.SUCCESS


def test_categorize_event_unknown_object() -> None:
    assert categorize_event(object()) == FailureCategory.UNKNOWN


def test_categorize_run_with_only_verification_failure() -> None:
    result = AgentRunResult(
        success=False,
        output="",
        messages=[],
        token_usage=TokenUsage(),
        tool_calls=[],
        verification_results=[
            VerificationResult(
                sensor_name="ruff",
                kind="computational",
                passed=False,
                severity="error",
                message="lint",
            )
        ],
        stop_reason=StopReason(code="success", message="ok"),
    )
    from anycode.harness.failure_taxonomy import categorize_run

    assert categorize_run(result) == FailureCategory.VERIFICATION_FAILURE
