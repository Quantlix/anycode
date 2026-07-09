"""Distill a single run into a :class:`TrajectoryEvidence` bundle."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from anycode.harness.failure_taxonomy import (
    DEFAULT_TAXONOMY_VERSION,
    categorize_event,
    categorize_run,
)
from anycode.helpers.uuid7 import uuid7
from anycode.types import (
    AgentRunResult,
    EvidencePacket,
    FailureCategory,
    FailureMapEntry,
    LifecycleEvent,
    RunSummary,
    ToolCallRecord,
    TrajectoryEvent,
    TrajectoryEvidence,
    VerificationResult,
)

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]+", re.IGNORECASE),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
)

_REDACTED = "<redacted-secret>"


def _scrub_secrets(text: str) -> str:
    cleaned = text
    for pattern in _SECRET_PATTERNS:
        cleaned = pattern.sub(_REDACTED, cleaned)
    return cleaned


def _scrub_payload(payload: Any) -> Any:
    if isinstance(payload, str):
        return _scrub_secrets(payload)
    if isinstance(payload, dict):
        return {key: _scrub_payload(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_scrub_payload(item) for item in payload]
    return payload


def _lifecycle_to_event(event: LifecycleEvent, index: int) -> TrajectoryEvent:
    attrs: dict[str, str | int | float | bool] = {
        "phase": event.phase,
        "agent": event.agent_name,
        "run_id": event.run_id,
    }
    if event.stop_reason is not None:
        attrs["stop_reason"] = event.stop_reason.code
    for key, value in event.metadata.items():
        attrs[key] = value
    return TrajectoryEvent(
        id=f"lifecycle-{index:04d}",
        kind="lifecycle",
        name=event.phase,
        timestamp=float(index),
        attributes=attrs,
    )


def _tool_to_event(call: ToolCallRecord, index: int) -> TrajectoryEvent:
    return TrajectoryEvent(
        id=f"tool-{index:04d}",
        kind="tool_call",
        name=call.tool_name,
        timestamp=float(index),
        attributes={
            "tool": call.tool_name,
            "duration": call.duration,
            "output_chars": len(call.output or ""),
        },
    )


def _verification_to_event(result: VerificationResult, index: int) -> TrajectoryEvent:
    return TrajectoryEvent(
        id=f"verify-{index:04d}",
        kind="verification",
        name=result.sensor_name,
        timestamp=float(index),
        attributes={
            "sensor": result.sensor_name,
            "passed": result.passed,
            "severity": result.severity,
            "kind": result.kind,
        },
    )


def _build_timeline(result: AgentRunResult) -> list[TrajectoryEvent]:
    timeline: list[TrajectoryEvent] = []
    for index, lifecycle in enumerate(result.lifecycle_events):
        timeline.append(_lifecycle_to_event(lifecycle, index))
    base = len(timeline)
    for index, call in enumerate(result.tool_calls):
        timeline.append(_tool_to_event(call, base + index))
    base = len(timeline)
    for index, vr in enumerate(result.verification_results):
        timeline.append(_verification_to_event(vr, base + index))
    return timeline


def _suggested_components_for_tool(call: ToolCallRecord) -> tuple[str, ...]:
    return (f"tool:{call.tool_name}",)


def _packet_for_tool(call: ToolCallRecord, event_id: str) -> EvidencePacket | None:
    category = categorize_event(call)
    if category == FailureCategory.SUCCESS:
        return None
    severity = "high" if category == FailureCategory.TOOL_RUNTIME_ERROR else "medium"
    summary = _scrub_secrets((call.output or "").strip())[:240] or "tool produced empty output"
    return EvidencePacket(
        id=f"packet-tool-{uuid7()}",
        category=category,
        summary=summary,
        event_ids=(event_id,),
        severity=severity,  # type: ignore[arg-type]
        suggested_component_ids=_suggested_components_for_tool(call),
        evidence={"tool": call.tool_name, "duration": f"{call.duration:.4f}"},
    )


def _packet_for_verification(result: VerificationResult, event_id: str) -> EvidencePacket | None:
    if result.passed:
        return None
    severity_map = {"info": "low", "warning": "medium", "error": "high", "critical": "critical"}
    severity = severity_map.get(result.severity, "medium")
    return EvidencePacket(
        id=f"packet-verify-{uuid7()}",
        category=FailureCategory.VERIFICATION_FAILURE,
        summary=_scrub_secrets(result.message)[:240] or "verification failed",
        event_ids=(event_id,),
        severity=severity,  # type: ignore[arg-type]
        suggested_component_ids=(f"verification:{result.sensor_name}",),
        evidence={"sensor": result.sensor_name, "kind": result.kind},
    )


def _packet_for_stop_reason(result: AgentRunResult) -> EvidencePacket | None:
    if result.stop_reason is None or result.stop_reason.code == "success":
        return None
    category = categorize_run(result)
    severity = "critical" if not result.stop_reason.recoverable else "high"
    return EvidencePacket(
        id=f"packet-run-{uuid7()}",
        category=category,
        summary=_scrub_secrets(result.stop_reason.message)[:240] or category.value,
        event_ids=(),
        severity=severity,  # type: ignore[arg-type]
        suggested_component_ids=(),
        evidence={"stop_reason": result.stop_reason.code, "recoverable": str(result.stop_reason.recoverable)},
    )


def _build_packets(
    result: AgentRunResult,
    timeline: list[TrajectoryEvent],
) -> list[EvidencePacket]:
    packets: list[EvidencePacket] = []
    tool_events = [e for e in timeline if e.kind == "tool_call"]
    verify_events = [e for e in timeline if e.kind == "verification"]
    for call, event in zip(result.tool_calls, tool_events):
        packet = _packet_for_tool(call, event.id)
        if packet is not None:
            packets.append(packet)
    for vr, event in zip(result.verification_results, verify_events):
        packet = _packet_for_verification(vr, event.id)
        if packet is not None:
            packets.append(packet)
    stop_packet = _packet_for_stop_reason(result)
    if stop_packet is not None:
        packets.append(stop_packet)
    if not packets:
        packets.append(
            EvidencePacket(
                id=f"packet-success-{uuid7()}",
                category=FailureCategory.SUCCESS,
                summary="run completed without failures",
                severity="low",
            )
        )
    return packets


def _build_failure_map(packets: list[EvidencePacket]) -> tuple[FailureMapEntry, ...]:
    counter: Counter[FailureCategory] = Counter()
    representatives: dict[FailureCategory, list[str]] = {}
    for packet in packets:
        counter[packet.category] += 1
        bucket = representatives.setdefault(packet.category, [])
        if packet.event_ids and len(bucket) < 5:
            bucket.append(packet.event_ids[0])
    return tuple(
        FailureMapEntry(
            category=category,
            count=count,
            representative_event_ids=tuple(representatives.get(category, ())),
        )
        for category, count in counter.most_common()
    )


def _build_summary(
    *,
    run_id: str,
    task: str,
    result: AgentRunResult,
) -> RunSummary:
    outcome: str
    if result.success:
        outcome = "pass"
    elif result.stop_reason is not None and result.stop_reason.code != "success":
        outcome = "fail"
    else:
        outcome = "error"
    gate = "unknown"
    if result.gate_decisions:
        last = result.gate_decisions[-1]
        gate = "pass" if last.outcome == "pass" else "warn" if last.outcome == "warn" else "fail"
    runtime_seconds = 0.0
    if result.lifecycle_events:
        runtime_seconds = float(len(result.lifecycle_events))
    return RunSummary(
        run_id=run_id,
        task=task,
        outcome=outcome,  # type: ignore[arg-type]
        stop_reason=result.stop_reason.code if result.stop_reason else None,
        runtime_seconds=runtime_seconds,
        turns=len([m for m in result.messages if m.role == "assistant"]),
        quality_gate=gate,  # type: ignore[arg-type]
        verification_failures=len([v for v in result.verification_results if not v.passed]),
    )


def distill_evidence(
    result: AgentRunResult,
    *,
    run_id: str | None = None,
    task: str = "unknown",
    manifest_checksum: str | None = None,
    raw_trace_path: str | Path | None = None,
) -> TrajectoryEvidence:
    """Convert a finished :class:`AgentRunResult` into a :class:`TrajectoryEvidence`.

    The pipeline is deterministic: every event id, packet id (except UUID v7
    components), and counter is derived from the inputs. Secret patterns in tool
    output are redacted before being persisted.
    """

    timeline = _build_timeline(result)
    packets = _build_packets(result, timeline)
    failure_map = _build_failure_map(packets)
    summary = _build_summary(run_id=run_id or f"run-{uuid7()}", task=task, result=result)
    return TrajectoryEvidence(
        run_summary=summary,
        failure_map=failure_map,
        decision_timeline=tuple(timeline),
        evidence_packets=tuple(packets),
        raw_trace_path=str(raw_trace_path) if raw_trace_path else None,
        manifest_checksum=manifest_checksum,
    )


def replay_raw_trace(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL raw trace file and return one decoded record per line."""

    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"raw trace not found: {target}")
    records: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


__all__ = [
    "DEFAULT_TAXONOMY_VERSION",
    "distill_evidence",
    "replay_raw_trace",
]
