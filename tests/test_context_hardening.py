"""Tests for context lifecycle hardening: mask stage, archive-first compaction,
summarizer fallback, invariant re-injection, provider calibration."""

from __future__ import annotations

import json
from pathlib import Path

from anycode.core.context_manager import ContextManager, _classify_pressure
from anycode.types import (
    ContextPolicy,
    LLMMessage,
    TextBlock,
    TokenUsage,
    ToolResultBlock,
    ToolUseBlock,
)


def _policy(tmp_path: Path, **overrides: object) -> ContextPolicy:
    defaults: dict[str, object] = {
        "enabled": True,
        "max_context_tokens": 100_000,
        "artifact_dir": str(tmp_path / "artifacts"),
        "keep_recent_messages": 2,
    }
    defaults.update(overrides)
    return ContextPolicy(**defaults)  # type: ignore[arg-type]


def _tool_turn(i: int, output_chars: int = 600) -> list[LLMMessage]:
    return [
        LLMMessage(
            role="assistant",
            content=[TextBlock(text=f"step {i}"), ToolUseBlock(id=f"t{i}", name="run", input={"i": i})],
        ),
        LLMMessage(role="user", content=[ToolResultBlock(tool_use_id=f"t{i}", content="x" * output_chars)]),
    ]


def test_pressure_ladder_includes_mask_stage(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    assert _classify_pressure(0.66, policy) == "trim"
    assert _classify_pressure(0.71, policy) == "mask"
    assert _classify_pressure(0.76, policy) == "offload"
    assert _classify_pressure(0.86, policy) == "compact"
    assert _classify_pressure(0.96, policy) == "handoff"


def test_mask_stage_replaces_aged_tool_results(tmp_path: Path) -> None:
    messages = [m for i in range(6) for m in _tool_turn(i)]
    # Size the window so the measured estimate lands in the mask band (70-75%).
    probe = ContextManager(_policy(tmp_path), provider="fake", model="fake-model")
    estimated = probe.tokenizer.count_messages(messages)
    policy = _policy(tmp_path, max_context_tokens=int(estimated / 0.72))
    manager = ContextManager(policy, provider="fake", model="fake-model")

    prepared, manifest = manager.assemble(messages)
    assert manifest.pressure == "mask", manifest.pressure

    aged = prepared[:-2]
    masked = [
        b
        for m in aged
        for b in m.content
        if isinstance(b, ToolResultBlock) and b.content.startswith("[masked tool result:")
    ]
    assert masked, "aged tool results should be masked"
    # tool_use pairing is preserved on masked results.
    assert all(b.tool_use_id for b in masked)
    # The protected recency window stays verbatim.
    recent_results = [b for m in prepared[-2:] for b in m.content if isinstance(b, ToolResultBlock)]
    assert all(not b.content.startswith("[masked") for b in recent_results)


def test_compaction_archives_full_history_first(tmp_path: Path) -> None:
    policy = _policy(tmp_path, max_context_tokens=900)
    manager = ContextManager(policy, provider="fake", model="fake-model")
    messages = [m for i in range(8) for m in _tool_turn(i)]

    _prepared, manifest = manager.assemble(messages)
    assert manifest.pressure in ("compact", "handoff")
    assert manifest.archive_path is not None
    archived = json.loads(Path(manifest.archive_path).read_text(encoding="utf-8"))
    # The archive holds the untouched pre-compaction history.
    assert len(archived["messages"]) == len(messages)
    # The summary points at the archive so nothing is unrecoverable.
    assert manifest.compaction_summary is not None
    assert manifest.archive_path in manifest.compaction_summary


def test_invariants_reinjected_after_compaction(tmp_path: Path) -> None:
    policy = _policy(
        tmp_path,
        max_context_tokens=900,
        preserved_task_state={"objective": "finish migration"},
    )
    manager = ContextManager(policy, provider="fake", model="fake-model")
    messages = [m for i in range(8) for m in _tool_turn(i)]

    prepared, manifest = manager.assemble(messages)
    assert manifest.pressure in ("compact", "handoff")
    last_texts = [b.text for m in prepared[-3:] for b in m.content if isinstance(b, TextBlock)]
    assert any("[context-invariants]" in t and "objective: finish migration" in t for t in last_texts)


def test_summarizer_used_when_provided(tmp_path: Path) -> None:
    calls: list[int] = []

    def summarizer(head: list[LLMMessage]) -> str:
        calls.append(len(head))
        return "INTENT: migrate DB. CHANGES: schema v2. DECISIONS: keep WAL. NEXT: backfill."

    policy = _policy(tmp_path, max_context_tokens=900)
    manager = ContextManager(policy, provider="fake", model="fake-model", summarizer=summarizer)
    messages = [m for i in range(8) for m in _tool_turn(i)]

    _prepared, manifest = manager.assemble(messages)
    assert calls, "summarizer should be invoked at compact pressure"
    assert manifest.compaction_summary is not None
    assert "INTENT: migrate DB" in manifest.compaction_summary


def test_summarizer_failure_falls_back_deterministically(tmp_path: Path) -> None:
    def broken_summarizer(head: list[LLMMessage]) -> str:
        raise RuntimeError("LLM unavailable")

    policy = _policy(tmp_path, max_context_tokens=900)
    manager = ContextManager(policy, provider="fake", model="fake-model", summarizer=broken_summarizer)
    messages = [m for i in range(8) for m in _tool_turn(i)]

    _prepared, manifest = manager.assemble(messages)
    # Degrade, never corrupt: the deterministic extractive summary still lands.
    assert manifest.compaction_summary is not None
    assert "[CONTEXT COMPACTED]" in manifest.compaction_summary


def test_calibration_tightens_pressure_from_provider_actuals(tmp_path: Path) -> None:
    policy = _policy(tmp_path, max_context_tokens=10_000)
    manager = ContextManager(policy, provider="fake", model="fake-model")
    messages = [m for i in range(4) for m in _tool_turn(i)]

    _prepared, manifest = manager.assemble(messages)
    assert manager.calibration == 1.0

    # Provider reports twice the locally estimated tokens.
    report = manifest.usage_report
    assert report is not None
    actual = (report.used_tokens - report.reserved_response_tokens) * 2
    reconciled = ContextManager.reconcile(manifest, TokenUsage(input_tokens=actual, output_tokens=10))
    manager.note_actual(reconciled)
    assert manager.calibration > 1.0

    # Repeated signals converge upward but stay clamped.
    for _ in range(20):
        manager.note_actual(reconciled)
    assert 1.0 < manager.calibration <= 3.0
