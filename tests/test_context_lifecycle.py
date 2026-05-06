"""Tests for adaptive context lifecycle."""

from __future__ import annotations

from pathlib import Path

import pytest

from anycode import (
    ContextManager,
    ContextManifest,
    ContextPolicy,
    LLMMessage,
    TextBlock,
    ToolResultBlock,
    estimate_messages_tokens,
    offload_text,
    rebuild_from_handoff,
    restore_text,
)


def _msg(role: str, text: str) -> LLMMessage:
    return LLMMessage(role=role, content=[TextBlock(text=text)])


def _tool_result_msg(payload: str) -> LLMMessage:
    return LLMMessage(role="user", content=[ToolResultBlock(tool_use_id="call_1", content=payload)])


def test_token_estimate_grows_with_content() -> None:
    base = estimate_messages_tokens([_msg("user", "hi")])
    bigger = estimate_messages_tokens([_msg("user", "hello world " * 200)])
    assert bigger > base


def test_normal_pressure_passes_through(tmp_path: Path) -> None:
    policy = ContextPolicy(enabled=True, max_context_tokens=1_000_000, artifact_dir=str(tmp_path))
    manager = ContextManager(policy)
    messages = [_msg("user", "hi"), _msg("assistant", "ok")]
    new_messages, manifest = manager.assemble(messages)
    assert manifest.pressure == "normal"
    assert new_messages == messages
    assert manifest.offloaded == []
    assert manifest.compaction_summary is None


def test_offload_pressure_offloads_large_tool_output(tmp_path: Path) -> None:
    big = "x" * 80_000  # ~20k tokens at 0.25 tokens/char
    policy = ContextPolicy(
        enabled=True,
        max_context_tokens=10_000,
        offload_ratio=0.3,
        compact_ratio=0.95,
        handoff_ratio=0.99,
        max_tool_output_tokens=200,
        artifact_dir=str(tmp_path),
    )
    manager = ContextManager(policy)
    new_messages, manifest = manager.assemble([_msg("user", "do work"), _tool_result_msg(big)])
    assert manifest.pressure in ("offload", "compact", "handoff")
    assert len(manifest.offloaded) == 1
    artifact = manifest.offloaded[0]
    assert Path(artifact.path).exists()
    assert restore_text(artifact) == big
    placeholder_block = new_messages[-1].content[0]
    assert isinstance(placeholder_block, ToolResultBlock)
    assert "OFFLOADED ARTIFACT" in placeholder_block.content


def test_compact_pressure_summarizes_history(tmp_path: Path) -> None:
    policy = ContextPolicy(
        enabled=True,
        max_context_tokens=400,
        compact_ratio=0.4,
        handoff_ratio=0.99,
        keep_recent_messages=2,
        artifact_dir=str(tmp_path),
    )
    manager = ContextManager(policy)
    history = [_msg("user" if i % 2 == 0 else "assistant", f"step {i} " + "lorem " * 30) for i in range(10)]
    new_messages, manifest = manager.assemble(history)
    assert manifest.pressure in ("compact", "handoff")
    assert manifest.compaction_summary is not None
    assert len(new_messages) <= 1 + policy.keep_recent_messages
    assert any("CONTEXT COMPACTED" in str(b) for b in new_messages[0].content)


def test_handoff_pressure_writes_recoverable_file(tmp_path: Path) -> None:
    policy = ContextPolicy(
        enabled=True,
        max_context_tokens=200,
        handoff_ratio=0.4,
        compact_ratio=0.3,
        offload_ratio=0.2,
        trim_ratio=0.1,
        keep_recent_messages=2,
        artifact_dir=str(tmp_path),
    )
    manager = ContextManager(policy)
    history = [_msg("user", "do stuff " + "data " * 50) for _ in range(6)]
    _, manifest = manager.assemble(history)
    assert manifest.pressure == "handoff"
    assert manifest.handoff_path is not None
    restored = rebuild_from_handoff(manifest.handoff_path)
    assert len(restored) >= 1
    assert isinstance(restored[0], LLMMessage)


def test_disabled_policy_returns_normal(tmp_path: Path) -> None:
    policy = ContextPolicy(enabled=False, max_context_tokens=10, artifact_dir=str(tmp_path))
    manager = ContextManager(policy)
    new_messages, manifest = manager.assemble([_msg("user", "x" * 10_000)])
    assert manifest.pressure == "normal"
    assert manifest.offloaded == []
    assert new_messages[0].content[0].text.startswith("x")  # type: ignore[union-attr]


def test_offload_text_digest_roundtrip(tmp_path: Path) -> None:
    artifact = offload_text("hello world", tmp_path)
    assert restore_text(artifact) == "hello world"


def test_offload_text_rejects_tampered_payload(tmp_path: Path) -> None:
    artifact = offload_text("hello world", tmp_path)
    Path(artifact.path).write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError):
        restore_text(artifact)


def test_manifest_is_immutable(tmp_path: Path) -> None:
    policy = ContextPolicy(enabled=True, max_context_tokens=10_000, artifact_dir=str(tmp_path))
    manager = ContextManager(policy)
    _, manifest = manager.assemble([_msg("user", "hi")])
    assert isinstance(manifest, ContextManifest)
    with pytest.raises(Exception):
        manifest.pressure = "trim"  # type: ignore[misc]
