"""Tests for the section-aware context engineering engine."""

from __future__ import annotations

from anycode.core.context_manager import ContextManager
from anycode.types import (
    ContextPolicy,
    ContextSectionBudget,
    ContextSectionInput,
    LLMMessage,
    ModelContextProfile,
    TextBlock,
    TokenUsage,
    ToolResultBlock,
)


def _msgs(*texts: tuple[str, str]) -> list[LLMMessage]:
    return [LLMMessage(role=role, content=[TextBlock(text=t)]) for role, t in texts]  # type: ignore[arg-type]


def test_assemble_emits_usage_report_with_sections() -> None:
    policy = ContextPolicy(enabled=True, mode="manual", max_context_tokens=10_000, reserved_response_tokens=1_000)
    cm = ContextManager(policy, provider="fake", model="fake-model")
    messages = _msgs(("user", "hello world"), ("assistant", "hi back"))
    _, manifest = cm.assemble(messages, system_prompt="You are a helper.", tool_definitions_text='[{"name":"echo"}]')
    report = manifest.usage_report
    assert report is not None
    kinds = {s.kind for s in report.sections}
    assert "reserved_response" in kinds
    assert "system_instructions" in kinds
    assert "tool_definitions" in kinds
    assert "user_messages" in kinds


def test_reserved_response_subtracted_from_window() -> None:
    policy = ContextPolicy(enabled=True, max_context_tokens=8_000, reserved_response_tokens=2_000)
    cm = ContextManager(policy, provider="fake", model="fake-model")
    _, manifest = cm.assemble(_msgs(("user", "test")))
    assert manifest.usage_report is not None
    assert manifest.usage_report.reserved_response_tokens == 2_000


def test_section_trim_truncates_oversized_tool_results() -> None:
    big_payload = "X" * 4000  # ~1000 heuristic tokens
    msgs = [
        LLMMessage(role="user", content=[ToolResultBlock(tool_use_id="t1", content=big_payload)]),
        LLMMessage(role="user", content=[ToolResultBlock(tool_use_id="t2", content=big_payload)]),
    ]
    policy = ContextPolicy(
        enabled=True,
        max_context_tokens=100_000,
        sections={
            "tool_results": ContextSectionBudget(kind="tool_results", max_tokens=200, overflow="trim"),
        },
    )
    cm = ContextManager(policy, provider="fake", model="fake-model")
    out, manifest = cm.assemble(msgs)
    # After trim, the second/oldest result should be heavily truncated.
    contents = [b.content for m in out for b in m.content if isinstance(b, ToolResultBlock)]
    assert any(len(c) < len(big_payload) for c in contents)
    # Manifest records the trim strategy.
    assert manifest.usage_report is not None
    strategies = {s.strategy_applied for s in manifest.usage_report.sections if s.kind == "tool_results"}
    assert "trim" in strategies


def test_section_drop_removes_kind_entirely() -> None:
    msgs = [LLMMessage(role="user", content=[ToolResultBlock(tool_use_id="t1", content="payload payload payload")])]
    policy = ContextPolicy(
        enabled=True,
        max_context_tokens=100_000,
        sections={"tool_results": ContextSectionBudget(kind="tool_results", max_tokens=1, overflow="drop")},
    )
    cm = ContextManager(policy, provider="fake", model="fake-model")
    out, _manifest = cm.assemble(msgs)
    assert all(not isinstance(b, ToolResultBlock) for m in out for b in m.content)


def test_reconcile_attaches_actual_input_tokens() -> None:
    policy = ContextPolicy(enabled=True, max_context_tokens=10_000)
    cm = ContextManager(policy, provider="fake", model="fake-model")
    _, manifest = cm.assemble(_msgs(("user", "hello")))
    reconciled = ContextManager.reconcile(manifest, TokenUsage(input_tokens=42, output_tokens=7))
    assert reconciled.actual_input_tokens == 42
    assert reconciled.usage_report is not None
    assert reconciled.usage_report.counting_confidence == "provider"


def test_auto_mode_uses_profile_window() -> None:
    profile = ModelContextProfile(
        provider="fake",
        model="huge-model",
        max_context_tokens=2_000_000,
    )
    policy = ContextPolicy(enabled=True, mode="auto", model_profile=profile)
    cm = ContextManager(policy, provider="fake", model="huge-model")
    assert cm.effective_window == 2_000_000


def test_auto_mode_unbounded_when_profile_has_no_max() -> None:
    profile = ModelContextProfile(provider="fake", model="unbounded", max_context_tokens=None)
    policy = ContextPolicy(enabled=True, mode="auto", model_profile=profile)
    cm = ContextManager(policy, provider="fake", model="unbounded")
    assert cm.effective_window is None
    _, manifest = cm.assemble(_msgs(("user", "x" * 1000)))
    assert manifest.max_tokens == 0  # 0 represents unbounded in the manifest.
    assert manifest.pressure == "normal"  # No ceiling => no pressure.


def test_assemble_accepts_first_class_context_sections() -> None:
    policy = ContextPolicy(enabled=True, max_context_tokens=20_000, reserved_response_tokens=500)
    cm = ContextManager(policy, provider="fake", model="fake-model")
    out, manifest = cm.assemble(
        _msgs(("user", "Use the supplied context.")),
        context_sections=(
            ContextSectionInput(kind="files", label="src/app.py", content="def run():\n    return 'ok'"),
            ContextSectionInput(kind="memory_rag", label="prior_session", content="The project prefers deterministic tests."),
            ContextSectionInput(kind="verification", label="pytest", content="tests/test_app.py::test_run failed before the fix."),
        ),
        task_state={"phase": "verify"},
    )

    section_text = "\n".join(block.text for msg in out for block in msg.content if isinstance(block, TextBlock))
    assert "Files and Code Excerpts: src/app.py" in section_text
    assert "Memory and RAG Context: prior_session" in section_text
    assert "Task and Checkpoint State: task_state" in section_text
    assert "Verification State: pytest" in section_text
    assert manifest.usage_report is not None
    kinds = {section.kind for section in manifest.usage_report.sections}
    assert {"files", "memory_rag", "task_state", "verification"}.issubset(kinds)
    source_kinds = {source.kind for source in manifest.sources}
    assert {"files", "external_memory", "task_state", "verification"}.issubset(source_kinds)


def test_context_section_budget_can_offload(tmp_path) -> None:
    policy = ContextPolicy(
        enabled=True,
        max_context_tokens=20_000,
        artifact_dir=str(tmp_path),
        sections={"files": ContextSectionBudget(kind="files", max_tokens=10, overflow="offload")},
    )
    cm = ContextManager(policy, provider="fake", model="fake-model")
    out, manifest = cm.assemble(
        _msgs(("user", "Read the large file.")),
        file_contexts={"huge.py": "print('x')\n" * 200},
    )

    section_text = "\n".join(block.text for msg in out for block in msg.content if isinstance(block, TextBlock))
    assert "OFFLOADED ARTIFACT" in section_text
    assert manifest.offloaded
    assert manifest.usage_report is not None
    file_sections = [section for section in manifest.usage_report.sections if section.kind == "files"]
    assert file_sections[0].strategy_applied == "offload"
