"""Tests for automatic context reset (handoff) and session chaining."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anycode.core.context_manager import build_invariant_message, rebuild_from_handoff
from anycode.core.runner import AgentRunner
from anycode.core.session_chain import SessionChain, load_contract, save_contract
from anycode.providers.fake import FakeAdapter, FakeResponse
from anycode.tools.executor import ToolExecutor
from anycode.tools.registry import ToolRegistry
from anycode.types import (
    ContextPolicy,
    GoalContract,
    GoalCriterion,
    LLMMessage,
    RunnerOptions,
    RunResult,
    TextBlock,
)


def _echo_tool() -> object:
    from pydantic import BaseModel as _BM

    from anycode.types import ToolDefinition, ToolResult

    class _Empty(_BM):
        pass

    async def _execute(**_kwargs: object) -> ToolResult:
        return ToolResult(data="echoed")

    return ToolDefinition(name="echo", description="echo", input_model=_Empty, execute=_execute)


def _plain_runner(adapter: FakeAdapter, *, context_policy: ContextPolicy | None = None, max_turns: int = 12) -> AgentRunner:
    registry = ToolRegistry()
    registry.register(_echo_tool())  # type: ignore[arg-type]
    executor = ToolExecutor(registry)
    options = RunnerOptions(model="fake-model", max_turns=max_turns, agent_name="chain")
    return AgentRunner(adapter, registry, executor, options, context_policy=context_policy)


# -- automatic reset-and-rebuild --


async def test_auto_reset_on_handoff_pressure(tmp_path: Path) -> None:
    """A run that exceeds one context window completes via automatic reset."""
    policy = ContextPolicy(
        enabled=True,
        max_context_tokens=600,
        auto_reset_on_handoff=True,
        keep_recent_messages=2,
        artifact_dir=str(tmp_path / "artifacts"),
        preserved_task_state={"goal": "keep going"},
    )
    # Each turn adds ~150 tokens of text and keeps looping via a tool call,
    # so pressure climbs into handoff before the final plain response.
    filler = "long analysis text " * 30
    adapter = FakeAdapter(
        responses=[
            *[FakeResponse(text=f"{filler} step {i}", tool_calls=(("echo", {"step": i}),)) for i in range(8)],
            FakeResponse(text="all finished"),
        ]
    )
    runner = _plain_runner(adapter, context_policy=policy)

    captured: list[LLMMessage] = []
    result = await runner.run(
        [LLMMessage(role="user", content=[TextBlock(text="begin the long task")])],
        on_message=captured.append,
    )

    pressures = [m.pressure for m in result.context_manifests]
    assert "handoff" in pressures, f"expected handoff pressure, saw {pressures}"
    reset_manifest = next(m for m in result.context_manifests if m.pressure == "handoff")
    assert reset_manifest.handoff_path is not None

    # The artifact is rebuildable and carries the five-layer structure.
    payload = json.loads(Path(reset_manifest.handoff_path).read_text(encoding="utf-8"))
    assert payload["version"] == 2
    for layer in ("state", "narrative", "decisions", "next_steps", "warnings"):
        assert layer in payload
    assert payload["state"]["task_state"] == {"goal": "keep going"}
    assert rebuild_from_handoff(reset_manifest.handoff_path)

    # After a reset the run keeps going and finishes normally.
    assert result.stop_reason is not None and result.stop_reason.code == "success"


def test_invariant_message_carries_rules_and_state() -> None:
    policy = ContextPolicy(
        enabled=True,
        preserved_task_state={"phase": "implement"},
        preserved_verification_failures=("pytest failing: test_x",),
    )
    message = build_invariant_message(policy, notice="Context was reset.")
    text = message.content[0].text  # type: ignore[union-attr]
    assert message.role == "user"
    assert "Context was reset." in text
    assert "phase: implement" in text
    assert "pytest failing: test_x" in text


# -- goal contract --


def test_contract_roundtrip_and_gated_flips(tmp_path: Path) -> None:
    contract = GoalContract(
        goal="ship feature",
        criteria=(
            GoalCriterion(id="c1", description="write code"),
            GoalCriterion(id="c2", description="write tests"),
        ),
    )
    save_contract(tmp_path, contract)
    loaded = load_contract(tmp_path)
    assert loaded is not None and not loaded.complete
    assert loaded.next_incomplete().id == "c1"  # type: ignore[union-attr]

    passed = loaded.mark_passed("c1", evidence="gate:pass@turn3")
    assert passed.criteria[0].passes and passed.criteria[0].evidence == "gate:pass@turn3"
    assert not passed.criteria[1].passes
    assert passed.next_incomplete().id == "c2"  # type: ignore[union-attr]


def test_contract_persistence_redacts_secrets_by_default(tmp_path: Path) -> None:
    contract = GoalContract(
        goal="use api_key=plain-value",
        criteria=(GoalCriterion(id="c1", description="Bearer abcdefghijklmnop"),),
    )

    save_contract(tmp_path, contract)

    persisted = (tmp_path / "contract.json").read_text(encoding="utf-8")
    assert "plain-value" not in persisted
    assert "Bearer" not in persisted


async def test_session_progress_redacts_result_output(tmp_path: Path) -> None:
    contract = GoalContract(goal="g", criteria=(GoalCriterion(id="c1", description="d"),))

    async def deny_verifier(criterion, result):  # type: ignore[no-untyped-def]
        return None

    chain = SessionChain(
        runner_factory=lambda: _plain_runner(FakeAdapter.from_texts(["token=abcdefghijklmnop"])),
        contract=contract,
        work_dir=tmp_path,
        verifier=deny_verifier,
        max_sessions=1,
    )

    await chain.run()

    assert "abcdefghijklmnop" not in (tmp_path / "progress.md").read_text(encoding="utf-8")


def test_chain_refuses_changed_contract(tmp_path: Path) -> None:
    original = GoalContract(goal="g", criteria=(GoalCriterion(id="a", description="a"),))
    save_contract(tmp_path, original)
    different = GoalContract(goal="g", criteria=(GoalCriterion(id="b", description="b"),))

    async def _verifier(criterion, result):  # type: ignore[no-untyped-def]
        return None

    with pytest.raises(ValueError, match="refusing to continue"):
        SessionChain(
            runner_factory=lambda: _plain_runner(FakeAdapter.from_texts(["x"])),
            contract=different,
            work_dir=tmp_path,
            verifier=_verifier,
        )


# -- session chain --


async def test_session_chain_completes_across_fresh_contexts(tmp_path: Path) -> None:
    contract = GoalContract(
        goal="two-part task",
        criteria=(
            GoalCriterion(id="part1", description="finish part one"),
            GoalCriterion(id="part2", description="finish part two"),
        ),
    )

    session_prompts: list[str] = []

    class _RecordingAdapter(FakeAdapter):
        async def chat(self, messages, options):  # type: ignore[no-untyped-def]
            if messages:
                session_prompts.append(messages[0].content[0].text)
            return await super().chat(messages, options)

    def runner_factory() -> AgentRunner:
        # A brand-new adapter per session = a genuinely fresh context.
        return _plain_runner(_RecordingAdapter(responses=[FakeResponse(text="done with this part")]))

    async def verifier(criterion: GoalCriterion, result: RunResult) -> str | None:
        # External verification: evidence comes from the run result, not the
        # agent's claim alone.
        if result.stop_reason is not None and result.stop_reason.code == "success":
            return f"verified:{criterion.id}"
        return None

    chain = SessionChain(
        runner_factory=runner_factory,
        contract=contract,
        work_dir=tmp_path,
        verifier=verifier,
        max_sessions=5,
    )
    final = await chain.run()

    assert final.complete
    assert len(session_prompts) == 2  # one fresh session per criterion, no repeats
    assert "part1" in session_prompts[0] and "THIS SESSION: work only on criterion 'part1'" in session_prompts[0]
    assert "THIS SESSION: work only on criterion 'part2'" in session_prompts[1]
    assert "[x] part1" in session_prompts[1]  # completed work visible, not re-done

    progress = (tmp_path / "progress.md").read_text(encoding="utf-8")
    assert "criterion `part1`" in progress and "criterion `part2`" in progress
    assert load_contract(tmp_path).complete  # type: ignore[union-attr]


async def test_session_chain_does_not_flip_without_evidence(tmp_path: Path) -> None:
    contract = GoalContract(goal="g", criteria=(GoalCriterion(id="c1", description="d"),))

    async def deny_verifier(criterion, result):  # type: ignore[no-untyped-def]
        return None

    chain = SessionChain(
        runner_factory=lambda: _plain_runner(FakeAdapter.from_texts(["I claim this is done!"])),
        contract=contract,
        work_dir=tmp_path,
        verifier=deny_verifier,
        max_sessions=2,
    )
    final = await chain.run()
    assert not final.complete  # the agent's own claim is never enough
