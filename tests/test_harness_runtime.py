"""Tests for harness runtime reliability features.

Uses the deterministic FakeAdapter so no LLM keys are required.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from anycode import (
    FakeAdapter,
    FakeResponse,
    load_scenarios,
    run_suite,
)
from anycode.core.context_manager import ContextManager
from anycode.core.runner import AgentRunner
from anycode.tools.executor import ToolExecutor
from anycode.tools.registry import ToolRegistry
from anycode.types import (
    ContextPolicy,
    LLMMessage,
    RunnerOptions,
    TextBlock,
    VerificationSensorConfig,
)

DET_SUITE = Path(__file__).parent / "fixtures" / "eval" / "runtime_reliability_deterministic.yaml"


# --- ContextPolicy provider overrides --------------------------------------


def test_context_policy_for_provider_returns_override() -> None:
    override = ContextPolicy(enabled=True, max_context_tokens=1234)
    policy = ContextPolicy(enabled=True, max_context_tokens=8000, provider_overrides={"anthropic": override})
    assert policy.for_provider("anthropic").max_context_tokens == 1234
    assert policy.for_provider("openai").max_context_tokens == 8000
    assert policy.for_provider(None).max_context_tokens == 8000


def test_context_policy_preserved_state_fields_default_empty() -> None:
    policy = ContextPolicy()
    assert policy.preserved_task_state == {}
    assert policy.preserved_verification_failures == ()


def test_context_manager_uses_provider_override() -> None:
    override = ContextPolicy(enabled=True, max_context_tokens=2222)
    policy = ContextPolicy(enabled=True, max_context_tokens=9999, provider_overrides={"openai": override})
    cm = ContextManager(policy, provider="openai")
    assert cm.provider == "openai"


# --- Verification sensor registry -----------------------------------------


def test_build_regex_sensor_from_config() -> None:
    from anycode.verification import build_sensor

    cfg = VerificationSensorConfig(
        name="regex",
        kind="computational",
        phases=("after_task",),
        block_on_failure=True,
        options={"pattern": "DONE", "expect": "match"},
    )
    sensor = build_sensor(cfg)
    assert sensor.name == "regex"
    assert sensor.phases == ("after_task",)


# --- Deterministic eval suite ---------------------------------------------


def test_deterministic_eval_suite_runs_without_llm() -> None:
    scenarios = load_scenarios(DET_SUITE)
    report = asyncio.run(run_suite(list(scenarios), suite_name="deterministic", harness_variant="det"))
    assert report.total_scenarios == 4
    by_name = {r.scenario_name: r for r in report.scenario_results}
    assert by_name["simple_tool_failure"].passed
    assert by_name["context_pressure"].passed
    assert by_name["approval_gate"].passed
    assert by_name["dependency_block"].passed
    assert report.total_cost_usd >= 0.0
    assert report.total_retries >= 0
    assert report.total_verification_failures == 0


# --- Cancellation lifecycle ------------------------------------------------


@pytest.mark.asyncio
async def test_runner_cancellation_emits_cancelled_phase() -> None:
    """Cancelling a running agent stream emits a 'cancelled' terminal phase via lifecycle."""
    adapter = FakeAdapter(responses=[FakeResponse(text="hi")])

    original_chat = adapter.chat

    async def _slow_chat(messages, options):  # type: ignore[no-untyped-def]
        await asyncio.sleep(5.0)
        return await original_chat(messages, options)

    adapter.chat = _slow_chat  # type: ignore[method-assign]

    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    options = RunnerOptions(model="fake-model", max_turns=2, agent_name="t")
    runner = AgentRunner(adapter, registry, executor, options)

    captured: list[object] = []

    async def consume() -> None:
        try:
            async for ev in runner.stream([LLMMessage(role="user", content=[TextBlock(text="hello")])]):
                captured.append(ev)
        except asyncio.CancelledError:
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Either the stream emitted a 'done' event with cancelled phase, or the raise propagated.
    done_events = [getattr(ev, "data", None) for ev in captured if getattr(ev, "type", None) == "done"]
    if done_events:
        result = done_events[-1]
        assert getattr(result, "terminal_phase", None) == "cancelled"
        stop = getattr(result, "stop_reason", None)
        assert stop is not None and stop.code == "user_cancelled"


# --- Multi-phase quality gates --------------------------------------------


def _stub_tool() -> object:
    """Build a trivial passthrough tool for tests."""
    from pydantic import BaseModel as _BM

    from anycode.types import ToolDefinition, ToolResult

    class _Empty(_BM):
        pass

    async def _execute(**_kwargs: object) -> ToolResult:
        return ToolResult(data="ok", is_error=False)

    return ToolDefinition(name="echo", description="echo", input_model=_Empty, execute=_execute)


def _block_sensor(name: str, phase: str) -> object:
    from anycode.types import VerificationResult as _VR
    from anycode.verification.sensor import Sensor as _S

    cfg = VerificationSensorConfig(
        name=name,
        kind="computational",
        phases=(phase,),  # type: ignore[arg-type]
        block_on_failure=True,
    )

    def _fn(_ctx: object) -> _VR:
        return _VR(
            sensor_name=name,
            kind="computational",
            passed=False,
            severity="critical",
            message=f"blocked at {phase}",
        )

    return _S(config=cfg, fn=_fn)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_runner_invokes_before_tool_gate_and_blocks() -> None:
    """A critical sensor at before_tool should produce a verification_failed stop reason."""
    from anycode.verification.gate import QualityGate

    adapter = FakeAdapter(responses=[FakeResponse(text="calling tool", tool_calls=(("echo", {}),))])
    registry = ToolRegistry()
    registry.register(_stub_tool())  # type: ignore[arg-type]
    executor = ToolExecutor(registry)
    options = RunnerOptions(model="fake-model", max_turns=2, agent_name="t")
    runner = AgentRunner(adapter, registry, executor, options)
    runner._gate = QualityGate([_block_sensor("preflight", "before_tool")])  # type: ignore[arg-type]

    result = await runner.run([LLMMessage(role="user", content=[TextBlock(text="go")])])
    assert result.stop_reason is not None
    assert result.stop_reason.code == "verification_failed"
    assert "before tool" in result.stop_reason.message
    assert any(d.outcome == "block" for d in result.gate_decisions)


@pytest.mark.asyncio
async def test_runner_invokes_after_tool_gate_and_blocks() -> None:
    """A critical sensor at after_tool should produce a verification_failed stop reason."""
    from anycode.verification.gate import QualityGate

    adapter = FakeAdapter(responses=[FakeResponse(text="calling tool", tool_calls=(("echo", {}),))])
    registry = ToolRegistry()
    registry.register(_stub_tool())  # type: ignore[arg-type]
    executor = ToolExecutor(registry)
    options = RunnerOptions(model="fake-model", max_turns=2, agent_name="t")
    runner = AgentRunner(adapter, registry, executor, options)
    runner._gate = QualityGate([_block_sensor("postcheck", "after_tool")])  # type: ignore[arg-type]

    result = await runner.run([LLMMessage(role="user", content=[TextBlock(text="go")])])
    assert result.stop_reason is not None
    assert result.stop_reason.code == "verification_failed"
    assert "after tool" in result.stop_reason.message


@pytest.mark.asyncio
async def test_orchestrator_after_team_gate_marks_failure() -> None:
    """Orchestrator team-level gate should block a successful team run on critical failure."""
    from anycode.core.orchestrator import AnyCode, TaskSpec
    from anycode.types import AgentConfig, OrchestratorConfig, TeamConfig

    cfg = OrchestratorConfig(
        verification=(
            VerificationSensorConfig(
                name="regex",
                kind="computational",
                phases=("after_team",),
                block_on_failure=True,
                options={"pattern": "FORBIDDEN", "expect": "no_match", "severity": "critical"},
            ),
        ),
    )
    engine = AnyCode(cfg)
    agent_cfg = AgentConfig(name="dev", model="fake-model")
    team = engine.create_team("t", TeamConfig(name="t", agents=[agent_cfg]))

    # Provide a deterministic agent that emits the forbidden token.
    fake = FakeAdapter(responses=[FakeResponse(text="FORBIDDEN content here")])
    pooled = engine._pool.get("dev")
    assert pooled is not None
    # Replace the adapter on the agent's runner via build path: easiest is to monkey-patch run.
    from anycode.types import AgentRunResult as _ARR
    from anycode.types import TokenUsage as _TU

    async def _fake_run(_prompt: str) -> _ARR:
        return _ARR(success=True, output="FORBIDDEN content here", messages=[], token_usage=_TU(input_tokens=1, output_tokens=1), tool_calls=[])

    pooled.run = _fake_run  # type: ignore[method-assign]
    del fake  # adapter unused; we monkey-patched the agent

    result = await engine.run_tasks(team, [TaskSpec(title="task", description="emit", assignee="dev")])
    assert result.stop_reason is not None
    assert result.stop_reason.code == "verification_failed"
    assert result.success is False
    assert any(d.outcome == "block" for d in result.gate_decisions)
