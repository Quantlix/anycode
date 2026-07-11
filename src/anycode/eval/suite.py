"""Run an EvalSuite of scenarios against real LLM providers or a deterministic fake.

Provider selection:
  - Per-scenario `deterministic=True` always uses the in-memory FakeAdapter and
    replays `fake_responses` (one assistant text per turn) without calling any
    external service.
  - Per-scenario `provider` and `model` fields take precedence.
  - Otherwise, scan environment for `ANTHROPIC_API_KEY` (claude-sonnet-4-5) or
    `OPENAI_API_KEY` (gpt-4o-mini), in that order.
  - If a live scenario has no provider available, RuntimeError is raised so the
    suite fails loud instead of silently falling back to fake data.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable

from anycode.core.agent import Agent
from anycode.core.runner import AgentRunner
from anycode.eval.scorer import score
from anycode.helpers.usage_tracker import EMPTY_USAGE
from anycode.providers.fake import FakeAdapter, FakeResponse
from anycode.security.redaction import safe_exception_message
from anycode.tools.executor import ToolExecutor
from anycode.tools.registry import ToolRegistry
from anycode.types import (
    AgentConfig,
    AgentRunResult,
    EvalReport,
    EvalScenario,
    EvalScenarioResult,
    LLMMessage,
    RunnerOptions,
    TextBlock,
)

_PROVIDER_DEFAULT_MODEL: dict[str, str] = {
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-4o-mini",
}
_PROVIDER_ENV_KEY: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def detect_provider() -> tuple[str, str]:
    """Return (provider, model) from env. Raises RuntimeError when no key is set."""
    for provider, env_key in _PROVIDER_ENV_KEY.items():
        if os.getenv(env_key):
            return provider, _PROVIDER_DEFAULT_MODEL[provider]
    raise RuntimeError("No LLM provider available. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in the environment.")


def _resolve_provider(scenario: EvalScenario) -> tuple[str, str]:
    if scenario.provider and scenario.model:
        return scenario.provider, scenario.model
    provider, model = detect_provider()
    return scenario.provider or provider, scenario.model or model


def build_agent(scenario: EvalScenario) -> Agent:
    """Construct a real Agent for `scenario` using a live provider."""
    provider, model = _resolve_provider(scenario)
    config = AgentConfig(
        name=scenario.name,
        provider=provider,  # type: ignore[arg-type]
        model=model,
        system_prompt=scenario.system_prompt,
        tools=list(scenario.allowed_tools) if scenario.allowed_tools else None,
        max_turns=scenario.max_turns,
        max_tokens=scenario.max_tokens,
        temperature=scenario.temperature,
    )
    registry = ToolRegistry()
    return Agent(config, tool_registry=registry, tool_executor=ToolExecutor(registry))


async def _run_deterministic(scenario: EvalScenario) -> AgentRunResult:
    """Execute a scenario using the FakeAdapter without any external service."""
    responses = [FakeResponse(text=t) for t in scenario.fake_responses] or [FakeResponse(text="OK")]
    adapter = FakeAdapter(responses=responses, model_name=scenario.model or "fake-model")
    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    options = RunnerOptions(
        model=scenario.model or "fake-model",
        system_prompt=scenario.system_prompt,
        max_turns=scenario.max_turns,
        max_tokens=scenario.max_tokens,
        temperature=scenario.temperature,
        allowed_tools=list(scenario.allowed_tools) if scenario.allowed_tools else None,
        agent_name=scenario.name,
    )
    runner = AgentRunner(adapter, registry, executor, options)
    seed = [LLMMessage(role="user", content=[TextBlock(text=scenario.prompt)])]
    run_result = await runner.run(seed)
    return AgentRunResult(
        success=run_result.stop_reason is None or run_result.stop_reason.code == "success",
        output=run_result.output,
        messages=run_result.messages,
        token_usage=run_result.token_usage or EMPTY_USAGE,
        tool_calls=run_result.tool_calls,
        terminal_phase=run_result.terminal_phase,
        stop_reason=run_result.stop_reason,
        lifecycle_events=run_result.lifecycle_events,
        context_manifests=run_result.context_manifests,
        verification_results=run_result.verification_results,
        gate_decisions=run_result.gate_decisions,
        retries=run_result.retries,
    )


async def run_scenario(scenario: EvalScenario, *, agent: Agent | None = None) -> EvalScenarioResult:
    """Execute one scenario against a real LLM provider or the deterministic fake."""
    started = time.monotonic()
    failure: str | None = None
    result: AgentRunResult | None = None
    try:
        if scenario.deterministic:
            result = await _run_deterministic(scenario)
        else:
            runner_agent = agent or build_agent(scenario)
            result = await runner_agent.run(scenario.prompt)
    except Exception as exc:  # noqa: BLE001 - capture as scoring failure
        failure = f"{type(exc).__name__}: {safe_exception_message(exc)}"
    elapsed = time.monotonic() - started
    return score(scenario, result, runtime_seconds=elapsed, failure_reason=failure)


async def run_suite(
    scenarios: Iterable[EvalScenario],
    *,
    suite_name: str = "default",
    harness_variant: str = "baseline",
) -> EvalReport:
    """Run every scenario sequentially and aggregate the results into an EvalReport."""
    materialized = list(scenarios)
    results: list[EvalScenarioResult] = []
    for scenario in materialized:
        results.append(await run_scenario(scenario))

    passed = sum(1 for r in results if r.passed)
    return EvalReport(
        suite_name=suite_name,
        harness_variant=harness_variant,
        total_scenarios=len(materialized),
        passed=passed,
        failed=len(materialized) - passed,
        total_runtime_seconds=sum(r.runtime_seconds for r in results),
        total_input_tokens=sum(r.token_usage.input_tokens for r in results),
        total_output_tokens=sum(r.token_usage.output_tokens for r in results),
        total_cost_usd=sum(r.cost_usd for r in results),
        total_retries=sum(r.retries for r in results),
        total_verification_failures=sum(r.verification_failures for r in results),
        scenario_results=tuple(results),
    )
