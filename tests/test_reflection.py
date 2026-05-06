"""Tests for self-reflection (Phase 5.1)."""

from __future__ import annotations

import pytest

from anycode import AgentInfo, AgentRunResult, CriticResult, ReflectionConfig, TokenUsage
from anycode.reflection.evaluator import parse_critic_json
from anycode.reflection.loop import ReflectionLoop


def test_parse_critic_json_happy() -> None:
    result = parse_critic_json(
        '{"score": 0.85, "passed": true, "feedback": "great", "suggestions": ["a"]}',
        threshold=0.7,
    )
    assert result.score == 0.85
    assert result.passed is True
    assert result.suggestions == ["a"]


def test_parse_critic_json_malformed_returns_default() -> None:
    result = parse_critic_json("not json at all", threshold=0.7)
    assert 0.0 <= result.score <= 1.0
    assert result.passed is False


def test_parse_critic_json_extracts_from_prose() -> None:
    text = 'Sure! Here is my review: {"score": 0.9, "passed": true, "feedback": "ok"}'
    result = parse_critic_json(text, threshold=0.7)
    assert result.score == 0.9


class _StubAgent:
    name = "stub"

    def __init__(self, outputs: list[str]) -> None:
        self._outputs = outputs
        self.calls = 0

    async def run(self, prompt: str) -> AgentRunResult:
        idx = min(self.calls, len(self._outputs) - 1)
        self.calls += 1
        return AgentRunResult(
            success=True,
            output=self._outputs[idx],
            messages=[],
            token_usage=TokenUsage(input_tokens=10, output_tokens=10),
            tool_calls=[],
        )


class _StubCritic:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.calls = 0

    async def evaluate(self, output: str, prompt: str, context: AgentInfo) -> CriticResult:
        idx = min(self.calls, len(self._scores) - 1)
        score = self._scores[idx]
        self.calls += 1
        return CriticResult(score=score, passed=score >= 0.8, feedback="f", suggestions=[])


@pytest.mark.asyncio
async def test_reflection_loop_passes_first_try() -> None:
    config = ReflectionConfig(enabled=True, mode="custom", quality_threshold=0.8, max_reflections=2, custom_critic=_StubCritic([0.9]))
    loop = ReflectionLoop(config)
    agent = _StubAgent(["good"])
    info = AgentInfo(name="stub", role="r", model="claude-haiku-4-5")
    result = await loop.run(agent, "prompt", agent_info=info, agent_provider="anthropic")
    assert agent.calls == 1
    assert result.reflections_count == 0
    assert result.quality_score == 0.9


@pytest.mark.asyncio
async def test_reflection_loop_retries_until_threshold() -> None:
    config = ReflectionConfig(enabled=True, mode="custom", quality_threshold=0.8, max_reflections=3, custom_critic=_StubCritic([0.4, 0.6, 0.95]))
    loop = ReflectionLoop(config)
    agent = _StubAgent(["v1", "v2", "v3"])
    info = AgentInfo(name="stub", role="r", model="m")
    result = await loop.run(agent, "p", agent_info=info, agent_provider="anthropic")
    assert agent.calls == 3
    assert result.reflections_count == 2
    assert result.quality_score == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_reflection_loop_disabled_passthrough() -> None:
    loop = ReflectionLoop(ReflectionConfig(enabled=False))
    agent = _StubAgent(["once"])
    info = AgentInfo(name="stub", role="r", model="m")
    result = await loop.run(agent, "p", agent_info=info, agent_provider=None)
    assert agent.calls == 1
    assert result.output == "once"
