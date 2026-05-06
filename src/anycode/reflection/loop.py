"""Reflection loop — runs an agent, evaluates output, retries with feedback if needed."""

from __future__ import annotations

import logging
from typing import Protocol

from anycode.helpers.usage_tracker import EMPTY_USAGE, merge_usage
from anycode.reflection.critic import LLMCritic
from anycode.types import AgentInfo, AgentRunResult, Critic, CriticResult, ReflectionConfig

logger = logging.getLogger(__name__)


class _AgentRunnable(Protocol):
    name: str

    async def run(self, prompt: str) -> AgentRunResult: ...


class ReflectionLoop:
    """Runs an agent under a reflection loop. Retries with critic feedback up to ``max_reflections``."""

    def __init__(self, config: ReflectionConfig) -> None:
        self._config = config

    def _resolve_critic(self, agent_model: str, agent_provider: str | None) -> Critic | None:
        if not self._config.enabled:
            return None
        if self._config.mode == "custom":
            return self._config.custom_critic
        if self._config.mode == "peer":
            model = self._config.critic_model or agent_model
            provider = self._config.critic_provider or agent_provider or "anthropic"
            return LLMCritic(model=model, provider=provider, critic_prompt=self._config.critic_prompt, threshold=self._config.quality_threshold)
        # self
        provider = agent_provider or "anthropic"
        return LLMCritic(model=agent_model, provider=provider, critic_prompt=self._config.critic_prompt, threshold=self._config.quality_threshold)

    async def run(
        self,
        agent: _AgentRunnable,
        prompt: str,
        *,
        agent_info: AgentInfo,
        agent_provider: str | None,
    ) -> AgentRunResult:
        if not self._config.enabled:
            return await agent.run(prompt)

        critic = self._resolve_critic(agent_info.model, agent_provider)
        if critic is None:
            return await agent.run(prompt)

        attempt_prompt = prompt
        best_result: AgentRunResult | None = None
        best_score = -1.0
        cumulative_usage = EMPTY_USAGE
        result: AgentRunResult | None = None

        for attempt in range(self._config.max_reflections + 1):
            result = await agent.run(attempt_prompt)
            cumulative_usage = merge_usage(cumulative_usage, result.token_usage)
            if not result.success or not result.output:
                return result.model_copy(update={"token_usage": cumulative_usage, "reflections_count": attempt})

            critique = await critic.evaluate(result.output, prompt, agent_info)
            logger.debug("Reflection attempt %d: score=%.2f passed=%s", attempt, critique.score, critique.passed)

            if critique.score > best_score:
                best_score = critique.score
                best_result = result

            if critique.passed or critique.score >= self._config.quality_threshold:
                return result.model_copy(
                    update={
                        "token_usage": cumulative_usage,
                        "reflections_count": attempt,
                        "quality_score": critique.score,
                    }
                )

            if attempt >= self._config.max_reflections:
                break

            attempt_prompt = self._build_retry_prompt(prompt, result.output, critique)

        final = best_result if best_result is not None else result
        if final is None:
            return await agent.run(prompt)
        return final.model_copy(
            update={
                "token_usage": cumulative_usage,
                "reflections_count": self._config.max_reflections,
                "quality_score": best_score if best_score >= 0 else None,
            }
        )

    @staticmethod
    def _build_retry_prompt(original: str, previous_output: str, critique: CriticResult) -> str:
        suggestions = "\n".join(f"- {s}" for s in critique.suggestions) if critique.suggestions else "(no specific suggestions)"
        return (
            f"{original}\n\n"
            f"---\nYour previous response scored {critique.score:.2f} (threshold not met).\n"
            f"Reviewer feedback: {critique.feedback}\n"
            f"Suggestions:\n{suggestions}\n\n"
            "Please produce an improved response addressing the feedback."
        )
