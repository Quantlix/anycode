"""LLM-backed critic that evaluates agent output."""

from __future__ import annotations

from anycode.providers.adapter import create_adapter
from anycode.reflection.evaluator import parse_critic_json
from anycode.types import AgentInfo, CriticResult, LLMChatOptions, LLMMessage, TextBlock

DEFAULT_CRITIC_PROMPT = (
    "You are a strict quality reviewer. Evaluate the following output on:\n"
    "1. Accuracy — facts and logic correct\n"
    "2. Completeness — fully addresses the prompt\n"
    "3. Clarity — well-structured and easy to understand\n"
    "4. Quality — production-grade writing or code\n\n"
    "Score from 0.0 (terrible) to 1.0 (excellent).\n"
    "Provide specific, actionable feedback for improvement.\n\n"
    "Respond with JSON ONLY in this exact shape: "
    '{"score": 0.85, "passed": true, "feedback": "...", "suggestions": ["..."]}'
)


def build_critic_prompt(custom: str | None = None) -> str:
    return custom or DEFAULT_CRITIC_PROMPT


class LLMCritic:
    """Critic that calls an LLM to score an output."""

    def __init__(
        self,
        model: str,
        provider: str = "anthropic",
        critic_prompt: str | None = None,
        threshold: float = 0.7,
    ) -> None:
        self._model = model
        self._provider = provider
        self._prompt = build_critic_prompt(critic_prompt)
        self._threshold = threshold
        self._adapter = None  # lazy-init

    async def evaluate(self, output: str, prompt: str, context: AgentInfo) -> CriticResult:
        if self._adapter is None:
            self._adapter = await create_adapter(self._provider)

        body = f"Original prompt:\n{prompt}\n\nAgent ({context.name}) output:\n{output}\n\nEvaluate the output and respond with the required JSON."
        messages = [LLMMessage(role="user", content=[TextBlock(text=body)])]
        options = LLMChatOptions(model=self._model, system_prompt=self._prompt, max_tokens=512)
        response = await self._adapter.chat(messages, options)

        text = "".join(b.text for b in response.content if isinstance(b, TextBlock))
        parsed = parse_critic_json(text, threshold=self._threshold)
        return parsed
