"""Pre/post turn lifecycle hooks."""

from __future__ import annotations

from anycode.types import AgentInfo, LLMMessage, LLMResponse, TurnHook


class HookRunner:
    """Executes lifecycle hooks in registration order."""

    def __init__(self, hooks: list[TurnHook] | None = None) -> None:
        self._hooks: list[TurnHook] = list(hooks) if hooks else []

    @property
    def hooks(self) -> list[TurnHook]:
        return list(self._hooks)

    def add(self, hook: TurnHook) -> None:
        self._hooks.append(hook)

    async def run_before_turn(self, messages: list[LLMMessage], context: AgentInfo) -> list[LLMMessage]:
        """Run all before_turn hooks in order, each can modify messages."""
        current = messages
        for hook in self._hooks:
            current = await hook.before_turn(current, context)
        return current

    async def run_after_turn(self, response: LLMResponse, context: AgentInfo) -> LLMResponse:
        """Run all after_turn hooks in order, each can modify the response."""
        current = response
        for hook in self._hooks:
            current = await hook.after_turn(current, context)
        return current


class LoggingHook:
    """Example hook that records turn messages and responses for auditing."""

    def __init__(self) -> None:
        self.before_turn_log: list[tuple[int, str]] = []
        self.after_turn_log: list[tuple[str, int]] = []
        self._turn_count = 0

    async def before_turn(self, messages: list[LLMMessage], context: AgentInfo) -> list[LLMMessage]:
        self._turn_count += 1
        self.before_turn_log.append((len(messages), context.name))
        return messages

    async def after_turn(self, response: LLMResponse, context: AgentInfo) -> LLMResponse:
        total_tokens = response.usage.input_tokens + response.usage.output_tokens
        self.after_turn_log.append((context.name, total_tokens))
        return response
