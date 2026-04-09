"""Default router implementation using heuristic classifier + declarative rules."""

from __future__ import annotations

from anycode.routing.classifier import classify_task
from anycode.routing.rules import evaluate_rules
from anycode.types import AgentConfig, RouteDecision, RoutingConfig, Task


class DefaultRouter:
    """Routes tasks to optimal models based on complexity classification and rules.

    Uses the heuristic classifier (zero-cost) by default. When no rule matches,
    falls back to the config's default_model / default_provider.
    """

    def __init__(self, config: RoutingConfig) -> None:
        self._config = config

    async def route(self, task: Task, agents: list[AgentConfig]) -> RouteDecision | None:
        """Route a task to the optimal model. Returns None if routing is disabled or no decision."""
        if not self._config.enabled:
            return None

        complexity = classify_task(task)
        original_model = agents[0].model if agents else "unknown"

        # Evaluate declarative rules
        if self._config.rules:
            matched = evaluate_rules(task, complexity, self._config.rules)
            if matched:
                return RouteDecision(
                    task_id=task.id,
                    original_model=original_model,
                    routed_model=matched.target_model,
                    routed_provider=matched.target_provider,
                    complexity=complexity,
                    reason=f"Matched rule: {matched.condition}",
                )

        # Fall back to default
        if self._config.default_model:
            return RouteDecision(
                task_id=task.id,
                original_model=original_model,
                routed_model=self._config.default_model,
                routed_provider=self._config.default_provider,
                complexity=complexity,
                reason=f"Default route (complexity: {complexity})",
            )

        return None
