"""Declarative routing rule matching."""

from __future__ import annotations

import re

from anycode.types import ComplexityLevel, RoutingRule, Task


def match_rule(task: Task, complexity: ComplexityLevel, rule: RoutingRule) -> bool:
    """Evaluate whether a routing rule matches a given task and complexity.

    Supports:
    - Complexity conditions: ``complexity == 'trivial'``
    - Keyword conditions: ``'test' in task.title.lower()``
    - Regex conditions: ``re:pattern`` (matches against title + description)
    """
    condition = rule.condition.strip()

    # Complexity equality check
    if condition.startswith("complexity"):
        parts = condition.split("==")
        if len(parts) == 2:
            target = parts[1].strip().strip("'\"")
            return complexity == target
        return False

    # Keyword 'in' check
    if " in " in condition:
        # Parse patterns like: 'keyword' in task.title.lower()
        match = re.match(r"['\"](.+?)['\"].*in\s+task\.(title|description)(?:\.lower\(\))?", condition)
        if match:
            keyword = match.group(1).lower()
            field = match.group(2)
            text = getattr(task, field, "").lower()
            return keyword in text
        return False

    # Regex match (prefix: re:)
    if condition.startswith("re:"):
        pattern = condition[3:].strip()
        text = f"{task.title} {task.description}"
        return bool(re.search(pattern, text, re.IGNORECASE))

    return False


def evaluate_rules(
    task: Task,
    complexity: ComplexityLevel,
    rules: list[RoutingRule],
) -> RoutingRule | None:
    """Evaluate routing rules in priority order (highest first), return the first match."""
    sorted_rules = sorted(rules, key=lambda r: r.priority, reverse=True)
    for rule in sorted_rules:
        if match_rule(task, complexity, rule):
            return rule
    return None
