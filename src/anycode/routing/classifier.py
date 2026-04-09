"""Heuristic task complexity classifier."""

from __future__ import annotations

from anycode.constants import (
    ROUTING_COMPLEX_MAX_LEN,
    ROUTING_MODERATE_MAX_DEPS,
    ROUTING_MODERATE_MAX_LEN,
    ROUTING_SIMPLE_MAX_DEPS,
    ROUTING_SIMPLE_MAX_LEN,
    ROUTING_TRIVIAL_MAX_DEPS,
    ROUTING_TRIVIAL_MAX_LEN,
)
from anycode.types import ComplexityLevel, Task


def classify_task(task: Task) -> ComplexityLevel:
    """Classify a task by complexity using description length and dependency count.

    Runs in microseconds — zero cost, no LLM call.
    """
    desc_len = len(task.description)
    dep_count = len(task.depends_on or [])

    if desc_len < ROUTING_TRIVIAL_MAX_LEN and dep_count <= ROUTING_TRIVIAL_MAX_DEPS:
        return "trivial"
    elif desc_len < ROUTING_SIMPLE_MAX_LEN and dep_count <= ROUTING_SIMPLE_MAX_DEPS:
        return "simple"
    elif desc_len < ROUTING_MODERATE_MAX_LEN and dep_count <= ROUTING_MODERATE_MAX_DEPS:
        return "moderate"
    elif desc_len < ROUTING_COMPLEX_MAX_LEN:
        return "complex"
    else:
        return "expert"
