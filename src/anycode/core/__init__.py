"""Agent runtime, orchestration, pooling, and scheduling.

Exports resolve lazily so importing an Agent does not pull in the orchestrator and
everything it depends on."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from anycode._lazy import build_export_map, lazy_getattr

_EXPORTS = build_export_map(
    {
        "anycode.core.agent": ("Agent",),
        "anycode.core.orchestrator": (
            "AnyCode",
            "TaskSpec",
        ),
        "anycode.core.pool": ("AgentPool",),
        "anycode.core.runner": ("AgentRunner",),
        "anycode.core.scheduler": ("Scheduler",),
    },
)


def __getattr__(name: str) -> Any:
    return lazy_getattr(__name__, name, _EXPORTS, globals())


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    from anycode.core.agent import Agent
    from anycode.core.orchestrator import AnyCode, TaskSpec
    from anycode.core.pool import AgentPool
    from anycode.core.runner import AgentRunner
    from anycode.core.scheduler import Scheduler

__all__ = [
    "Agent",
    "AgentRunner",
    "AgentPool",
    "AnyCode",
    "TaskSpec",
    "Scheduler",
]
