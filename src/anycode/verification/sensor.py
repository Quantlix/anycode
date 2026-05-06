"""Verification sensor abstraction.

Sensors are first-class plugins that produce VerificationResults. They run
synchronously or asynchronously at lifecycle phase boundaries and provide
deterministic or inferential evidence about an agent run.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from anycode.types import (
    SensorPhase,
    VerificationResult,
    VerificationSensorConfig,
)

if TYPE_CHECKING:
    from anycode.types import LifecycleEvent, LLMMessage, ToolCallRecord


@dataclass(frozen=True)
class SensorContext:
    """Information passed to a sensor when it runs."""

    phase: SensorPhase
    agent_name: str
    run_id: str
    output: str | None = None
    messages: list[LLMMessage] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    lifecycle_events: list[LifecycleEvent] = field(default_factory=list)
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


SensorFn = Callable[[SensorContext], VerificationResult | Awaitable[VerificationResult]]


@dataclass(frozen=True)
class Sensor:
    """A registered sensor with its config and callable."""

    config: VerificationSensorConfig
    fn: SensorFn

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def phases(self) -> tuple[SensorPhase, ...]:
        return self.config.phases

    async def invoke(self, ctx: SensorContext) -> VerificationResult:
        start = time.monotonic()
        try:
            outcome = self.fn(ctx)
            if inspect.isawaitable(outcome):
                outcome = await outcome
            duration_ms = (time.monotonic() - start) * 1000.0
            if outcome.duration_ms == 0.0:
                outcome = outcome.model_copy(update={"duration_ms": duration_ms})
            return outcome
        except Exception as exc:  # noqa: BLE001 - sensor failures must not crash the pipeline
            duration_ms = (time.monotonic() - start) * 1000.0
            return VerificationResult(
                sensor_name=self.config.name,
                kind=self.config.kind,
                passed=False,
                severity="error",
                message=f"sensor raised {type(exc).__name__}: {exc}",
                duration_ms=duration_ms,
            )
