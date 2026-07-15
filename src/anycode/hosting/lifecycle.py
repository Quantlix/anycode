"""Framework-neutral liveness, readiness, admission, and drain state."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Literal

from pydantic import Field

from anycode.contracts.models import ContractError, ContractModel, utc_now


class HostLifecycleSnapshot(ContractModel):
    state: Literal["starting", "ready", "draining", "stopped"]
    accepting: bool
    admitted: int = Field(ge=0)
    inflight: int = Field(ge=0)
    max_inflight: int = Field(ge=1)
    started_at: datetime
    updated_at: datetime


class HostAdmissionResult(ContractModel):
    accepted: bool
    duplicate: bool = False
    error: ContractError | None = None


class DrainResult(ContractModel):
    drained: bool
    timed_out: bool = False
    durably_returned: tuple[str, ...] = ()
    remaining: tuple[str, ...] = ()


DurableReturn = Callable[[tuple[str, ...]], Awaitable[None]]


class HostLifecycle:
    """Coordinates graceful rolling shutdown without owning a web framework."""

    def __init__(self, *, max_inflight: int = 100) -> None:
        if max_inflight < 1:
            raise ValueError("max_inflight must be at least 1")
        self._max_inflight = max_inflight
        self._state: Literal["starting", "ready", "draining", "stopped"] = "starting"
        self._admitted: set[str] = set()
        self._inflight: set[str] = set()
        self._started_at = utc_now()
        self._updated_at = self._started_at
        self._condition = asyncio.Condition()

    async def start(self) -> None:
        async with self._condition:
            if self._state == "stopped":
                raise RuntimeError("A stopped host lifecycle cannot be restarted")
            self._state = "ready"
            self._updated_at = utc_now()

    async def admit(self, work_id: str) -> HostAdmissionResult:
        if not work_id:
            raise ValueError("work_id must not be empty")
        async with self._condition:
            if work_id in self._admitted:
                return HostAdmissionResult(accepted=True, duplicate=True)
            if self._state != "ready":
                return HostAdmissionResult(
                    accepted=False,
                    error=ContractError(code="host_draining", message="Host is not accepting new work.", retryable=True),
                )
            if len(self._admitted) >= self._max_inflight:
                return HostAdmissionResult(
                    accepted=False,
                    error=ContractError(code="host_capacity", message="Host has reached its in-flight limit.", retryable=True),
                )
            self._admitted.add(work_id)
            self._updated_at = utc_now()
            return HostAdmissionResult(accepted=True)

    async def begin(self, work_id: str) -> bool:
        async with self._condition:
            if work_id not in self._admitted or self._state == "stopped" or len(self._inflight) >= self._max_inflight:
                return False
            self._inflight.add(work_id)
            self._updated_at = utc_now()
            return True

    async def complete(self, work_id: str) -> None:
        async with self._condition:
            self._inflight.discard(work_id)
            self._admitted.discard(work_id)
            self._updated_at = utc_now()
            self._condition.notify_all()

    async def drain(self, *, timeout_seconds: float, durable_return: DurableReturn | None = None) -> DrainResult:
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must not be negative")
        async with self._condition:
            if self._state == "stopped":
                return DrainResult(drained=True)
            self._state = "draining"
            self._updated_at = utc_now()
            if not self._admitted:
                self._state = "stopped"
                self._updated_at = utc_now()
                return DrainResult(drained=True)
            try:
                await asyncio.wait_for(self._condition.wait_for(lambda: not self._admitted), timeout=timeout_seconds)
            except TimeoutError:
                remaining = tuple(sorted(self._admitted))
            else:
                self._state = "stopped"
                self._updated_at = utc_now()
                return DrainResult(drained=True)
        if durable_return is None:
            return DrainResult(drained=False, timed_out=True, remaining=remaining)
        await durable_return(remaining)
        async with self._condition:
            self._admitted.difference_update(remaining)
            self._inflight.difference_update(remaining)
            self._state = "stopped"
            self._updated_at = utc_now()
            self._condition.notify_all()
        return DrainResult(drained=True, timed_out=True, durably_returned=remaining)

    def snapshot(self) -> HostLifecycleSnapshot:
        return HostLifecycleSnapshot(
            state=self._state,
            accepting=self._state == "ready" and len(self._admitted) < self._max_inflight,
            admitted=len(self._admitted),
            inflight=len(self._inflight),
            max_inflight=self._max_inflight,
            started_at=self._started_at,
            updated_at=self._updated_at,
        )

    def live(self) -> bool:
        return self._state != "stopped"

    def ready(self) -> bool:
        return self.snapshot().accepting
