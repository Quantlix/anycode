"""Bounded concurrency gate for parallel task limiting."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


class Semaphore:
    """Async semaphore with active/pending tracking."""

    def __init__(self, max_concurrency: int) -> None:
        if max_concurrency < 1:
            raise ValueError(f"Concurrency limit must be >= 1, received {max_concurrency}")
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._active = 0
        self._pending = 0

    async def acquire(self) -> None:
        self._pending += 1
        await self._semaphore.acquire()
        self._pending -= 1
        self._active += 1

    def release(self) -> None:
        self._active -= 1
        self._semaphore.release()

    async def run(self, fn: Callable[[], Awaitable[T]]) -> T:
        await self.acquire()
        try:
            return await fn()
        finally:
            self.release()

    @property
    def active(self) -> int:
        return self._active

    @property
    def pending(self) -> int:
        return self._pending
