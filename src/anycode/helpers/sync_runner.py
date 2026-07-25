"""Blocking wrappers around the async runtime for scripts and notebooks."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncGenerator, Callable, Coroutine, Iterator
from queue import SimpleQueue

_ITEM = "item"
_ERROR = "error"
_DONE = "done"


def _reject_active_loop(sync_call: str, async_call: str) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(f"{sync_call} cannot be called while an event loop is running. Use `{async_call}` instead.")


def run_coroutine_blocking[T](coro: Coroutine[object, object, T], *, sync_call: str, async_call: str) -> T:
    """Run *coro* to completion on a fresh event loop.

    Raises ``RuntimeError`` when a loop is already running, because nesting event loops
    silently deadlocks; the message names the async call to use instead.
    """
    try:
        _reject_active_loop(sync_call, async_call)
    except RuntimeError:
        coro.close()
        raise
    return asyncio.run(coro)


def iterate_async_blocking[T](
    factory: Callable[[], AsyncGenerator[T, None]],
    *,
    sync_call: str,
    async_call: str,
) -> Iterator[T]:
    """Drain an async generator from synchronous code, yielding items as they arrive.

    The generator runs on its own event loop in a worker thread, so items surface
    incrementally rather than only after the stream completes.
    """
    _reject_active_loop(sync_call, async_call)

    queue: SimpleQueue[tuple[str, object]] = SimpleQueue()
    stop = threading.Event()

    async def _pump() -> None:
        try:
            async for item in factory():
                queue.put((_ITEM, item))
                if stop.is_set():
                    break
        except BaseException as error:  # surfaced to the consuming thread verbatim
            queue.put((_ERROR, error))
        finally:
            queue.put((_DONE, None))

    worker = threading.Thread(target=lambda: asyncio.run(_pump()), name="anycode-sync-stream", daemon=True)
    worker.start()

    def _consume() -> Iterator[T]:
        try:
            while True:
                kind, payload = queue.get()
                if kind == _DONE:
                    return
                if kind == _ERROR:
                    raise payload  # type: ignore[misc]
                yield payload  # type: ignore[misc]
        finally:
            stop.set()
            worker.join()

    return _consume()
