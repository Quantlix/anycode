"""Event-driven task queue with dependency resolution and cascading failure."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from anycode.constants import (
    QUEUE_EVENT_ALL_COMPLETE,
    QUEUE_EVENT_TASK_COMPLETE,
    QUEUE_EVENT_TASK_FAILED,
    QUEUE_EVENT_TASK_READY,
)
from anycode.tasks.task import is_task_ready
from anycode.types import Task, TaskStatus


class TaskQueue:
    """Manages tasks with dependency blocking, cascading failure, and event notifications."""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._listeners: dict[str, dict[int, Callable[..., None]]] = {}
        self._next_id = 0

    def add(self, task: Task) -> None:
        resolved = self._determine_initial_status(task)
        self._tasks[resolved.id] = resolved
        if resolved.status == "pending":
            self._emit(QUEUE_EVENT_TASK_READY, resolved)

    def add_batch(self, tasks: list[Task]) -> None:
        for task in tasks:
            self.add(task)

    def update(self, task_id: str, **kwargs: str | TaskStatus | None) -> Task:
        task = self._lookup(task_id)
        updated = task.model_copy(update={**kwargs, "updated_at": datetime.now(UTC)})
        self._tasks[task_id] = updated
        return updated

    def complete(self, task_id: str, result: str | None = None) -> Task:
        completed = self.update(task_id, status="completed", result=result)
        self._emit(QUEUE_EVENT_TASK_COMPLETE, completed)
        self._promote_blocked(task_id)
        if self.is_done():
            self._emit_all_complete()
        return completed

    def fail(self, task_id: str, error: str) -> Task:
        failed = self.update(task_id, status="failed", result=error)
        self._emit(QUEUE_EVENT_TASK_FAILED, failed)
        self._propagate_failure(task_id)
        if self.is_done():
            self._emit_all_complete()
        return failed

    def next(self, assignee: str | None = None) -> Task | None:
        if assignee is None:
            return self.next_available()
        for task in self._tasks.values():
            if task.status == "pending" and task.assignee == assignee:
                return task
        return None

    def next_available(self) -> Task | None:
        fallback: Task | None = None
        for task in self._tasks.values():
            if task.status != "pending":
                continue
            if not task.assignee:
                return task
            if fallback is None:
                fallback = task
        return fallback

    def list(self) -> list[Task]:
        return list(self._tasks.values())

    def get_by_status(self, status: TaskStatus) -> list[Task]:
        return [t for t in self._tasks.values() if t.status == status]

    def is_done(self) -> bool:
        return all(t.status in ("completed", "failed") for t in self._tasks.values())

    def get_progress(self) -> dict[str, int]:
        counts = {"total": len(self._tasks), "completed": 0, "failed": 0, "in_progress": 0, "pending": 0, "blocked": 0}
        for t in self._tasks.values():
            key = t.status if t.status != "in_progress" else "in_progress"
            counts[key] = counts.get(key, 0) + 1
        return counts

    def serialize(self) -> list[dict[str, object]]:
        return [t.model_dump(mode="json") for t in self._tasks.values()]

    def restore(self, data: list[dict[str, object]]) -> None:
        self._tasks.clear()
        for entry in data:
            task = Task(**entry)  # type: ignore[arg-type]
            self._tasks[task.id] = task

    def on(self, event: str, handler: Callable[..., None]) -> Callable[[], None]:
        subs = self._listeners.setdefault(event, {})
        sub_id = self._next_id
        self._next_id += 1
        subs[sub_id] = handler

        def _unsub() -> None:
            subs.pop(sub_id, None)

        return _unsub

    def _determine_initial_status(self, task: Task) -> Task:
        if not task.depends_on:
            return task
        all_current = list(self._tasks.values())
        if is_task_ready(task, all_current):
            return task
        return task.model_copy(update={"status": "blocked", "updated_at": datetime.now(UTC)})

    def _promote_blocked(self, completed_id: str) -> None:
        all_tasks = list(self._tasks.values())
        task_by_id = {t.id: t for t in all_tasks}
        for task in all_tasks:
            if task.status != "blocked":
                continue
            if not task.depends_on or completed_id not in task.depends_on:
                continue
            as_pending = task.model_copy(update={"status": "pending"})
            if is_task_ready(as_pending, all_tasks, task_by_id):
                unblocked = task.model_copy(update={"status": "pending", "updated_at": datetime.now(UTC)})
                self._tasks[task.id] = unblocked
                task_by_id[task.id] = unblocked
                self._emit(QUEUE_EVENT_TASK_READY, unblocked)

    def _propagate_failure(self, failed_id: str) -> None:
        for task in list(self._tasks.values()):
            if task.status not in ("blocked", "pending"):
                continue
            if not task.depends_on or failed_id not in task.depends_on:
                continue
            cascaded = self.update(task.id, status="failed", result=f'Cancelled: prerequisite "{failed_id}" failed.')
            self._emit(QUEUE_EVENT_TASK_FAILED, cascaded)
            self._propagate_failure(task.id)

    def _emit(self, event: str, task: Task) -> None:
        for handler in self._listeners.get(event, {}).values():
            handler(task)

    def _emit_all_complete(self) -> None:
        for handler in self._listeners.get(QUEUE_EVENT_ALL_COMPLETE, {}).values():
            handler()

    def _lookup(self, task_id: str) -> Task:
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f'TaskQueue: no task with id "{task_id}".')
        return task
