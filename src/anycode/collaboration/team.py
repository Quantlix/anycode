"""Manages a named group of agents with messaging, task queue, and shared memory."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from anycode.collaboration.message_bus import MessageBus
from anycode.collaboration.shared_mem import SharedMemory
from anycode.tasks.queue import TaskQueue
from anycode.tasks.task import create_task
from anycode.types import AgentConfig, MemoryStore, Message, OrchestratorEvent, Task, TaskStatus, TeamConfig


class _EventBus:
    def __init__(self) -> None:
        self._listeners: dict[str, dict[int, Callable[[Any], None]]] = {}
        self._next_id = 0

    def on(self, event: str, handler: Callable[[Any], None]) -> Callable[[], None]:
        subs = self._listeners.setdefault(event, {})
        sub_id = self._next_id
        self._next_id += 1
        subs[sub_id] = handler
        return lambda: subs.pop(sub_id, None)

    def emit(self, event: str, data: Any) -> None:
        for handler in self._listeners.get(event, {}).values():
            handler(data)


class Team:
    """Agent team with messaging, task queue, and shared memory."""

    def __init__(self, config: TeamConfig) -> None:
        self.name = config.name
        self.config = config
        self._agent_map: dict[str, AgentConfig] = {a.name: a for a in config.agents}
        self._bus = MessageBus()
        self._queue = TaskQueue()
        self._memory = SharedMemory() if config.shared_memory else None
        self._events = _EventBus()

        # Relay queue events
        self._queue.on("task:ready", lambda t: self._events.emit("task:ready", OrchestratorEvent(type="task_start", task=t.id, data=t)))
        self._queue.on("task:complete", lambda t: self._events.emit("task:complete", OrchestratorEvent(type="task_complete", task=t.id, data=t)))
        self._queue.on("task:failed", lambda t: self._events.emit("task:failed", OrchestratorEvent(type="error", task=t.id, data=t)))
        self._queue.on("all:complete", lambda: self._events.emit("all:complete", None))

    def get_agents(self) -> list[AgentConfig]:
        return list(self._agent_map.values())

    def get_agent(self, name: str) -> AgentConfig | None:
        return self._agent_map.get(name)

    def send_message(self, from_agent: str, to_agent: str, content: str) -> None:
        msg = self._bus.send(from_agent, to_agent, content)
        self._events.emit("message", OrchestratorEvent(type="message", agent=from_agent, data=msg))

    def get_messages(self, agent_name: str) -> list[Message]:
        return self._bus.get_all(agent_name)

    def broadcast(self, from_agent: str, content: str) -> None:
        msg = self._bus.broadcast(from_agent, content)
        self._events.emit("broadcast", OrchestratorEvent(type="message", agent=from_agent, data=msg))

    def add_task(
        self,
        *,
        title: str,
        description: str,
        status: TaskStatus = "pending",
        assignee: str | None = None,
        depends_on: list[str] | None = None,
        result: str | None = None,
    ) -> Task:
        created = create_task(title=title, description=description, assignee=assignee, depends_on=depends_on)
        final = created.model_copy(update={"status": status, "result": result}) if status != "pending" else created
        self._queue.add(final)
        return final

    def get_tasks(self) -> list[Task]:
        return self._queue.list()

    def get_tasks_by_assignee(self, agent_name: str) -> list[Task]:
        return [t for t in self._queue.list() if t.assignee == agent_name]

    def update_task(self, task_id: str, **kwargs: Any) -> Task:
        return self._queue.update(task_id, **kwargs)

    def get_next_task(self, agent_name: str) -> Task | None:
        return self._queue.next(agent_name) or self._queue.next_available()

    def get_shared_memory(self) -> MemoryStore | None:
        return self._memory.get_store() if self._memory else None

    def get_shared_memory_instance(self) -> SharedMemory | None:
        return self._memory

    def on(self, event: str, handler: Callable[[Any], None]) -> Callable[[], None]:
        return self._events.on(event, handler)

    def emit(self, event: str, data: Any) -> None:
        self._events.emit(event, data)
