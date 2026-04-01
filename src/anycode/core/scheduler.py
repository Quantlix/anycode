"""Scheduling strategies for distributing tasks across agents."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from anycode.types import AgentConfig, SchedulingStrategy, Task

if TYPE_CHECKING:
    from anycode.tasks.queue import TaskQueue

NOISE_WORDS = {
    "the", "and", "for", "that", "this", "with", "are", "from", "have",
    "will", "your", "you", "can", "all", "each", "when", "then", "they",
    "them", "about", "into", "more", "also", "should", "must", "been",
    "some", "what", "than",
}


def _tokenize(text: str) -> list[str]:
    words = set(re.split(r"\W+", text.lower()))
    return [w for w in words if len(w) > 3 and w not in NOISE_WORDS]


def _relevance_score(text: str, terms: list[str]) -> int:
    lower = text.lower()
    return sum(1 for t in terms if t in lower)


def _downstream_block_count(task_id: str, children: dict[str, list[str]], task_ids: set[str]) -> int:
    seen: set[str] = set()
    frontier = [task_id]
    while frontier:
        current = frontier.pop(0)
        for child in children.get(current, []):
            if child not in seen and child in task_ids:
                seen.add(child)
                frontier.append(child)
    return len(seen)


def _build_children_map(tasks: list[Task]) -> tuple[dict[str, list[str]], set[str]]:
    task_ids = {t.id for t in tasks}
    children: dict[str, list[str]] = {}
    for t in tasks:
        for dep_id in t.depends_on or []:
            children.setdefault(dep_id, []).append(t.id)
    return children, task_ids


class Scheduler:
    """Distributes tasks across agents using one of four strategies."""

    def __init__(self, strategy: SchedulingStrategy = "dependency-first") -> None:
        self._strategy = strategy
        self._rr_cursor = 0

    def schedule(self, tasks: list[Task], agents: list[AgentConfig]) -> dict[str, str]:
        if not agents:
            return {}
        pending = [t for t in tasks if t.status == "pending" and not t.assignee]

        match self._strategy:
            case "round-robin":
                return self._round_robin(pending, agents)
            case "least-busy":
                return self._least_busy(pending, agents, tasks)
            case "capability-match":
                return self._capability_match(pending, agents)
            case "dependency-first":
                return self._dependency_first(pending, agents, tasks)

    def auto_assign(self, queue: TaskQueue, agents: list[AgentConfig]) -> None:
        snapshot = queue.list()
        assignments = self.schedule(snapshot, agents)
        for task_id, agent_name in assignments.items():
            try:
                queue.update(task_id, assignee=agent_name)
            except Exception:
                pass

    def _round_robin(self, pending: list[Task], agents: list[AgentConfig]) -> dict[str, str]:
        result: dict[str, str] = {}
        for task in pending:
            result[task.id] = agents[self._rr_cursor % len(agents)].name
            self._rr_cursor = (self._rr_cursor + 1) % len(agents)
        return result

    def _least_busy(self, pending: list[Task], agents: list[AgentConfig], all_tasks: list[Task]) -> dict[str, str]:
        workload = {a.name: 0 for a in agents}
        for t in all_tasks:
            if t.status == "in_progress" and t.assignee:
                workload[t.assignee] = workload.get(t.assignee, 0) + 1

        result: dict[str, str] = {}
        for task in pending:
            pick = min(agents, key=lambda a: workload.get(a.name, 0))
            result[task.id] = pick.name
            workload[pick.name] = workload.get(pick.name, 0) + 1
        return result

    def _capability_match(self, pending: list[Task], agents: list[AgentConfig]) -> dict[str, str]:
        agent_terms = {a.name: _tokenize(f"{a.name} {a.system_prompt or ''} {a.model}") for a in agents}
        result: dict[str, str] = {}
        for task in pending:
            task_text = f"{task.title} {task.description}"
            task_terms = _tokenize(task_text)
            best = max(
                agents,
                key=lambda a: _relevance_score(f"{a.name} {a.system_prompt or ''}", task_terms)
                + _relevance_score(task_text, agent_terms.get(a.name, [])),
            )
            result[task.id] = best.name
        return result

    def _dependency_first(self, pending: list[Task], agents: list[AgentConfig], all_tasks: list[Task]) -> dict[str, str]:
        children, task_ids = _build_children_map(all_tasks)
        ranked = sorted(pending, key=lambda t: _downstream_block_count(t.id, children, task_ids), reverse=True)
        result: dict[str, str] = {}
        cursor = self._rr_cursor
        for task in ranked:
            result[task.id] = agents[cursor % len(agents)].name
            cursor = (cursor + 1) % len(agents)
        self._rr_cursor = cursor
        return result
