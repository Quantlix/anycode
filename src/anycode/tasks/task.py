"""Task utilities — creation, readiness checks, topological sort, and cycle detection."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from anycode.types import Task


def create_task(
    *, title: str, description: str, assignee: str | None = None, depends_on: list[str] | None = None
) -> Task:
    now = datetime.now(UTC)
    return Task(
        id=str(uuid4()),
        title=title,
        description=description,
        status="pending",
        assignee=assignee,
        depends_on=list(depends_on) if depends_on else None,
        created_at=now,
        updated_at=now,
    )


def is_task_ready(task: Task, all_tasks: list[Task], task_by_id: dict[str, Task] | None = None) -> bool:
    """True when task is pending and every dependency has completed."""
    if task.status != "pending":
        return False
    if not task.depends_on:
        return True
    lookup = task_by_id or {t.id: t for t in all_tasks}
    stub = Task(id="", title="", description="", created_at=datetime.min, updated_at=datetime.min)
    return all(lookup.get(dep_id, stub).status == "completed" for dep_id in task.depends_on)


def get_task_dependency_order(tasks: list[Task]) -> list[Task]:
    """Topological ordering via Kahn's algorithm."""
    if not tasks:
        return []

    task_by_id = {t.id: t for t in tasks}
    in_degree: dict[str, int] = {t.id: 0 for t in tasks}
    successors: dict[str, list[str]] = {t.id: [] for t in tasks}

    for task in tasks:
        for dep_id in task.depends_on or []:
            if dep_id in task_by_id:
                in_degree[task.id] = in_degree.get(task.id, 0) + 1
                successors.setdefault(dep_id, []).append(task.id)

    frontier = [tid for tid, deg in in_degree.items() if deg == 0]
    ordered: list[Task] = []

    while frontier:
        tid = frontier.pop(0)
        task = task_by_id.get(tid)
        if task:
            ordered.append(task)
        for succ in successors.get(tid, []):
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                frontier.append(succ)

    return ordered


def validate_task_dependencies(tasks: list[Task]) -> tuple[bool, list[str]]:
    """Check dependency graph for missing refs, self-references, and cycles."""
    errors: list[str] = []
    task_by_id = {t.id: t for t in tasks}

    for task in tasks:
        for dep_id in task.depends_on or []:
            if dep_id == task.id:
                errors.append(f'Task "{task.title}" ({task.id}) has a self-dependency.')
                continue
            if dep_id not in task_by_id:
                errors.append(f'Task "{task.title}" ({task.id}) references missing dependency "{dep_id}".')

    # Cycle detection via DFS colouring
    color: dict[str, int] = {t.id: 0 for t in tasks}  # 0=white, 1=gray, 2=black

    def visit(tid: str, path: list[str]) -> None:
        if color.get(tid) == 2:
            return
        if color.get(tid) == 1:
            cycle_start = path.index(tid)
            cycle = path[cycle_start:] + [tid]
            errors.append(f"Dependency cycle found: {' -> '.join(cycle)}")
            return
        color[tid] = 1
        task = task_by_id.get(tid)
        for dep_id in (task.depends_on or []) if task else []:
            if dep_id in task_by_id:
                visit(dep_id, [*path, tid])
        color[tid] = 2

    for task in tasks:
        if color.get(task.id) == 0:
            visit(task.id, [])

    return (len(errors) == 0, errors)
