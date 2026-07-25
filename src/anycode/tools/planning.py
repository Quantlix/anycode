"""Explicit planning for long-horizon work.

An agent writes its plan with ``write_todos`` and gets the rendered checklist back as the
tool result, so the plan re-enters the conversation on every update. That feedback loop is
what keeps a long run on track.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from anycode.types import ToolDefinition, ToolResult, ToolUseContext

TODO_TOOL_NAME = "write_todos"

TodoStatus = Literal["pending", "in_progress", "completed"]

_STATUS_MARKERS: dict[TodoStatus, str] = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}


class TodoItem(BaseModel):
    """One step of an agent's plan."""

    model_config = ConfigDict(frozen=True)

    content: str = Field(min_length=1, description="What this step accomplishes.")
    status: TodoStatus = Field(default="pending", description='One of "pending", "in_progress", or "completed".')


class TodoWriteInput(BaseModel):
    todos: list[TodoItem] = Field(description="The full plan. Always send every step, not just the ones that changed.")


class TodoStore:
    """Holds the plan for one agent."""

    def __init__(self) -> None:
        self._items: tuple[TodoItem, ...] = ()

    @property
    def items(self) -> tuple[TodoItem, ...]:
        return self._items

    def replace(self, items: list[TodoItem]) -> None:
        self._items = tuple(items)

    def clear(self) -> None:
        self._items = ()

    def render(self) -> str:
        if not self._items:
            return "The plan is empty."
        lines = [f"{_STATUS_MARKERS[item.status]} {item.content}" for item in self._items]
        done = sum(1 for item in self._items if item.status == "completed")
        return "\n".join([*lines, "", f"{done}/{len(self._items)} complete."])


def build_todo_tool(store: TodoStore) -> ToolDefinition:
    """Create a ``write_todos`` tool backed by *store*."""

    async def execute(params: TodoWriteInput, _context: ToolUseContext) -> ToolResult:
        in_progress = [item.content for item in params.todos if item.status == "in_progress"]
        if len(in_progress) > 1:
            return ToolResult(
                data=(
                    f"Rejected: {len(in_progress)} steps are in_progress ({', '.join(in_progress)}). "
                    "Exactly one step may be in_progress at a time. Resend the plan with the others "
                    "marked pending or completed."
                ),
                is_error=True,
            )
        store.replace(params.todos)
        return ToolResult(data=store.render())

    return ToolDefinition(
        name=TODO_TOOL_NAME,
        description=(
            "Record or update your plan as a checklist. Call this before starting work and again "
            "whenever a step changes state. Send the complete plan every time. Keep exactly one "
            "step in_progress."
        ),
        input_model=TodoWriteInput,
        execute=execute,
    )
