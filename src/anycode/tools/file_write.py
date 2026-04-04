"""File writer tool — creates or replaces files, auto-creating parent directories."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from anycode.tools.registry import define_tool
from anycode.types import ToolResult, ToolUseContext


class FileWriteInput(BaseModel):
    path: str = Field(description="Absolute path of the file to write.")
    content: str = Field(description="Complete content to place in the file.")


async def _execute(input: FileWriteInput, context: ToolUseContext) -> ToolResult:
    target = Path(input.path)
    existed = target.exists()

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return ToolResult(data=f'Could not create parent directory "{target.parent}": {e}', is_error=True)

    try:
        target.write_text(input.content, encoding="utf-8")
    except Exception as e:
        return ToolResult(data=f'Could not write file "{input.path}": {e}', is_error=True)

    line_count = input.content.count("\n") + (1 if input.content and not input.content.endswith("\n") else 0)
    byte_count = len(input.content.encode("utf-8"))
    action = "Overwrote" if existed else "Created"
    return ToolResult(
        data=f'{action} "{input.path}" ({line_count} line{"s" if line_count != 1 else ""}, {byte_count} bytes).',
        is_error=False,
    )


file_write_tool = define_tool(
    name="file_write",
    description=(
        "Write content to a file. Creates the file and any missing parent directories if it does not exist, or overwrites the file if it does."
    ),
    input_model=FileWriteInput,
    execute=_execute,
)
