"""File writer tool — creates or replaces files, auto-creating parent directories."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from anycode.constants import DEFAULT_ENCODING
from anycode.security.policy import ToolSecurityError, resolve_tool_path
from anycode.security.redaction import safe_exception_message
from anycode.tools._fsutil import atomic_write_text
from anycode.tools.registry import define_tool
from anycode.types import ToolResult, ToolUseContext


class FileWriteInput(BaseModel):
    path: str = Field(description="Absolute path of the file to write.")
    content: str = Field(description="Complete content to place in the file.")


async def _execute(input: FileWriteInput, context: ToolUseContext) -> ToolResult:
    try:
        target = resolve_tool_path(input.path, context)
    except ToolSecurityError as error:
        return ToolResult(data=safe_exception_message(error), is_error=True)
    existed = target.exists()

    try:
        await asyncio.to_thread(atomic_write_text, target, input.content)
    except Exception as e:
        return ToolResult(data=f'Could not write file "{input.path}": {safe_exception_message(e)}', is_error=True)

    line_count = input.content.count("\n") + (1 if input.content and not input.content.endswith("\n") else 0)
    byte_count = len(input.content.encode(DEFAULT_ENCODING))
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
