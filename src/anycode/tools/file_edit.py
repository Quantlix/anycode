"""Targeted string replacement tool — edits existing files in place."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from anycode.constants import DEFAULT_ENCODING
from anycode.tools.registry import define_tool
from anycode.types import ToolResult, ToolUseContext


class FileEditInput(BaseModel):
    path: str = Field(description="Absolute path of the target file.")
    old_string: str = Field(description="The exact text to locate and replace.")
    new_string: str = Field(description="The substitution text.")
    replace_all: bool = Field(default=False, description="When true, replace every match.")


async def _execute(input: FileEditInput, context: ToolUseContext) -> ToolResult:
    try:
        original = Path(input.path).read_text(encoding=DEFAULT_ENCODING)
    except Exception as e:
        return ToolResult(data=f'Unable to read "{input.path}": {e}', is_error=True)

    hits = _tally(original, input.old_string)

    if hits == 0:
        return ToolResult(
            data=f'The target string was not found in "{input.path}".\nVerify that old_string matches the file content exactly.',
            is_error=True,
        )

    if hits > 1 and not input.replace_all:
        return ToolResult(
            data=f'old_string appears {hits} times in "{input.path}". Use a more specific string or set replace_all=True.',
            is_error=True,
        )

    updated = original.replace(input.old_string, input.new_string) if input.replace_all else original.replace(input.old_string, input.new_string, 1)

    try:
        Path(input.path).write_text(updated, encoding=DEFAULT_ENCODING)
    except Exception as e:
        return ToolResult(data=f'Unable to write "{input.path}": {e}', is_error=True)

    replaced = hits if input.replace_all else 1
    return ToolResult(data=f'Replaced {replaced} occurrence{"s" if replaced != 1 else ""} in "{input.path}".', is_error=False)


def _tally(source: str, search: str) -> int:
    if not search:
        return 0
    count = 0
    pos = 0
    while True:
        pos = source.find(search, pos)
        if pos == -1:
            break
        count += 1
        pos += len(search)
    return count


file_edit_tool = define_tool(
    name="file_edit",
    description="Modify a file by swapping a specific string with new content. The old_string must match verbatim within the file.",
    input_model=FileEditInput,
    execute=_execute,
)
