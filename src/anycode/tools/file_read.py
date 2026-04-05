"""File reader tool — retrieves file contents with optional range windowing."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from anycode.constants import DEFAULT_ENCODING
from anycode.tools.registry import define_tool
from anycode.types import ToolResult, ToolUseContext


class FileReadInput(BaseModel):
    path: str = Field(description="Absolute path of the file to read.")
    offset: int | None = Field(default=None, description="1-based starting line number.")
    limit: int | None = Field(default=None, description="Maximum lines to return.")


async def _execute(input: FileReadInput, context: ToolUseContext) -> ToolResult:
    try:
        raw = Path(input.path).read_text(encoding=DEFAULT_ENCODING)
    except Exception as e:
        return ToolResult(data=f'Unable to read "{input.path}": {e}', is_error=True)

    lines = raw.split("\n")
    if lines and lines[-1] == "":
        lines.pop()

    total = len(lines)
    start = max(0, (input.offset or 1) - 1)

    if start >= total > 0:
        return ToolResult(
            data=f'File "{input.path}" contains {total} line{"s" if total != 1 else ""} but offset {input.offset} exceeds the range.',
            is_error=True,
        )

    end = min(start + input.limit, total) if input.limit else total
    numbered = "\n".join(f"{start + i + 1}\t{line}" for i, line in enumerate(lines[start:end]))
    meta = f"\n\n(showing lines {start + 1}\u2013{end} of {total})" if end < total else ""
    return ToolResult(data=numbered + meta, is_error=False)


file_read_tool = define_tool(
    name="file_read",
    description=(
        "Retrieve the contents of a file from disk. Output includes line numbers in the format 'N\\t<line>'."
        " Specify offset and limit to page through large files."
    ),
    input_model=FileReadInput,
    execute=_execute,
)
