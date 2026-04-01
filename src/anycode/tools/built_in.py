"""Default tool set — bulk-register every bundled tool."""

from __future__ import annotations

from anycode.tools.bash import bash_tool
from anycode.tools.file_edit import file_edit_tool
from anycode.tools.file_read import file_read_tool
from anycode.tools.file_write import file_write_tool
from anycode.tools.grep import grep_tool
from anycode.tools.registry import ToolRegistry
from anycode.types import ToolDefinition

BUILT_IN_TOOLS: list[ToolDefinition] = [bash_tool, file_read_tool, file_write_tool, file_edit_tool, grep_tool]


def register_built_in_tools(registry: ToolRegistry) -> None:
    for tool in BUILT_IN_TOOLS:
        registry.register(tool)
