# Demo 34 — list_files Tool
# Execute: uv run python examples/34_list_files.py
#
# Demonstrates the phase-11 built-in `list_files` tool:
#   1. It is registered automatically by register_built_in_tools()
#   2. Fast, ignore-aware discovery that prefers git / ripgrep / fd and falls
#      back to a stdlib walk, reporting which backend served the result
#   3. Optional filename-glob filtering and a result cap
#
# Fully deterministic: runs the tool directly through ToolExecutor, no API key.

import asyncio

from anycode.tools.built_in import BUILT_IN_TOOLS, register_built_in_tools
from anycode.tools.executor import ToolExecutor
from anycode.tools.registry import ToolRegistry
from anycode.types import AgentInfo, ToolUseContext

SEPARATOR = "-" * 60


async def main() -> None:
    print("=== list_files Tool Demo ===\n")

    registry = ToolRegistry()
    register_built_in_tools(registry)
    executor = ToolExecutor(registry)
    ctx = ToolUseContext(agent=AgentInfo(name="lister", role="demo", model="none"))

    # --- Section A: registration ---
    print(SEPARATOR)
    print("Section A: registration\n")

    names = [t.name for t in BUILT_IN_TOOLS]
    print(f"  built-in tools: {names}")
    print(f"  'list_files' present: {registry.has('list_files')}")

    # --- Section B: list a directory with a glob filter ---
    print(f"\n{SEPARATOR}")
    print("Section B: examples/*.py\n")

    result = await executor.execute("list_files", {"path": "examples", "glob": "*.py"}, ctx)
    print(f"  is_error: {result.is_error}")
    for line in result.data.splitlines()[:8]:
        print(f"    {line}")
    print("    ...")
    # The trailing note reports count + which backend served the result.
    print(f"    {result.data.splitlines()[-1]}")

    # --- Section C: repo root with a cap ---
    print(f"\n{SEPARATOR}")
    print("Section C: repo root, capped at 5\n")

    result = await executor.execute("list_files", {"path": ".", "max_results": 5}, ctx)
    for line in result.data.splitlines():
        print(f"    {line}")

    # --- Section D: missing path is a graceful tool error ---
    print(f"\n{SEPARATOR}")
    print("Section D: nonexistent path\n")

    result = await executor.execute("list_files", {"path": "does/not/exist"}, ctx)
    print(f"  is_error: {result.is_error}")
    print(f"  message:  {result.data}")

    print(f"\n{SEPARATOR}\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
