"""Regex search tool — ripgrep fast-path with Python fallback."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
from pathlib import Path

from pydantic import BaseModel, Field

from anycode.tools.registry import define_tool
from anycode.types import ToolResult, ToolUseContext

MATCH_CEILING = 100
IGNORED_DIRS = {".git", ".svn", ".hg", "node_modules", ".next", "dist", "build", "__pycache__", ".venv"}


class GrepInput(BaseModel):
    pattern: str = Field(description="Regex pattern to search for.")
    path: str | None = Field(default=None, description="Directory or file to search. Defaults to cwd.")
    glob: str | None = Field(default=None, description='Glob filter for filenames (e.g. "*.py").')
    max_results: int = Field(default=MATCH_CEILING, description="Upper bound on matching lines.")


async def _execute(input: GrepInput, context: ToolUseContext) -> ToolResult:
    search_path = input.path or os.getcwd()
    cap = input.max_results

    try:
        regex = re.compile(input.pattern)
    except re.error:
        return ToolResult(data=f'Invalid regex pattern: "{input.pattern}"', is_error=True)

    if _has_ripgrep():
        return await _ripgrep_search(input.pattern, search_path, glob=input.glob, max_results=cap)
    return await _python_search(regex, search_path, glob=input.glob, max_results=cap)


async def _ripgrep_search(
    pattern: str, search_path: str, *, glob: str | None, max_results: int
) -> ToolResult:
    args = ["rg", "--line-number", "--no-heading", "--color=never", f"--max-count={max_results}"]
    if glob:
        args.extend(["--glob", glob])
    args.extend(["--", pattern, search_path])

    try:
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        output = stdout.decode("utf-8", errors="replace").strip()
        if proc.returncode not in (0, 1):
            return ToolResult(data=f"ripgrep exited with code {proc.returncode}: {stderr.decode().strip()}", is_error=True)
        return ToolResult(data=output or "No matches.", is_error=False)
    except Exception as e:
        return ToolResult(data=f"ripgrep error: {e}", is_error=True)


async def _python_search(
    regex: re.Pattern[str], search_path: str, *, glob: str | None, max_results: int
) -> ToolResult:
    target = Path(search_path)
    try:
        files = [target] if target.is_file() else list(_gather_files(target, glob))
    except Exception as e:
        return ToolResult(data=f'Cannot access "{search_path}": {e}', is_error=True)

    hits: list[str] = []
    for file in files:
        if len(hits) >= max_results:
            break
        try:
            content = file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for i, line in enumerate(content.split("\n")):
            if len(hits) >= max_results:
                break
            if regex.search(line):
                rel = os.path.relpath(file) if file.is_absolute() else str(file)
                hits.append(f"{rel}:{i + 1}:{line}")

    if not hits:
        return ToolResult(data="No matches.", is_error=False)
    note = f"\n\n(results capped at {max_results} — increase max_results for more)" if len(hits) >= max_results else ""
    return ToolResult(data="\n".join(hits) + note, is_error=False)


def _gather_files(directory: Path, glob_pattern: str | None) -> list[Path]:
    results: list[Path] = []
    _traverse(directory, glob_pattern, results)
    return results


def _traverse(directory: Path, glob_pattern: str | None, results: list[Path]) -> None:
    try:
        entries = list(directory.iterdir())
    except PermissionError:
        return
    for entry in entries:
        if entry.is_dir():
            if entry.name not in IGNORED_DIRS:
                _traverse(entry, glob_pattern, results)
        elif entry.is_file():
            if glob_pattern is None or _glob_match(entry.name, glob_pattern):
                results.append(entry)


def _glob_match(filename: str, glob_pattern: str) -> bool:
    pattern = glob_pattern.removeprefix("**/")
    regex_src = re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".")
    return bool(re.match(f"^{regex_src}$", filename, re.IGNORECASE))


def _has_ripgrep() -> bool:
    return shutil.which("rg") is not None


grep_tool = define_tool(
    name="grep",
    description="Search files for a regular-expression pattern. Returns matching lines with file paths and line numbers.",
    input_model=GrepInput,
    execute=_execute,
)
