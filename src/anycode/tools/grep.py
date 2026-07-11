"""Regex search tool — ripgrep fast-path with Python fallback."""

from __future__ import annotations

import asyncio
import functools
import os
import re
import shutil
from pathlib import Path

from pydantic import BaseModel, Field

from anycode.constants import DEFAULT_ENCODING, GREP_IGNORED_DIRS, GREP_MATCH_CEILING
from anycode.security.policy import ToolSecurityError, resolve_tool_path
from anycode.security.redaction import redact_text, safe_exception_message
from anycode.tools.registry import define_tool
from anycode.types import ToolResult, ToolUseContext


class GrepInput(BaseModel):
    pattern: str = Field(description="Regex pattern to search for.")
    path: str | None = Field(default=None, description="Directory or file to search. Defaults to cwd.")
    glob: str | None = Field(default=None, description='Glob filter for filenames (e.g. "*.py").')
    max_results: int = Field(default=GREP_MATCH_CEILING, description="Upper bound on matching lines.")


async def _execute(input: GrepInput, context: ToolUseContext) -> ToolResult:
    try:
        search_path = str(resolve_tool_path(input.path, context))
    except ToolSecurityError as error:
        return ToolResult(data=safe_exception_message(error), is_error=True)
    cap = input.max_results

    if _has_ripgrep():
        return await _ripgrep_search(input.pattern, search_path, glob=input.glob, max_results=cap)

    try:
        regex = re.compile(input.pattern)
    except re.error:
        return ToolResult(data=f'Invalid regex pattern: "{input.pattern}"', is_error=True)
    return await asyncio.to_thread(_python_search, regex, search_path, glob=input.glob, max_results=cap)


async def _ripgrep_search(pattern: str, search_path: str, *, glob: str | None, max_results: int) -> ToolResult:
    args = ["rg", "--line-number", "--no-heading", "--color=never", f"--max-count={max_results}"]
    if glob:
        args.extend(["--glob", glob])
    args.extend(["--", pattern, search_path])

    try:
        proc = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await proc.communicate()
        output = stdout.decode(DEFAULT_ENCODING, errors="replace").strip()
        if proc.returncode not in (0, 1):
            error_output = redact_text(stderr.decode(DEFAULT_ENCODING, errors="replace").strip())
            return ToolResult(data=f"ripgrep exited with code {proc.returncode}: {error_output}", is_error=True)
        return ToolResult(data=output or "No matches.", is_error=False)
    except Exception as e:
        return ToolResult(data=f"ripgrep error: {safe_exception_message(e)}", is_error=True)


def _python_search(regex: re.Pattern[str], search_path: str, *, glob: str | None, max_results: int) -> ToolResult:
    target = Path(search_path)
    try:
        files = [target] if target.is_file() else list(_gather_files(target, glob))
    except Exception as e:
        return ToolResult(data=f'Cannot access "{search_path}": {safe_exception_message(e)}', is_error=True)

    hits: list[str] = []
    for file in files:
        if len(hits) >= max_results:
            break
        try:
            content = file.read_text(encoding=DEFAULT_ENCODING, errors="replace")
        except Exception:
            continue
        for i, line in enumerate(content.split("\n")):
            if len(hits) >= max_results:
                break
            if regex.search(line):
                hits.append(f"{_display_path(file)}:{i + 1}:{line}")

    if not hits:
        return ToolResult(data="No matches.", is_error=False)
    note = f"\n\n(results capped at {max_results} — increase max_results for more)" if len(hits) >= max_results else ""
    return ToolResult(data="\n".join(hits) + note, is_error=False)


def _display_path(file: Path) -> str:
    if not file.is_absolute():
        return str(file)
    try:
        return os.path.relpath(file)
    except ValueError:
        return str(file)


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
            if entry.name not in GREP_IGNORED_DIRS:
                _traverse(entry, glob_pattern, results)
        elif entry.is_file():
            if glob_pattern is None or _glob_match(entry.name, glob_pattern):
                results.append(entry)


def _glob_match(filename: str, glob_pattern: str) -> bool:
    pattern = glob_pattern.removeprefix("**/")
    regex_src = re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".")
    return bool(re.match(f"^{regex_src}$", filename, re.IGNORECASE))


@functools.lru_cache(maxsize=1)
def _has_ripgrep() -> bool:
    return shutil.which("rg") is not None


grep_tool = define_tool(
    name="grep",
    description="Search files for a regular-expression pattern. Returns matching lines with file paths and line numbers.",
    input_model=GrepInput,
    execute=_execute,
)
