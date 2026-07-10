"""Fast project file discovery — native fast paths with a deterministic Python fallback.

Discovery order prefers ignore-aware native tools and degrades gracefully:

1. ``git ls-files`` (tracked + untracked-non-ignored) inside a Git repository.
2. ``rg --files`` when ripgrep is available.
3. ``fd`` when installed.
4. A stdlib directory walk using the same ignored-directory policy as ``grep``.

Every backend is capped and reports which tool served the result so callers can
reason about coverage.
"""

from __future__ import annotations

import asyncio
import functools
import os
import re
import shutil
from pathlib import Path

from pydantic import BaseModel, Field

from anycode.constants import DEFAULT_ENCODING, GREP_IGNORED_DIRS, LIST_FILES_CEILING, LIST_FILES_TIMEOUT_S
from anycode.tools.registry import define_tool
from anycode.types import ToolResult, ToolUseContext


class ListFilesInput(BaseModel):
    path: str | None = Field(default=None, description="Directory to list. Defaults to cwd.")
    glob: str | None = Field(default=None, description='Glob filter for filenames (e.g. "*.py").')
    max_results: int = Field(default=LIST_FILES_CEILING, description="Upper bound on returned paths.")


async def _execute(input: ListFilesInput, context: ToolUseContext) -> ToolResult:
    root = Path(input.path or os.getcwd())
    if not root.exists():
        return ToolResult(data=f'Path does not exist: "{root}".', is_error=True)

    backend, paths = await _discover(root)
    matcher = _compile_glob(input.glob)
    filtered = [p for p in paths if not _is_ignored(p) and matcher(Path(p).name)]
    total = len(filtered)
    capped = filtered[: input.max_results]

    if not capped:
        return ToolResult(data=f"No files found (backend: {backend}).", is_error=False)

    body = "\n".join(capped)
    note = f"\n\n({len(capped)} of {total} file{'s' if total != 1 else ''}, backend: {backend}"
    note += ", capped — increase max_results for more)" if total > input.max_results else ")"
    return ToolResult(data=body + note, is_error=False)


async def _discover(root: Path) -> tuple[str, list[str]]:
    """Return (backend_name, relative_paths) from the first available fast path."""
    if _has("git"):
        result = await _run(["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard"], root)
        if result is not None:
            return "git", result
    if _has("rg"):
        result = await _run(["rg", "--files", str(root)], root)
        if result is not None:
            return "ripgrep", result
    if _has("fd"):
        result = await _run(["fd", "--type", "f", "--strip-cwd-prefix", ".", str(root)], root)
        if result is not None:
            return "fd", result
    return "python", await asyncio.to_thread(_python_walk, root)


async def _run(args: list[str], root: Path) -> list[str] | None:
    """Run a discovery command; return relative paths or None if it fails."""
    try:
        proc = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=LIST_FILES_TIMEOUT_S)
    except (OSError, TimeoutError):
        return None
    if proc.returncode != 0:
        return None
    lines = stdout.decode(DEFAULT_ENCODING, errors="replace").splitlines()
    return [_relativize(line, root) for line in lines if line.strip()]


def _relativize(line: str, root: Path) -> str:
    candidate = Path(line)
    if candidate.is_absolute():
        try:
            return str(candidate.relative_to(root))
        except ValueError:
            return line
    return line


def _python_walk(root: Path) -> list[str]:
    if root.is_file():
        return [root.name]
    results: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in GREP_IGNORED_DIRS]
        for name in filenames:
            results.append(os.path.relpath(os.path.join(dirpath, name), root))
    results.sort()
    return results


def _is_ignored(path: str) -> bool:
    return any(part in GREP_IGNORED_DIRS for part in Path(path).parts)


def _compile_glob(glob: str | None):
    if glob is None:
        return lambda _name: True
    pattern = glob.removeprefix("**/")
    regex_src = re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".")
    compiled = re.compile(f"^{regex_src}$", re.IGNORECASE)
    return lambda name: bool(compiled.match(name))


@functools.lru_cache(maxsize=8)
def _has(executable: str) -> bool:
    return shutil.which(executable) is not None


list_files_tool = define_tool(
    name="list_files",
    description=(
        "List project files quickly, preferring git/ripgrep/fd and respecting ignore rules. "
        "Optionally filter by a filename glob. Returns relative paths and the backend used."
    ),
    input_model=ListFilesInput,
    execute=_execute,
)
