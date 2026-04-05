"""Shell execution tool — runs commands with configurable timeout."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from anycode.constants import BASH_TIMEOUT_LIMIT_S, DEFAULT_ENCODING, EXIT_CODE_NOT_FOUND, EXIT_CODE_TIMEOUT
from anycode.tools.registry import define_tool
from anycode.types import ToolResult, ToolUseContext


class BashInput(BaseModel):
    command: str = Field(description="The shell command to run.")
    timeout: float | None = Field(default=None, description=f"Time limit in seconds. Defaults to {BASH_TIMEOUT_LIMIT_S}s.")
    cwd: str | None = Field(default=None, description="Directory to execute the command in.")


async def _execute(input: BashInput, context: ToolUseContext) -> ToolResult:
    limit = input.timeout or BASH_TIMEOUT_LIMIT_S
    stdout, stderr, exit_code = await _exec_command(input.command, cwd=input.cwd, timeout=limit)
    return ToolResult(data=_compose_result(stdout, stderr, exit_code), is_error=exit_code != 0)


async def _exec_command(command: str, cwd: str | None, timeout: float) -> tuple[str, str, int]:
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return "", "Process timed out", EXIT_CODE_TIMEOUT
        return (
            stdout_bytes.decode(DEFAULT_ENCODING, errors="replace"),
            stderr_bytes.decode(DEFAULT_ENCODING, errors="replace"),
            proc.returncode or 0,
        )
    except Exception as e:
        return "", str(e), EXIT_CODE_NOT_FOUND


def _compose_result(stdout: str, stderr: str, exit_code: int) -> str:
    parts: list[str] = []
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f"-- stderr --\n{stderr}" if stdout else stderr)
    if not parts:
        return "(completed silently — no output)" if exit_code == 0 else f"(exited with code {exit_code} — no output produced)"
    if exit_code != 0:
        parts.append(f"\n(exit code {exit_code})")
    return "\n".join(parts)


bash_tool = define_tool(
    name="bash",
    description=(
        "Run a shell command and capture its stdout and stderr. "
        "Useful for file-system tasks, script execution, package management, "
        "or anything requiring a shell session."
    ),
    input_model=BashInput,
    execute=_execute,
)
