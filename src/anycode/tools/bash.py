"""Shell execution tool — runs commands with timeout, output caps, and process-group cleanup."""

from __future__ import annotations

import asyncio
import os
import signal
import sys

from pydantic import BaseModel, Field

from anycode.constants import (
    BASH_MAX_OUTPUT_BYTES,
    BASH_READ_CHUNK_BYTES,
    BASH_TIMEOUT_LIMIT_S,
    DEFAULT_ENCODING,
    EXIT_CODE_NOT_FOUND,
    EXIT_CODE_TIMEOUT,
)
from anycode.security.policy import ToolSecurityError, build_subprocess_environment, resolve_tool_path, validate_shell_command
from anycode.security.redaction import safe_exception_message
from anycode.tools.registry import define_tool
from anycode.types import ToolResult, ToolUseContext

_IS_WINDOWS = sys.platform == "win32"


class BashInput(BaseModel):
    command: str = Field(description="The shell command to run.")
    timeout: float | None = Field(default=None, description=f"Time limit in seconds. Defaults to {BASH_TIMEOUT_LIMIT_S}s.")
    cwd: str | None = Field(default=None, description="Directory to execute the command in.")
    max_output_bytes: int | None = Field(default=None, description=f"Cap on captured stdout/stderr bytes. Defaults to {BASH_MAX_OUTPUT_BYTES}.")


async def _execute(input: BashInput, context: ToolUseContext) -> ToolResult:
    limit = input.timeout or BASH_TIMEOUT_LIMIT_S
    cap = input.max_output_bytes if input.max_output_bytes is not None else BASH_MAX_OUTPUT_BYTES
    try:
        validate_shell_command(input.command, context)
        cwd = str(resolve_tool_path(input.cwd, context)) if input.cwd is not None or context.security_policy is not None else None
        env = build_subprocess_environment(context)
    except ToolSecurityError as error:
        return ToolResult(data=safe_exception_message(error), is_error=True)
    stdout, stderr, exit_code = await _exec_command(input.command, cwd=cwd, timeout=limit, cap=cap, env=env)
    return ToolResult(data=_compose_result(stdout, stderr, exit_code), is_error=exit_code != 0)


def _spawn_kwargs() -> dict[str, object]:
    """Isolate the child in its own process group so the whole tree can be killed."""
    if _IS_WINDOWS:
        import subprocess

        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


async def _read_capped(stream: asyncio.StreamReader | None, cap: int) -> tuple[bytes, int]:
    """Read a stream, storing at most ``cap`` bytes but draining the rest.

    Returns the captured (possibly truncated) bytes and the total byte count so
    the caller can report exactly how much was dropped.
    """
    if stream is None:
        return b"", 0
    stored = bytearray()
    total = 0
    while True:
        chunk = await stream.read(BASH_READ_CHUNK_BYTES)
        if not chunk:
            break
        if total < cap:
            stored.extend(chunk[: cap - total])
        total += len(chunk)
    return bytes(stored), total


async def _terminate(proc: asyncio.subprocess.Process) -> None:
    """Kill the process and every child in its group, then reap it."""
    if proc.returncode is not None:
        return
    try:
        if _IS_WINDOWS:
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/F",
                "/T",
                "/PID",
                str(proc.pid),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await killer.wait()
        else:
            # POSIX-only APIs; unreachable on Windows where pyright type-checks.
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)  # type: ignore[attr-defined]
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.kill()
    except ProcessLookupError:
        pass
    await proc.wait()


async def _exec_command(command: str, cwd: str | None, timeout: float, cap: int, env: dict[str, str] | None = None) -> tuple[str, str, int]:
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
            **_spawn_kwargs(),  # type: ignore[arg-type]
        )
    except Exception as e:
        return "", safe_exception_message(e), EXIT_CODE_NOT_FOUND

    async def _collect() -> tuple[bytes, int, bytes, int]:
        out, out_total = await _read_capped(proc.stdout, cap)
        err, err_total = await _read_capped(proc.stderr, cap)
        await proc.wait()
        return out, out_total, err, err_total

    try:
        out_bytes, out_total, err_bytes, err_total = await asyncio.wait_for(_collect(), timeout=timeout)
    except TimeoutError:
        await _terminate(proc)
        return "", f"Process timed out after {timeout:g}s (process group terminated)", EXIT_CODE_TIMEOUT

    return (
        _decode_capped(out_bytes, out_total, cap),
        _decode_capped(err_bytes, err_total, cap),
        proc.returncode or 0,
    )


def _decode_capped(data: bytes, total: int, cap: int) -> str:
    text = data.decode(DEFAULT_ENCODING, errors="replace")
    if total > cap:
        dropped = total - cap
        text += f"\n[output truncated: showing first {cap} of {total} bytes; {dropped} dropped]"
    return text


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
