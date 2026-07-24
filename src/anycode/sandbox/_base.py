"""Shared scaffolding for remote sandbox provider adapters."""

from __future__ import annotations

import inspect
import shlex
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from typing import Any

from anycode.contracts.models import ContractError
from anycode.sandbox.models import SandboxCommand, SandboxCommandResult, SandboxOutputChunk, SandboxSpec


async def call_maybe_async(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Invoke an SDK callable that may be sync, async, or expose an ``.aio`` twin."""
    aio = getattr(fn, "aio", None)
    target = aio if callable(aio) else fn
    result = target(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


async def read_streamish(value: Any) -> str:
    """Normalize SDK output shapes (str, bytes, callable, or object with read())."""
    if callable(value):
        value = await call_maybe_async(value)
    reader = getattr(value, "read", None)
    if callable(reader):
        value = await call_maybe_async(reader)
    if value is None:
        return ""
    if isinstance(value, bytes | bytearray):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def to_bytes(value: Any) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes | bytearray):
        return bytes(value)
    return str(value).encode("utf-8")


def network_policy_error(provider: str, spec: SandboxSpec, *, supported: tuple[str, ...]) -> ContractError | None:
    """Fail closed when a backend cannot enforce the requested network mode."""
    if spec.network in supported:
        return None
    return ContractError(
        code="sandbox_network_policy_unsupported",
        message=f"{provider} cannot enforce network='{spec.network}'; supported modes: {', '.join(supported)}",
    )


def secrets_unsupported_error(provider: str, spec: SandboxSpec) -> ContractError | None:
    if not spec.secret_references:
        return None
    return ContractError(
        code="sandbox_secrets_unsupported",
        message=f"{provider} has no managed secret store; configure credentials on the provider side instead",
    )


def snapshot_restore_unsupported_error(provider: str, spec: SandboxSpec) -> ContractError | None:
    if not spec.snapshot:
        return None
    return ContractError(
        code="sandbox_snapshot_unsupported",
        message=f"{provider} cannot create a sandbox from a snapshot reference",
    )


def first_guard_error(*guards: ContractError | None) -> ContractError | None:
    for guard in guards:
        if guard is not None:
            return guard
    return None


def shell_command(command: SandboxCommand) -> str:
    """Fold cwd and environment into one POSIX shell string for id-based exec APIs."""
    parts: list[str] = []
    for key, value in command.environment.items():
        parts.append(f"export {key}={shlex.quote(value)}")
    if command.cwd:
        parts.append(f"cd {shlex.quote(command.cwd)}")
    parts.append(command.command)
    return " && ".join(parts)


def buffered_stream(
    execute: Callable[..., Awaitable[SandboxCommandResult]],
    handle: Any,
    command: SandboxCommand,
) -> AsyncIterable[SandboxOutputChunk]:
    """Satisfy the streaming contract from a buffered execute() implementation."""

    async def generate() -> AsyncIterator[SandboxOutputChunk]:
        result = await execute(handle, command)
        if not result.ok and result.error is not None:
            yield SandboxOutputChunk(stream="error", data=result.error.message, sequence=1)
            return
        sequence = 0
        for stream_name, data in (("stdout", result.stdout), ("stderr", result.stderr)):
            if data:
                sequence += 1
                yield SandboxOutputChunk(stream=stream_name, data=data, sequence=sequence)  # type: ignore[arg-type]

    return generate()
