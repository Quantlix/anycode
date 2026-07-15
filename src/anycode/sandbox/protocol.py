"""Stable sandbox provider boundary."""

from __future__ import annotations

from collections.abc import AsyncIterable
from typing import Protocol, runtime_checkable

from anycode.sandbox.models import (
    SandboxActionResult,
    SandboxCapabilities,
    SandboxCommand,
    SandboxCommandResult,
    SandboxCreateResult,
    SandboxFileResult,
    SandboxHandle,
    SandboxHealth,
    SandboxOutputChunk,
    SandboxSpec,
)


@runtime_checkable
class SandboxProvider(Protocol):
    async def create(self, spec: SandboxSpec) -> SandboxCreateResult: ...

    async def execute(self, handle: SandboxHandle, command: SandboxCommand) -> SandboxCommandResult: ...

    def stream(self, handle: SandboxHandle, command: SandboxCommand) -> AsyncIterable[SandboxOutputChunk]: ...

    async def write_file(self, handle: SandboxHandle, path: str, data: bytes) -> SandboxFileResult: ...

    async def read_file(self, handle: SandboxHandle, path: str) -> SandboxFileResult: ...

    async def cancel(self, handle: SandboxHandle) -> SandboxActionResult: ...

    async def snapshot(self, handle: SandboxHandle, name: str) -> SandboxActionResult: ...

    async def destroy(self, handle: SandboxHandle) -> SandboxActionResult: ...

    async def health(self) -> SandboxHealth: ...

    def capabilities(self) -> SandboxCapabilities: ...
