"""Immutable provider-neutral sandbox models."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from anycode.contracts.models import ContractError, ContractModel, utc_now
from anycode.identity.context import ExecutionContext

SANDBOX_CONTRACT_VERSION = "1.0"


class SandboxCapabilities(ContractModel):
    provider: str = Field(min_length=1)
    contract_version: Literal["1.0"] = SANDBOX_CONTRACT_VERSION
    isolation: Literal["process", "container", "microvm", "vm", "remote"]
    networking: Literal["none", "allowlist", "unrestricted"]
    persistent_filesystem: bool
    snapshots: bool
    command_streaming: bool
    cancellation: bool
    file_transfer: bool
    evidence: bool
    limitations: tuple[str, ...] = ()


class SandboxSpec(ContractModel):
    run_id: str = Field(min_length=1)
    task_id: str | None = None
    correlation_id: str = Field(min_length=1)
    context: ExecutionContext
    image: str | None = None
    snapshot: str | None = None
    language: str = "python"
    network: Literal["none", "allowlist", "unrestricted"] = "none"
    allowed_domains: tuple[str, ...] = ()
    allowed_cidrs: tuple[str, ...] = ()
    persistent: bool = False
    secret_references: dict[str, str] = Field(default_factory=dict)
    labels: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_spec(self) -> SandboxSpec:
        if self.image and self.snapshot:
            raise ValueError("Select image or snapshot, not both")
        if self.network == "none" and (self.allowed_domains or self.allowed_cidrs):
            raise ValueError("Network allowlists require network='allowlist'")
        if self.network == "allowlist" and not (self.allowed_domains or self.allowed_cidrs):
            raise ValueError("network='allowlist' requires a domain or CIDR")
        if any(not reference.startswith("daytona:") for reference in self.secret_references.values()):
            raise ValueError("Sandbox secrets must be Daytona secret references, never plaintext values")
        return self


class SandboxHandle(ContractModel):
    id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    task_id: str | None = None
    correlation_id: str = Field(min_length=1)
    context: ExecutionContext
    capabilities: SandboxCapabilities
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class SandboxCommand(ContractModel):
    command: str = Field(min_length=1)
    cwd: str | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _reject_environment_credentials(self) -> SandboxCommand:
        sensitive = ("key", "password", "secret", "token", "credential", "authorization")
        if any(any(part in key.casefold() for part in sensitive) for key in self.environment):
            raise ValueError("Command environments cannot contain credential-like fields; mount a provider secret reference")
        return self


class SandboxOutputChunk(ContractModel):
    stream: Literal["stdout", "stderr", "error"]
    data: str
    sequence: int = Field(ge=1)


class SandboxEvidence(ContractModel):
    operation: str = Field(min_length=1)
    digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    recorded_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @classmethod
    def from_bytes(cls, operation: str, payload: bytes, *, metadata: dict[str, JsonValue] | None = None) -> SandboxEvidence:
        return cls(operation=operation, digest=f"sha256:{hashlib.sha256(payload).hexdigest()}", metadata=metadata or {})


class SandboxCreateResult(ContractModel):
    ok: bool
    handle: SandboxHandle | None = None
    error: ContractError | None = None


class SandboxCommandResult(ContractModel):
    ok: bool
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    evidence: SandboxEvidence | None = None
    error: ContractError | None = None


class SandboxFileResult(ContractModel):
    ok: bool
    data: bytes | None = None
    evidence: SandboxEvidence | None = None
    error: ContractError | None = None


class SandboxActionResult(ContractModel):
    ok: bool
    reference: str | None = None
    evidence: SandboxEvidence | None = None
    error: ContractError | None = None


class SandboxHealth(ContractModel):
    status: Literal["healthy", "degraded", "unavailable"]
    message: str = ""
