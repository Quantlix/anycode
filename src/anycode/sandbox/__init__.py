"""Provider-neutral sandbox integrations."""

from anycode.sandbox.companion import CompanionSandboxAdapter, CompanionSandboxClient
from anycode.sandbox.daytona import DaytonaSandboxProvider
from anycode.sandbox.models import (
    SANDBOX_CONTRACT_VERSION,
    SandboxActionResult,
    SandboxCapabilities,
    SandboxCommand,
    SandboxCommandResult,
    SandboxCreateResult,
    SandboxEvidence,
    SandboxFileResult,
    SandboxHandle,
    SandboxHealth,
    SandboxOutputChunk,
    SandboxSpec,
)
from anycode.sandbox.policy import PolicySandboxProvider
from anycode.sandbox.protocol import SandboxProvider

__all__ = [
    "SANDBOX_CONTRACT_VERSION",
    "CompanionSandboxAdapter",
    "CompanionSandboxClient",
    "DaytonaSandboxProvider",
    "PolicySandboxProvider",
    "SandboxActionResult",
    "SandboxCapabilities",
    "SandboxCommand",
    "SandboxCommandResult",
    "SandboxCreateResult",
    "SandboxEvidence",
    "SandboxFileResult",
    "SandboxHandle",
    "SandboxHealth",
    "SandboxOutputChunk",
    "SandboxProvider",
    "SandboxSpec",
]
