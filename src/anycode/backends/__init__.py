"""Pluggable durability backends."""

from anycode.backends.dapr import DaprDurabilityBackend, DaprHTTPTransport, DaprStateRecord, DaprStateTransport
from anycode.backends.memory import AmbiguousBackendResultError, BackendUnavailableError, InMemoryDurabilityBackend
from anycode.backends.migration import export_filesystem_run, import_backend_snapshot
from anycode.backends.models import (
    BACKEND_CONTRACT_VERSION,
    Admission,
    AdmissionResult,
    AppendResult,
    ArtifactReferenceRecord,
    BackendCapabilities,
    BackendHealth,
    BackendSnapshot,
    BackendVersion,
    ClaimResult,
    CommitResult,
    ExternalSignal,
    WakeRegistration,
    WorkClaim,
    WorkItem,
)
from anycode.backends.protocol import DurabilityBackend
from anycode.backends.sqlite import SQLiteDurabilityBackend, UnsupportedBackendStateVersionError

__all__ = [
    "BACKEND_CONTRACT_VERSION",
    "Admission",
    "AdmissionResult",
    "AmbiguousBackendResultError",
    "AppendResult",
    "ArtifactReferenceRecord",
    "BackendCapabilities",
    "BackendHealth",
    "BackendSnapshot",
    "BackendUnavailableError",
    "BackendVersion",
    "ClaimResult",
    "CommitResult",
    "DaprDurabilityBackend",
    "DaprHTTPTransport",
    "DaprStateRecord",
    "DaprStateTransport",
    "DurabilityBackend",
    "ExternalSignal",
    "InMemoryDurabilityBackend",
    "SQLiteDurabilityBackend",
    "UnsupportedBackendStateVersionError",
    "WakeRegistration",
    "WorkClaim",
    "WorkItem",
    "export_filesystem_run",
    "import_backend_snapshot",
]
