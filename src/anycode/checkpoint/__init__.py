"""Workflow checkpointing and resume for crash recovery."""

from anycode.checkpoint.manager import CheckpointManager
from anycode.checkpoint.serializer import UnsupportedCheckpointVersionError
from anycode.checkpoint.store import FilesystemCheckpointStore

__all__ = [
    "CheckpointManager",
    "FilesystemCheckpointStore",
    "UnsupportedCheckpointVersionError",
]

try:
    from anycode.checkpoint.store import SQLiteCheckpointStore  # noqa: F401

    __all__.append("SQLiteCheckpointStore")
except ImportError:
    pass
