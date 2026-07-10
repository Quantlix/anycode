"""Shared filesystem helpers for editing tools.

Writes go through a same-directory temp file and an atomic ``os.replace`` so a
crash mid-write can never leave a truncated target. Content is written
byte-exact (``newline=""``) so line endings the model produced are preserved
rather than translated to the host's ``os.linesep``.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from anycode.constants import DEFAULT_ENCODING


def atomic_write_text(target: Path, content: str) -> None:
    """Atomically write ``content`` to ``target``, preserving exact bytes."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=DEFAULT_ENCODING, newline="") as handle:
            handle.write(content)
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
