"""RFC 9562 UUID v7 — time-ordered, k-sortable unique identifiers.

Bundled to avoid an external dependency (Python stdlib only has uuid1–uuid5).
UUID v7 encodes a Unix-epoch millisecond timestamp in the high 48 bits,
making IDs monotonically increasing and naturally sortable.
"""

from __future__ import annotations

import os
import time
from uuid import UUID


def uuid7() -> UUID:
    """Generate a UUID v7 per RFC 9562 §5.7.

    Layout (128 bits):
        unix_ts_ms (48) | ver (4) | rand_a (12) | var (2) | rand_b (62)
    """
    timestamp_ms = int(time.time() * 1000)

    rand_bytes = os.urandom(10)

    # Bits 0-47: unix_ts_ms
    uuid_int = timestamp_ms << 80

    # Bits 48-51: version = 0b0111 (7)
    uuid_int |= 0x7000 << 64

    # Bits 52-63: rand_a (12 bits from rand_bytes[0:2])
    rand_a = int.from_bytes(rand_bytes[:2], "big") & 0x0FFF
    uuid_int |= rand_a << 64

    # Bits 64-65: variant = 0b10
    uuid_int |= 0x8000_0000_0000_0000

    # Bits 66-127: rand_b (62 bits from rand_bytes[2:10])
    rand_b = int.from_bytes(rand_bytes[2:], "big") & 0x3FFF_FFFF_FFFF_FFFF
    uuid_int |= rand_b

    return UUID(int=uuid_int)
