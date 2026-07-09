"""Discover plugins published under the `anycode.plugins` entry-point group."""

from __future__ import annotations

import logging
from importlib.metadata import EntryPoint, entry_points

from anycode.types import Plugin

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "anycode.plugins"


def _load_entry_point(ep: EntryPoint) -> Plugin | None:
    try:
        loaded = ep.load()
    except Exception as e:  # pragma: no cover — defensive, depends on third-party code
        logger.warning("Failed to load anycode plugin entry-point '%s': %s", ep.name, e)
        return None

    candidate = loaded() if callable(loaded) else loaded
    if not isinstance(candidate, Plugin):
        logger.warning("Entry-point '%s' did not yield a Plugin (got %r).", ep.name, type(candidate))
        return None
    return candidate


def discover_entry_point_plugins() -> list[Plugin]:
    """Return every plugin published under the `anycode.plugins` entry-point group.

    Plugin failures are logged and skipped rather than raised so a single broken third-party
    package cannot prevent the engine from starting.
    """
    found: list[Plugin] = []
    try:
        eps = entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover — Python <3.10 compat path
        eps = entry_points().get(ENTRY_POINT_GROUP, [])  # type: ignore[attr-defined]

    for ep in eps:
        plugin = _load_entry_point(ep)
        if plugin is not None:
            found.append(plugin)
    return found
