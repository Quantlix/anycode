"""Discover plugins published under the `anycode.plugins` entry-point group."""

from __future__ import annotations

import logging
from importlib.metadata import EntryPoint, entry_points

from anycode.security.redaction import safe_exception_message
from anycode.types import Plugin, PluginTrustPolicy

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "anycode.plugins"


def _load_entry_point(ep: EntryPoint) -> Plugin | None:
    try:
        loaded = ep.load()
        candidate = loaded() if callable(loaded) else loaded
    except Exception as error:  # pragma: no cover — defensive, depends on third-party code
        logger.warning("Failed to load anycode plugin entry-point '%s': %s", ep.name, safe_exception_message(error))
        return None

    if not isinstance(candidate, Plugin):
        logger.warning("Entry-point '%s' did not yield a Plugin (got %r).", ep.name, type(candidate))
        return None
    return candidate


def discover_entry_point_plugins(policy: PluginTrustPolicy | None = None) -> list[Plugin]:
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
        if not _entry_point_allowed(ep, policy):
            logger.info("Skipped untrusted anycode plugin entry-point '%s'.", ep.name)
            continue
        plugin = _load_entry_point(ep)
        if plugin is not None:
            found.append(plugin)
    return found


def _entry_point_allowed(ep: EntryPoint, policy: PluginTrustPolicy | None) -> bool:
    if policy is None or policy.allow_unlisted:
        return True
    distribution = ep.dist.name if ep.dist is not None else None
    return ep.name in policy.allowed_entry_points or (distribution is not None and distribution in policy.allowed_distributions)
