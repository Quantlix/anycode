"""Base class authors can subclass to build a plugin with zero boilerplate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from anycode.types import (
    PluginManifest,
    ProviderFactory,
    ToolDefinition,
    TurnHook,
    VerificationSensorConfig,
)


class PluginBase:
    """Default no-op `Plugin` implementation.

    Subclasses override `manifest` and any accessor they actually contribute. The base
    methods return empty containers so the `Plugin` protocol is always satisfied.
    """

    manifest: PluginManifest

    def tools(self) -> Sequence[ToolDefinition]:
        return ()

    def provider_factories(self) -> Mapping[str, ProviderFactory]:
        return {}

    def sensors(self) -> Sequence[VerificationSensorConfig]:
        return ()

    def turn_hooks(self) -> Sequence[TurnHook]:
        return ()
