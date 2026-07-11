"""Plugin registry — installs plugins and tracks the extension points they contribute."""

from __future__ import annotations

import logging
from collections.abc import Iterable

from anycode.tools.registry import ToolRegistry
from anycode.types import (
    Plugin,
    PluginInstallation,
    ProviderFactory,
    TurnHook,
    VerificationSensorConfig,
)

logger = logging.getLogger(__name__)


_PROVIDER_FACTORIES: dict[str, ProviderFactory] = {}


def register_provider_factory(name: str, factory: ProviderFactory) -> None:
    """Register a provider factory under *name*.

    Used by plugins to extend `create_adapter` without modifying core code. Raises if the name
    is already registered to a different factory.
    """
    existing = _PROVIDER_FACTORIES.get(name)
    if existing is not None and existing is not factory:
        raise ValueError(f"Provider '{name}' already has a registered factory.")
    _PROVIDER_FACTORIES[name] = factory


def get_provider_factory(name: str) -> ProviderFactory | None:
    return _PROVIDER_FACTORIES.get(name)


def list_registered_providers() -> list[str]:
    return sorted(_PROVIDER_FACTORIES.keys())


class PluginRegistry:
    """Tracks installed plugins and wires their extension points into AnyCode."""

    def __init__(self, tool_registry: ToolRegistry | None = None) -> None:
        self._tool_registry = tool_registry or ToolRegistry()
        self._installations: dict[str, PluginInstallation] = {}
        self._hooks: list[TurnHook] = []
        self._sensors: list[VerificationSensorConfig] = []

    @property
    def tool_registry(self) -> ToolRegistry:
        return self._tool_registry

    def install(self, plugin: Plugin) -> PluginInstallation:
        manifest = plugin.manifest
        if manifest.name in self._installations:
            raise ValueError(f"Plugin '{manifest.name}' is already installed.")

        tools = list(plugin.tools())
        provider_factories = dict(plugin.provider_factories())
        sensors = list(plugin.sensors())
        hooks = list(plugin.turn_hooks())

        tools_to_register = {tool.name: tool for tool in tools if not self._tool_registry.has(tool.name)}
        for name, factory in provider_factories.items():
            existing = get_provider_factory(name)
            if existing is not None and existing is not factory:
                raise ValueError(f"Provider '{name}' already has a registered factory.")

        installation = PluginInstallation(
            manifest=manifest,
            tool_names=tuple(tools_to_register),
            provider_names=tuple(provider_factories),
            sensor_names=tuple(sensor.name for sensor in sensors),
            turn_hook_count=len(hooks),
        )

        for tool in tools_to_register.values():
            self._tool_registry.register(tool)

        for name, factory in provider_factories.items():
            register_provider_factory(name, factory)

        self._sensors.extend(sensors)
        self._hooks.extend(hooks)

        self._installations[manifest.name] = installation
        logger.info(
            "Installed plugin '%s' v%s — %d tools, %d providers, %d sensors, %d hooks",
            manifest.name,
            manifest.version,
            len(tools_to_register),
            len(provider_factories),
            len(sensors),
            len(hooks),
        )
        return installation

    def install_many(self, plugins: Iterable[Plugin]) -> list[PluginInstallation]:
        return [self.install(p) for p in plugins]

    def installations(self) -> list[PluginInstallation]:
        return list(self._installations.values())

    def get(self, name: str) -> PluginInstallation | None:
        return self._installations.get(name)

    def sensors(self) -> tuple[VerificationSensorConfig, ...]:
        return tuple(self._sensors)

    def turn_hooks(self) -> list[TurnHook]:
        return list(self._hooks)
