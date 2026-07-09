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

        tool_names: list[str] = []
        for tool in plugin.tools():
            if not self._tool_registry.has(tool.name):
                self._tool_registry.register(tool)
                tool_names.append(tool.name)

        provider_names: list[str] = []
        for name, factory in plugin.provider_factories().items():
            register_provider_factory(name, factory)
            provider_names.append(name)

        sensor_names: list[str] = []
        for sensor_cfg in plugin.sensors():
            self._sensors.append(sensor_cfg)
            sensor_names.append(sensor_cfg.name)

        hooks = list(plugin.turn_hooks())
        self._hooks.extend(hooks)

        installation = PluginInstallation(
            manifest=manifest,
            tool_names=tuple(tool_names),
            provider_names=tuple(provider_names),
            sensor_names=tuple(sensor_names),
            turn_hook_count=len(hooks),
        )
        self._installations[manifest.name] = installation
        logger.info(
            "Installed plugin '%s' v%s — %d tools, %d providers, %d sensors, %d hooks",
            manifest.name,
            manifest.version,
            len(tool_names),
            len(provider_names),
            len(sensor_names),
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
