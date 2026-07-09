"""Plugin / extension surface for AnyCode."""

from anycode.plugins.discovery import discover_entry_point_plugins
from anycode.plugins.plugin import PluginBase
from anycode.plugins.registry import (
    PluginRegistry,
    get_provider_factory,
    list_registered_providers,
    register_provider_factory,
)

__all__ = [
    "PluginBase",
    "PluginRegistry",
    "discover_entry_point_plugins",
    "get_provider_factory",
    "list_registered_providers",
    "register_provider_factory",
]
