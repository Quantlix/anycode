---
title: "Extend AnyCode with Plugins and Custom Providers"
description: "Package tools, provider factories, sensors, and turn hooks as an AnyCode plugin, install it on the engine, and publish it for entry-point discovery."
keywords: anycode plugins, PluginBase, PluginManifest, register_plugin, provider factory, register_provider_factory, entry point plugins, anycode.plugins, custom provider
---

# Extend AnyCode with Plugins

A plugin bundles extensions — custom tools, provider factories, verification sensors, and turn hooks — behind one installable object. Install it on an engine and its contributions become available to every agent. Publish it as a Python package and any AnyCode process can discover it automatically. This guide covers writing, installing, and distributing a plugin, plus the standalone provider-factory registry.

## What a plugin can contribute

Subclass `PluginBase` and override only the hooks you need. Every hook defaults to an empty collection, so a plugin can contribute one thing or all four.

| Override | Returns | Adds |
| --- | --- | --- |
| `manifest` (class attribute) | `PluginManifest` | Name, version, description (required) |
| `tools(self)` | `Sequence[ToolDefinition]` | Callable tools for agents |
| `provider_factories(self)` | `Mapping[str, ProviderFactory]` | New `provider="..."` names |
| `sensors(self)` | `Sequence[VerificationSensorConfig]` | Verification gates |
| `turn_hooks(self)` | `Sequence[TurnHook]` | Per-turn callbacks |

## Write a plugin

The manifest is required; everything else is optional. This plugin ships one tool, one fake provider, and one sensor.

```python title="acme_plugin.py"
from anycode import PluginBase, PluginManifest, VerificationSensorConfig
from anycode.providers.fake import FakeAdapter, FakeResponse


async def _build_acme(**_kwargs):
    return FakeAdapter(responses=[FakeResponse(text="acme reply")])


class AcmePlugin(PluginBase):
    manifest = PluginManifest(
        name="acme-bundle",
        version="0.1.0",
        description="Echo tool + provider + sanity sensor.",
    )

    def tools(self):
        return (ECHO_TOOL,)  # a ToolDefinition built with define_tool

    def provider_factories(self):
        return {"acme-fake": _build_acme}

    def sensors(self):
        return (VerificationSensorConfig(name="acme_sanity", kind="computational"),)
```

A `ProviderFactory` is an **async** callable that accepts the same keyword arguments as `create_adapter` (`api_key`, `base_url`, `endpoint`, `api_version`, `region`, `profile`, `model`) and returns an `LLMAdapter`.

!!! note "You can skip the base class"
    `Plugin` is a runtime-checkable protocol. Subclassing `PluginBase` is the easy path, but any object exposing the same attributes satisfies the contract.

## Install it on the engine

`register_plugin` installs a plugin and returns a `PluginInstallation` describing what it added. After installation, its tools, providers, and sensors are live.

```python title="use_plugin.py"
from anycode import AnyCode, create_adapter

engine = AnyCode()
installation = engine.register_plugin(AcmePlugin())

print(installation.tool_names)       # ('acme_echo',)
print(installation.provider_names)   # ('acme-fake',)

# The plugin's provider is now resolvable everywhere provider="..." is accepted:
adapter = await create_adapter("acme-fake")
```

Installing two plugins with the same `manifest.name` raises `ValueError`. A tool whose name is already registered is skipped rather than re-registered.

## Publish for auto-discovery

To ship a plugin as a package, expose it under the `anycode.plugins` entry-point group in your package's `pyproject.toml`:

```toml title="pyproject.toml"
[project.entry-points."anycode.plugins"]
acme = "acme_bundle:AcmePlugin"
```

Then any process can load every installed plugin at once:

```python title="discover.py"
engine = AnyCode()
engine.load_installed_plugins()   # discovers + installs all entry-point plugins
print(engine.list_plugins())
```

!!! tip "Broken plugins fail quietly"
    Entry-point discovery logs and skips a plugin that fails to import, so one bad third-party package can't stop your process from starting. Check `list_plugins()` to confirm what actually loaded.

## Register a provider without a full plugin

If all you need is a new provider name, use the standalone registry directly. It's a process-global table that `create_adapter` consults for any name it doesn't recognize.

```python title="register_provider.py"
from anycode import register_provider_factory, list_registered_providers, create_adapter


async def _build_myvendor(**kwargs):
    ...  # return an LLMAdapter

register_provider_factory("myvendor", _build_myvendor)
print(list_registered_providers())          # ['myvendor']
adapter = await create_adapter("myvendor")  # resolves through the registry
```

Re-registering the same name with a *different* factory raises `ValueError`; registering the identical factory again is a no-op.

## Next steps

- [Configure LLM providers](providers.md) — the built-in providers your factory sits alongside.
- [Work with tools](tools.md) — build the `ToolDefinition` objects a plugin ships.
- [Verify output with quality gates](verification-gates.md) — the sensors a plugin can contribute.
- [Public API](../reference/public-api.md) — `PluginBase`, `PluginRegistry`, and the registry functions.
