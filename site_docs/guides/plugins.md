---
title: "Extend AnyCode with Plugins and Custom Providers"
description: "Package tools, provider factories, sensors, and turn hooks as an AnyCode plugin, install it safely on the engine, and publish it for entry-point discovery."
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

Development processes can load every installed plugin at once:

```python title="discover.py"
engine = AnyCode()
engine.load_installed_plugins()   # discovers + installs all entry-point plugins
print(engine.list_plugins())
```

Production processes should allowlist entry points or distribution names before import. Filtering happens before `EntryPoint.load()`, so untrusted plugin code is never executed:

```python title="trusted_plugins.py"
from anycode import AnyCode, PluginTrustPolicy

engine = AnyCode()
engine.load_installed_plugins(
    PluginTrustPolicy(
        allowed_distributions=("quantlix-anycode-approved",),
        allowed_entry_points=("internal-tools",),
    )
)
```

!!! tip "Broken plugins fail quietly"
    Entry-point discovery logs and skips a plugin that fails to import, so one bad third-party package can't stop your process from starting. Check `list_plugins()` to confirm what actually loaded.

!!! warning "Plugins execute trusted Python code"
    A plugin can register tools, hooks, sensors, and provider factories inside the host process. Allowlists establish which installed packages may load, but they do not sandbox approved plugins. Pin and review plugin distributions with the same care as runtime dependencies.

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

## The complete, runnable program

The pieces above come together in one file. It defines a plugin that ships a tool, a provider factory, and a sensor; installs it on an engine; resolves the plugin's provider through `create_adapter`; and confirms every agent built from the engine inherits the plugin's tools. It runs entirely offline against the bundled `FakeAdapter`, so no API key is required.

```python title="plugin_ecosystem.py"
import asyncio
from collections.abc import Mapping, Sequence

from pydantic import BaseModel

from anycode import (
    AnyCode,
    PluginBase,
    PluginManifest,
    ProviderFactory,
    ToolDefinition,
    ToolResult,
    VerificationSensorConfig,
    create_adapter,
)
from anycode.providers.fake import FakeAdapter, FakeResponse


class EchoInput(BaseModel):
    text: str


async def _echo_execute(validated_input: EchoInput, context: object) -> ToolResult:
    return ToolResult(data=f"plugin tool echo: {validated_input.text}", is_error=False)


ECHO_TOOL = ToolDefinition(
    name="acme_echo",
    description="Echo back the provided text via the Acme plugin.",
    input_model=EchoInput,
    execute=_echo_execute,
)


async def _build_fake_acme(**_: object) -> FakeAdapter:
    return FakeAdapter(responses=[FakeResponse(text="acme provider reply")])


class AcmePlugin(PluginBase):
    manifest = PluginManifest(
        name="acme-bundle",
        version="0.1.0",
        description="Demo plugin: echo tool + provider factory + sensor",
        homepage="https://example.com/acme",
    )

    def tools(self) -> Sequence[ToolDefinition]:
        return (ECHO_TOOL,)

    def provider_factories(self) -> Mapping[str, ProviderFactory]:
        return {"acme-fake": _build_fake_acme}

    def sensors(self) -> Sequence[VerificationSensorConfig]:
        return (VerificationSensorConfig(name="acme_sanity", kind="computational"),)


async def main() -> None:
    engine = AnyCode()
    installation = engine.register_plugin(AcmePlugin())
    print(f"installed: {installation.manifest.name} v{installation.manifest.version}")
    print(f"  tools:     {installation.tool_names}")
    print(f"  providers: {installation.provider_names}")
    print(f"  sensors:   {installation.sensor_names}")

    # The plugin's provider factory is now reachable via create_adapter.
    adapter = await create_adapter("acme-fake")
    print(f"resolved provider 'acme-fake' -> {adapter.name}")

    # Every agent built from this engine gets the plugin's tools wired in.
    agent = engine.build_agent({"name": "demo", "model": "fake-model", "provider": "openai"})
    print(f"agent tools include: {sorted(agent.get_tools())}")

    print("plugins currently installed:")
    for entry in engine.list_plugins():
        print(f"  - {entry.manifest.name} v{entry.manifest.version}: {entry.manifest.description}")


if __name__ == "__main__":
    asyncio.run(main())
```

Run it from the project root:

```bash
uv run python plugin_ecosystem.py
```

!!! tip "Tested copy"
    See [`examples/27_plugin_ecosystem.py`](https://github.com/Quantlix/anycode/blob/main/examples/27_plugin_ecosystem.py).

## Next steps

- [Configure LLM providers](providers.md) — the built-in providers your factory sits alongside.
- [Work with tools](tools.md) — build the `ToolDefinition` objects a plugin ships.
- [Verify output with quality gates](verification-gates.md) — the sensors a plugin can contribute.
- [Public API](../reference/public-api.md) — `PluginBase`, `PluginRegistry`, and the registry functions.
