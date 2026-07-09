# Demo 27 — Plugin / Extension Ecosystem
# Execute: uv run python examples/27_plugin_ecosystem.py
#
# Demonstrates:
#   1. Authoring a plugin with a tool, a provider factory, and a sensor
#   2. Installing the plugin into an AnyCode engine
#   3. Resolving the plugin-registered provider through create_adapter
#   4. Listing installed plugins for inspection
#
# Runs entirely offline using the bundled FakeAdapter.

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
