"""Tests for the plugin / extension ecosystem."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import pytest
from pydantic import BaseModel

from anycode import (
    AgentConfig,
    AnyCode,
    FakeResponse,
    PluginBase,
    PluginManifest,
    PluginRegistry,
    ToolDefinition,
    ToolResult,
    TurnHook,
    VerificationSensorConfig,
)
from anycode.plugins import discovery
from anycode.plugins.discovery import discover_entry_point_plugins
from anycode.plugins.registry import (
    _PROVIDER_FACTORIES,
    get_provider_factory,
    list_registered_providers,
    register_provider_factory,
)
from anycode.providers.adapter import create_adapter
from anycode.providers.fake import FakeAdapter
from anycode.types import PluginTrustPolicy, ProviderFactory


class EchoInput(BaseModel):
    text: str


async def _echo_execute(validated_input: EchoInput, context: object) -> ToolResult:
    return ToolResult(data=validated_input.text, is_error=False)


def _echo_tool() -> ToolDefinition:
    return ToolDefinition(name="echo_plugin_tool", description="Echo plugin tool", input_model=EchoInput, execute=_echo_execute)


@pytest.fixture(autouse=True)
def _isolate_provider_registry() -> Iterator[None]:
    """Drop any test-registered provider factories between tests."""
    keep = set(_PROVIDER_FACTORIES.keys())
    yield
    for name in list(_PROVIDER_FACTORIES.keys()):
        if name not in keep:
            _PROVIDER_FACTORIES.pop(name, None)


class TestPluginBase:
    def test_default_returns_empty(self) -> None:
        class P(PluginBase):
            manifest = PluginManifest(name="empty", version="1.0.0")

        plugin = P()
        assert plugin.tools() == ()
        assert plugin.provider_factories() == {}
        assert plugin.sensors() == ()
        assert plugin.turn_hooks() == ()


class TestPluginRegistry:
    def test_install_records_contributions(self) -> None:
        sensor = VerificationSensorConfig(name="custom_sensor", kind="computational")

        class P(PluginBase):
            manifest = PluginManifest(name="acme", version="0.1.0", description="Acme bundle")

            def tools(self) -> Sequence[ToolDefinition]:
                return (_echo_tool(),)

            def sensors(self) -> Sequence[VerificationSensorConfig]:
                return (sensor,)

        registry = PluginRegistry()
        installation = registry.install(P())

        assert installation.manifest.name == "acme"
        assert installation.tool_names == ("echo_plugin_tool",)
        assert installation.sensor_names == ("custom_sensor",)
        assert installation.turn_hook_count == 0
        assert registry.tool_registry.has("echo_plugin_tool")
        assert registry.sensors() == (sensor,)

    def test_duplicate_plugin_raises(self) -> None:
        class P(PluginBase):
            manifest = PluginManifest(name="dup", version="0.0.1")

        registry = PluginRegistry()
        registry.install(P())
        with pytest.raises(ValueError, match="already installed"):
            registry.install(P())

    def test_turn_hook_aggregation(self) -> None:
        class StubHook:
            async def before_turn(self, messages, context):  # type: ignore[no-untyped-def]
                return messages

            async def after_turn(self, response, context):  # type: ignore[no-untyped-def]
                return response

        hook = StubHook()

        class P(PluginBase):
            manifest = PluginManifest(name="hooky", version="0.0.1")

            def turn_hooks(self) -> Sequence[TurnHook]:
                return (hook,)

        registry = PluginRegistry()
        registry.install(P())
        assert registry.turn_hooks() == [hook]

    def test_provider_conflict_does_not_partially_install_plugin(self) -> None:
        async def registered_factory(**_: object) -> FakeAdapter:
            return FakeAdapter()

        async def conflicting_factory(**_: object) -> FakeAdapter:
            return FakeAdapter()

        register_provider_factory("occupied-provider", registered_factory)

        class P(PluginBase):
            manifest = PluginManifest(name="atomic", version="0.0.1")

            def tools(self) -> Sequence[ToolDefinition]:
                return (_echo_tool(),)

            def provider_factories(self) -> Mapping[str, ProviderFactory]:
                return {"occupied-provider": conflicting_factory}

            def sensors(self) -> Sequence[VerificationSensorConfig]:
                return (VerificationSensorConfig(name="should-not-install", kind="computational"),)

        registry = PluginRegistry()
        with pytest.raises(ValueError, match="already has a registered factory"):
            registry.install(P())

        assert not registry.tool_registry.has("echo_plugin_tool")
        assert registry.sensors() == ()
        assert registry.installations() == []
        assert get_provider_factory("occupied-provider") is registered_factory

    def test_invalid_sensor_does_not_partially_install_plugin(self) -> None:
        class P(PluginBase):
            manifest = PluginManifest(name="invalid", version="0.0.1")

            def tools(self) -> Sequence[ToolDefinition]:
                return (_echo_tool(),)

            def sensors(self) -> Sequence[VerificationSensorConfig]:
                return (object(),)  # type: ignore[return-value]

        registry = PluginRegistry()
        with pytest.raises(AttributeError):
            registry.install(P())

        assert not registry.tool_registry.has("echo_plugin_tool")
        assert registry.sensors() == ()
        assert registry.installations() == []


class TestProviderFactory:
    async def test_register_and_resolve(self) -> None:
        async def _factory(**_: object) -> FakeAdapter:
            return FakeAdapter()

        register_provider_factory("acme-provider", _factory)
        assert "acme-provider" in list_registered_providers()
        assert get_provider_factory("acme-provider") is _factory

        adapter = await create_adapter("acme-provider")
        # create_adapter wraps every provider (built-in or plugin) in the
        # resilience layer by default; the raw adapter sits underneath.
        from anycode.providers.resilience import ResilientAdapter

        assert isinstance(adapter, ResilientAdapter)
        assert isinstance(adapter.inner, FakeAdapter)

    async def test_unknown_provider_still_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            await create_adapter("definitely-not-registered")

    async def test_registered_provider_runs_through_agent_config(self) -> None:
        async def _factory(**_: object) -> FakeAdapter:
            return FakeAdapter(responses=[FakeResponse(text="plugin reply")])

        register_provider_factory("agent-plugin-provider", _factory)
        engine = AnyCode()
        try:
            result = await engine.run_agent(
                AgentConfig(name="plugin-agent", model="fake-model", provider="agent-plugin-provider"),
                "hello",
            )
        finally:
            await engine.close()

        assert result.success is True
        assert result.output == "plugin reply"

    def test_duplicate_factory_for_same_name_rejected(self) -> None:
        async def _a(**_: object) -> FakeAdapter:
            return FakeAdapter()

        async def _b(**_: object) -> FakeAdapter:
            return FakeAdapter()

        register_provider_factory("dup-provider", _a)
        with pytest.raises(ValueError, match="already has a registered factory"):
            register_provider_factory("dup-provider", _b)


class TestEngineIntegration:
    def test_register_plugin_exposes_tools_to_agents(self) -> None:
        class P(PluginBase):
            manifest = PluginManifest(name="tooly", version="0.1.0")

            def tools(self) -> Sequence[ToolDefinition]:
                return (_echo_tool(),)

        engine = AnyCode()
        installation = engine.register_plugin(P())
        assert installation.tool_names == ("echo_plugin_tool",)

        agent = engine.build_agent({"name": "demo", "model": "fake-model", "provider": "openai"})
        assert "echo_plugin_tool" in agent.get_tools()

    def test_register_plugin_wires_provider_factory(self) -> None:
        async def _factory(**_: object) -> FakeAdapter:
            return FakeAdapter()

        class P(PluginBase):
            manifest = PluginManifest(name="provider-plugin", version="0.1.0")

            def provider_factories(self) -> Mapping[str, ProviderFactory]:
                return {"acme-plugin-provider": _factory}

        engine = AnyCode()
        engine.register_plugin(P())
        assert "acme-plugin-provider" in list_registered_providers()

    def test_list_plugins(self) -> None:
        class P(PluginBase):
            manifest = PluginManifest(name="list-me", version="0.0.1")

        engine = AnyCode()
        engine.register_plugin(P())
        installed = engine.list_plugins()
        assert [i.manifest.name for i in installed] == ["list-me"]


class TestEntryPointDiscovery:
    def test_discovery_returns_list(self) -> None:
        # Without any third-party plugins installed, discovery returns an empty list
        # — the call must not raise.
        result = discover_entry_point_plugins()
        assert isinstance(result, list)

    def test_policy_filters_entry_points_before_loading(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class P(PluginBase):
            manifest = PluginManifest(name="trusted", version="1.0.0")

        class Distribution:
            name = "acme-plugins"

        class EntryPoint:
            name = "trusted"
            dist = Distribution()

            @staticmethod
            def load():  # type: ignore[no-untyped-def]
                return P

        monkeypatch.setattr(discovery, "entry_points", lambda **_kwargs: [EntryPoint()])

        blocked = discover_entry_point_plugins(PluginTrustPolicy())
        allowed = discover_entry_point_plugins(PluginTrustPolicy(allowed_entry_points=("trusted",)))

        assert blocked == []
        assert len(allowed) == 1
