"""Tests for keyword-constructed agents, provider defaults, and blocking entry points."""

from __future__ import annotations

import asyncio

import pytest

from anycode import (
    Agent,
    AgentConfig,
    AgentConfigError,
    ToolDefinitionError,
    ToolExecutor,
    ToolRegistry,
    compose_instructions,
    register_built_in_tools,
    tool,
)
from anycode.core.defaults import DEFAULT_MODEL_ENV_VAR, DEFAULT_PROVIDER_ENV_VAR, default_model, detect_provider
from anycode.providers.fake import FakeAdapter, FakeResponse

PROVIDER_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AWS_DEFAULT_REGION",
    "OLLAMA_BASE_URL",
    "OLLAMA_API_KEY",
    "OLLAMA_MODEL",
    DEFAULT_MODEL_ENV_VAR,
    DEFAULT_PROVIDER_ENV_VAR,
)


@pytest.fixture
def clean_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in PROVIDER_ENV_VARS:
        monkeypatch.delenv(variable, raising=False)


@tool
def echo(text: str) -> str:
    """Echo the given text."""
    return text


# ---------------------------------------------------------------------------
# Keyword construction
# ---------------------------------------------------------------------------


def test_keyword_construction_builds_config(clean_provider_env: None) -> None:
    agent = Agent(
        name="researcher",
        model="gpt-4o-mini",
        provider="openai",
        instructions="Find primary sources.",
        max_turns=8,
        temperature=0.2,
    )
    assert agent.name == "researcher"
    assert agent.config.model == "gpt-4o-mini"
    assert agent.config.provider == "openai"
    assert agent.config.system_prompt == "Find primary sources."
    assert agent.config.max_turns == 8
    assert agent.config.temperature == 0.2


def test_keyword_construction_wires_built_in_tools(clean_provider_env: None) -> None:
    agent = Agent(name="worker", model="m", provider="openai")
    assert {"bash", "file_read", "file_write"} <= {definition.name for definition in agent.tools}
    assert agent.config.tools is None


def test_system_prompt_is_an_alias_for_instructions(clean_provider_env: None) -> None:
    agent = Agent(name="a", model="m", provider="openai", system_prompt="Be terse.")
    assert agent.config.system_prompt == "Be terse."


def test_missing_name_is_rejected(clean_provider_env: None) -> None:
    with pytest.raises(AgentConfigError, match="needs a name"):
        Agent(model="m", provider="openai")


# ---------------------------------------------------------------------------
# Legacy construction stays intact
# ---------------------------------------------------------------------------


def test_positional_construction_is_unchanged() -> None:
    config = AgentConfig(name="legacy", model="m", provider="openai")
    registry = ToolRegistry()
    register_built_in_tools(registry)
    executor = ToolExecutor(registry)

    agent = Agent(config, registry, executor)
    assert agent.config is config
    assert agent._registry is registry
    assert agent._executor is executor


def test_config_object_with_auto_wiring() -> None:
    config = AgentConfig(name="halfway", model="m", provider="openai")
    agent = Agent(config)
    assert {definition.name for definition in agent.tools}
    assert isinstance(agent._executor, ToolExecutor)


def test_config_plus_field_keyword_is_rejected() -> None:
    config = AgentConfig(name="x", model="m", provider="openai")
    with pytest.raises(AgentConfigError, match="both a config object and name="):
        Agent(config, name="y")


def test_config_plus_tools_is_allowed() -> None:
    config = AgentConfig(name="x", model="m", provider="openai", tools=["bash"])
    agent = Agent(config, tools=[echo])
    assert agent.config.tools == ["bash", "echo"]
    assert "echo" in {definition.name for definition in agent.tools}


# ---------------------------------------------------------------------------
# Role / goal / backstory framing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("role", "goal", "backstory", "expected"),
    [
        ("a Senior Analyst", None, None, "You are a Senior Analyst."),
        (None, "ship the report", None, "Your goal: ship the report"),
        (None, None, "15 years auditing", "Background: 15 years auditing"),
        ("an Editor", "tighten prose", None, "You are an Editor.\n\nYour goal: tighten prose"),
        ("an Editor", None, "ex-journalist", "You are an Editor.\n\nBackground: ex-journalist"),
        (None, "tighten prose", "ex-journalist", "Your goal: tighten prose\n\nBackground: ex-journalist"),
        (
            "an Editor",
            "tighten prose",
            "ex-journalist",
            "You are an Editor.\n\nYour goal: tighten prose\n\nBackground: ex-journalist",
        ),
    ],
)
def test_compose_instructions_covers_every_subset(role: str | None, goal: str | None, backstory: str | None, expected: str) -> None:
    assert compose_instructions(role, goal, backstory) == expected


def test_compose_instructions_strips_trailing_period_from_role() -> None:
    assert compose_instructions(role="a Senior Analyst.") == "You are a Senior Analyst."


def test_role_framing_reaches_the_config(clean_provider_env: None) -> None:
    agent = Agent(name="analyst", model="m", provider="openai", role="a Senior Analyst", goal="value the firm")
    assert agent.config.system_prompt == "You are a Senior Analyst.\n\nYour goal: value the firm"


def test_instructions_and_role_together_are_rejected(clean_provider_env: None) -> None:
    with pytest.raises(AgentConfigError, match="Pick one"):
        Agent(name="a", model="m", provider="openai", instructions="Do it.", role="an Editor")


def test_conflicting_prompt_aliases_are_rejected(clean_provider_env: None) -> None:
    with pytest.raises(AgentConfigError, match="aliases"):
        Agent(name="a", model="m", provider="openai", instructions="one", system_prompt="two")


# ---------------------------------------------------------------------------
# Provider and model detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("variable", "expected_provider", "expected_model"),
    [
        ("ANTHROPIC_API_KEY", "anthropic", "claude-haiku-4-5"),
        ("OPENAI_API_KEY", "openai", "gpt-4o-mini"),
        ("GOOGLE_API_KEY", "google", "gemini-2.0-flash"),
        ("AZURE_OPENAI_API_KEY", "azure", "gpt-4o-mini"),
    ],
)
def test_provider_detected_from_credentials(
    clean_provider_env: None,
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    expected_provider: str,
    expected_model: str,
) -> None:
    monkeypatch.setenv(variable, "value")
    agent = Agent(name="auto")
    assert agent.config.provider == expected_provider
    assert agent.config.model == expected_model


def test_detection_prefers_anthropic_over_openai(clean_provider_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "b")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    assert detect_provider() == "anthropic"


def test_explicit_provider_env_var_wins(clean_provider_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "b")
    monkeypatch.setenv(DEFAULT_PROVIDER_ENV_VAR, "google")
    assert detect_provider() == "google"


def test_default_model_env_var_overrides(clean_provider_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "b")
    monkeypatch.setenv(DEFAULT_MODEL_ENV_VAR, "gpt-5-mini")
    assert Agent(name="auto").config.model == "gpt-5-mini"


def test_ollama_model_comes_from_its_own_env_var(clean_provider_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:8b")
    assert default_model("ollama") == "qwen3:8b"


def test_no_credentials_produces_an_actionable_error(clean_provider_env: None) -> None:
    with pytest.raises(AgentConfigError, match="ANTHROPIC_API_KEY"):
        Agent(name="auto")


def test_unknown_provider_without_default_model_is_rejected(clean_provider_env: None) -> None:
    with pytest.raises(AgentConfigError, match="no default model is known"):
        Agent(name="auto", provider="ollama")


# ---------------------------------------------------------------------------
# Tool wiring
# ---------------------------------------------------------------------------


def test_tools_accepts_mixed_specifications(clean_provider_env: None) -> None:
    definition = ToolRegistry()
    register_built_in_tools(definition)
    grep = definition.get("grep")
    assert grep is not None

    agent = Agent(name="mixed", model="m", provider="openai", tools=[echo, "bash", grep])
    assert agent.config.tools == ["echo", "bash", "grep"]
    assert {tool_def.name for tool_def in agent.tools} == {"echo", "bash", "grep"}


def test_empty_tool_list_means_no_tools(clean_provider_env: None) -> None:
    agent = Agent(name="toolless", model="m", provider="openai", tools=[])
    assert agent.config.tools == []
    assert agent.tools == []


def test_duplicate_tool_names_are_rejected(clean_provider_env: None) -> None:
    with pytest.raises(ToolDefinitionError, match="Duplicate tool name"):
        Agent(name="dupes", model="m", provider="openai", tools=[echo, echo])


def test_unknown_tool_name_lists_alternatives(clean_provider_env: None) -> None:
    with pytest.raises(ToolDefinitionError, match="Available: bash"):
        Agent(name="typo", model="m", provider="openai", tools=["file_reed"])


async def test_call_tool_runs_without_the_llm(clean_provider_env: None) -> None:
    agent = Agent(name="direct", model="m", provider="openai", tools=[echo])
    result = await agent.call_tool("echo", text="ping")
    assert result.data == "ping"
    assert not result.is_error


def test_call_tool_sync_runs_without_the_llm(clean_provider_env: None) -> None:
    agent = Agent(name="direct", model="m", provider="openai", tools=[echo])
    assert agent.call_tool_sync("echo", text="pong").data == "pong"


async def test_call_tool_reports_unregistered_names(clean_provider_env: None) -> None:
    agent = Agent(name="direct", model="m", provider="openai", tools=[echo])
    result = await agent.call_tool("nope")
    assert result.is_error
    assert "not registered" in result.data


def test_repr_reports_identity_and_tool_count(clean_provider_env: None) -> None:
    agent = Agent(name="shown", model="m", provider="openai", tools=[echo])
    assert repr(agent) == "Agent(name='shown', model='m', provider='openai', tools=1)"


# ---------------------------------------------------------------------------
# Blocking entry points
# ---------------------------------------------------------------------------


def _fake_agent(monkeypatch: pytest.MonkeyPatch, text: str = "done") -> Agent:
    agent = Agent(name="sync", model="m", provider="openai", tools=[])

    async def _adapter(*_args: object, **_kwargs: object) -> FakeAdapter:
        return FakeAdapter(responses=[FakeResponse(text=text)])

    monkeypatch.setattr("anycode.core.agent.create_adapter", _adapter)
    return agent


def test_run_sync_returns_a_result(clean_provider_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _fake_agent(monkeypatch, "hello from sync")
    result = agent.run_sync("hi")
    assert result.success
    assert result.output == "hello from sync"


def test_prompt_sync_keeps_history(clean_provider_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _fake_agent(monkeypatch, "reply")
    agent.prompt_sync("first")
    assert len(agent.get_history()) >= 2


def test_stream_sync_yields_events(clean_provider_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _fake_agent(monkeypatch, "streamed")
    events = list(agent.stream_sync("hi"))
    assert [event.type for event in events][-1] == "done"
    assert "".join(str(event.data) for event in events if event.type == "text") == "streamed"


async def test_run_sync_inside_a_loop_is_rejected(clean_provider_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _fake_agent(monkeypatch)
    with pytest.raises(RuntimeError, match="await agent.run"):
        agent.run_sync("hi")


async def test_stream_sync_inside_a_loop_is_rejected(clean_provider_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _fake_agent(monkeypatch)
    with pytest.raises(RuntimeError, match="async for event in agent.stream"):
        list(agent.stream_sync("hi"))


def test_sync_runner_propagates_exceptions(clean_provider_env: None) -> None:
    from anycode.helpers.sync_runner import run_coroutine_blocking

    async def _boom() -> None:
        raise ValueError("kaboom")

    with pytest.raises(ValueError, match="kaboom"):
        run_coroutine_blocking(_boom(), sync_call="x()", async_call="await x()")


def test_sync_runner_does_not_leave_pending_coroutines(clean_provider_env: None) -> None:
    from anycode.helpers.sync_runner import run_coroutine_blocking

    async def _never() -> None:  # pragma: no cover - never awaited
        await asyncio.sleep(0)

    async def _inside() -> None:
        with pytest.raises(RuntimeError):
            run_coroutine_blocking(_never(), sync_call="x()", async_call="await x()")

    asyncio.run(_inside())
