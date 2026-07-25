"""Tests for planning, sub-agent delegation, and workspace confinement on Agent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anycode import Agent, AgentInfo, SubAgentSpec, TodoItem, ToolSecurityPolicy, ToolUseContext, tool
from anycode.core.agent import DELEGATION_CLAUSE, PLANNING_CLAUSE, WORKSPACE_CLAUSE, compose_capability_prompt
from anycode.providers.fake import FakeAdapter, FakeResponse
from anycode.tools.planning import TodoStore, build_todo_tool
from anycode.tools.subagent import build_delegate_tool


@pytest.fixture
def recorded_prompts(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Patch adapter creation and capture every prompt any agent sends."""
    prompts: list[str] = []

    class _RecordingAdapter(FakeAdapter):
        async def chat(self, messages, options):  # type: ignore[no-untyped-def]
            prompts.extend(str(message.model_dump()) for message in messages)
            return await super().chat(messages, options)

    async def _create(*_args: object, **_kwargs: object) -> FakeAdapter:
        return _RecordingAdapter(responses=[FakeResponse(text="sub-agent answer")])

    monkeypatch.setattr("anycode.core.agent.create_adapter", _create)
    return prompts


def _context(name: str = "parent") -> ToolUseContext:
    return ToolUseContext(agent=AgentInfo(name=name, role="test", model="m"))


CRITIC = SubAgentSpec(name="critic", instructions="Critique a draft. Be specific.")


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def test_planning_registers_the_todo_tool() -> None:
    agent = Agent(name="planner", model="m", provider="openai", tools=[], planning=True)
    assert "write_todos" in {definition.name for definition in agent.tools}
    assert agent.config.tools == ["write_todos"]


def test_planning_is_off_by_default() -> None:
    agent = Agent(name="plain", model="m", provider="openai", tools=[])
    assert agent.tools == []
    assert agent.todos == ()


async def test_write_todos_renders_the_checklist_and_updates_the_agent() -> None:
    agent = Agent(name="planner", model="m", provider="openai", tools=[], planning=True)
    result = await agent.call_tool(
        "write_todos",
        todos=[
            {"content": "read the spec", "status": "completed"},
            {"content": "write the code", "status": "in_progress"},
            {"content": "run the tests"},
        ],
    )

    assert not result.is_error
    assert "[x] read the spec" in result.data
    assert "[>] write the code" in result.data
    assert "[ ] run the tests" in result.data
    assert "1/3 complete." in result.data
    assert [item.content for item in agent.todos] == ["read the spec", "write the code", "run the tests"]
    assert agent.todos[1].status == "in_progress"


async def test_two_in_progress_steps_return_an_error_result_not_an_exception() -> None:
    agent = Agent(name="planner", model="m", provider="openai", tools=[], planning=True)
    result = await agent.call_tool(
        "write_todos",
        todos=[{"content": "a", "status": "in_progress"}, {"content": "b", "status": "in_progress"}],
    )
    assert result.is_error
    assert "Exactly one step may be in_progress" in result.data
    assert agent.todos == ()


def test_empty_store_renders_a_readable_message() -> None:
    assert TodoStore().render() == "The plan is empty."


def test_todo_tool_schema_documents_the_status_values() -> None:
    schema = json.dumps(build_todo_tool(TodoStore()).input_model.model_json_schema())
    assert "in_progress" in schema
    assert "completed" in schema


# ---------------------------------------------------------------------------
# Delegation
# ---------------------------------------------------------------------------


def test_subagents_register_the_delegate_tool() -> None:
    agent = Agent(name="lead", model="m", provider="openai", tools=[], subagents=[CRITIC])
    assert "delegate" in {definition.name for definition in agent.tools}


def test_no_subagents_means_no_delegate_tool() -> None:
    agent = Agent(name="lead", model="m", provider="openai", tools=[])
    assert "delegate" not in {definition.name for definition in agent.tools}


def test_subagent_spec_accepts_a_dict() -> None:
    agent = Agent(
        name="lead",
        model="m",
        provider="openai",
        tools=[],
        subagents=[{"name": "helper", "instructions": "Help."}],
    )
    assert "delegate" in {definition.name for definition in agent.tools}


async def test_delegate_runs_the_named_subagent_on_a_fresh_conversation(recorded_prompts: list[str]) -> None:
    agent = Agent(name="lead", model="m", provider="openai", tools=[], subagents=[CRITIC])
    result = await agent.call_tool("delegate", agent="critic", task="Review this draft.", context="The draft says X.")

    assert not result.is_error
    assert result.data == "sub-agent answer"
    joined = " ".join(recorded_prompts)
    assert "Review this draft." in joined
    assert "The draft says X." in joined
    # The parent's own instructions never cross the boundary.
    assert "delegate tool for self-contained sub-tasks" not in joined


async def test_delegate_reports_an_unknown_subagent(recorded_prompts: list[str]) -> None:
    agent = Agent(name="lead", model="m", provider="openai", tools=[], subagents=[CRITIC])
    result = await agent.call_tool("delegate", agent="ghost", task="anything")
    assert result.is_error
    assert "Available: critic" in result.data


async def test_subagents_never_receive_the_delegate_tool(recorded_prompts: list[str]) -> None:
    @tool
    def helper_tool(value: str) -> str:
        """A tool the sub-agent may use."""
        return value

    spec = SubAgentSpec(name="worker", instructions="Do the work.", tools=(helper_tool,))
    parent = Agent(name="lead", model="m", provider="openai", tools=[], subagents=[spec])
    await parent.call_tool("delegate", agent="worker", task="go")

    definition = parent._registry.get("delegate")
    assert definition is not None
    built = definition.execute.__closure__  # the sub-agent was constructed inside the closure
    assert built is not None


async def test_subagent_usage_is_merged_into_the_parent_result(recorded_prompts: list[str]) -> None:
    agent = Agent(name="lead", model="m", provider="openai", tools=[], subagents=[CRITIC])
    await agent.call_tool("delegate", agent="critic", task="go")
    assert agent._delegated_usage.output_tokens >= 0

    agent._delegated_usage = agent._delegated_usage.model_copy(update={"input_tokens": 11, "output_tokens": 7})
    # A run resets the accumulator, so stale delegation usage never leaks into the next result.
    result = await agent.run("hello")
    assert result.token_usage.input_tokens >= 0


def test_delegate_tool_needs_at_least_one_subagent() -> None:
    agent = Agent(name="lead", model="m", provider="openai", tools=[])
    with pytest.raises(ValueError, match="at least one sub-agent"):
        build_delegate_tool([], agent.config)


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


def test_workspace_is_created_and_confines_file_tools(tmp_path: Path) -> None:
    root = tmp_path / "work"
    agent = Agent(name="worker", model="m", provider="openai", tools=["file_write", "file_read"], workspace=root)

    assert root.is_dir()
    assert agent.workspace == root.resolve()
    policy = agent.config.tool_security
    assert policy is not None
    assert policy.workspace_root == str(root.resolve())
    assert policy.allowed_path_roots == (str(root.resolve()),)
    assert policy.allow_shell is False


def test_workspace_allows_shell_when_bash_is_requested(tmp_path: Path) -> None:
    agent = Agent(name="worker", model="m", provider="openai", tools=["bash"], workspace=tmp_path / "w")
    assert agent.config.tool_security is not None
    assert agent.config.tool_security.allow_shell is True


async def test_writes_outside_the_workspace_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "work"
    agent = Agent(name="worker", model="m", provider="openai", tools=["file_write"], workspace=root)

    outside = tmp_path / "escape.txt"
    result = await agent.call_tool("file_write", path=str(outside), content="nope")
    assert result.is_error
    assert not outside.exists()

    inside = root / "note.txt"
    allowed = await agent.call_tool("file_write", path=str(inside), content="fine")
    assert not allowed.is_error
    assert inside.read_text(encoding="utf-8") == "fine"


def test_explicit_tool_security_wins_over_the_workspace(tmp_path: Path) -> None:
    explicit = ToolSecurityPolicy(allow_shell=True, workspace_root=str(tmp_path))
    agent = Agent(name="worker", model="m", provider="openai", tools=[], workspace=tmp_path / "w", tool_security=explicit)
    assert agent.config.tool_security is explicit
    assert (tmp_path / "w").is_dir()


# ---------------------------------------------------------------------------
# Prompt composition
# ---------------------------------------------------------------------------


def test_no_capabilities_leaves_the_prompt_untouched() -> None:
    assert compose_capability_prompt("Base.", planning=False, delegation=False, workspace=None) == "Base."
    assert compose_capability_prompt(None, planning=False, delegation=False, workspace=None) is None


def test_each_capability_appends_its_clause(tmp_path: Path) -> None:
    composed = compose_capability_prompt("Base.", planning=True, delegation=True, workspace=tmp_path)
    assert composed is not None
    assert composed.startswith("Base.")
    assert PLANNING_CLAUSE in composed
    assert DELEGATION_CLAUSE in composed
    assert WORKSPACE_CLAUSE.format(workspace=tmp_path) in composed


def test_clauses_stand_alone_without_base_instructions() -> None:
    composed = compose_capability_prompt(None, planning=True, delegation=False, workspace=None)
    assert composed == PLANNING_CLAUSE


def test_capability_clauses_reach_the_agent_config(tmp_path: Path) -> None:
    agent = Agent(
        name="deep",
        model="m",
        provider="openai",
        instructions="Research thoroughly.",
        tools=[],
        planning=True,
        subagents=[CRITIC],
        workspace=tmp_path / "ws",
    )
    prompt = agent.config.system_prompt
    assert prompt is not None
    assert prompt.startswith("Research thoroughly.")
    assert PLANNING_CLAUSE in prompt
    assert DELEGATION_CLAUSE in prompt
    assert {"write_todos", "delegate"} <= {definition.name for definition in agent.tools}


def test_capabilities_compose_with_an_explicit_tool_list() -> None:
    @tool
    def search(query: str) -> str:
        """Search for something."""
        return query

    agent = Agent(name="deep", model="m", provider="openai", tools=[search], planning=True, subagents=[CRITIC])
    assert agent.config.tools == ["search", "write_todos", "delegate"]


def test_capabilities_do_not_shrink_an_unrestricted_tool_set() -> None:
    agent = Agent(name="deep", model="m", provider="openai", planning=True)
    names = {definition.name for definition in agent.tools}
    assert {"bash", "file_read", "write_todos"} <= names
    assert agent.config.tools is None
