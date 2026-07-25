"""Tests for the Crew facade over the wavefront scheduler."""

from __future__ import annotations

import pytest

from anycode import Agent, AgentConfig, AnyCode, Crew, CrewError, CrewResult, TaskSpec, tool
from anycode.providers.fake import FakeAdapter, FakeResponse


@pytest.fixture
def fake_adapter(monkeypatch: pytest.MonkeyPatch) -> list[list[dict[str, object]]]:
    """Patch adapter creation so every agent replies deterministically. Returns captured prompts."""
    captured: list[list[dict[str, object]]] = []

    class _RecordingAdapter(FakeAdapter):
        async def chat(self, messages, options):  # type: ignore[no-untyped-def]
            captured.append([message.model_dump() for message in messages])
            return await super().chat(messages, options)

    async def _create(*_args: object, **_kwargs: object) -> FakeAdapter:
        return _RecordingAdapter(responses=[FakeResponse(text="ok")])

    monkeypatch.setattr("anycode.core.agent.create_adapter", _create)
    return captured


def _agent(name: str) -> Agent:
    return Agent(name=name, model="fake-model", provider="openai", instructions=f"You are {name}.", tools=[])


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_crew_requires_agents() -> None:
    with pytest.raises(CrewError, match="at least one agent"):
        Crew(agents=[])


def test_crew_rejects_unknown_process() -> None:
    with pytest.raises(CrewError, match="Unknown process"):
        Crew(agents=[_agent("a")], process="parallel")  # type: ignore[arg-type]


def test_crew_accepts_agents_configs_and_dicts() -> None:
    crew = Crew(
        agents=[
            _agent("built"),
            AgentConfig(name="configured", model="m", provider="openai"),
            {"name": "dictated", "model": "m", "provider": "openai"},
        ]
    )
    assert [config.name for config in crew.team.get_agents()] == ["built", "configured", "dictated"]


def test_crew_rejects_unknown_orchestrator_option() -> None:
    with pytest.raises(CrewError, match="unknown option"):
        Crew(agents=[_agent("a")], nonsense=True)


def test_crew_forwards_known_orchestrator_options() -> None:
    crew = Crew(agents=[_agent("a")], max_handoff_depth=7)
    assert crew.engine._config.max_handoff_depth == 7


def test_repr_summarizes_the_crew() -> None:
    crew = Crew(agents=[_agent("a")], tasks=["do a thing"])
    assert repr(crew) == "Crew(name='crew', agents=1, tasks=1, process='dependency')"


# ---------------------------------------------------------------------------
# Task normalization
# ---------------------------------------------------------------------------


def test_string_task_becomes_a_spec_assigned_to_the_first_agent() -> None:
    crew = Crew(agents=[_agent("lead"), _agent("second")], tasks=["summarize the report"])
    spec = crew.tasks[0]
    assert spec.title == "summarize the report"
    assert spec.description == "summarize the report"
    assert spec.assignee == "lead"


def test_dict_task_becomes_a_spec() -> None:
    crew = Crew(agents=[_agent("lead")], tasks=[{"title": "T", "description": "D", "expected_output": "E"}])
    spec = crew.tasks[0]
    assert (spec.title, spec.description, spec.expected_output) == ("T", "D", "E")


def test_task_spec_accepts_an_agent_object() -> None:
    writer = _agent("writer")
    crew = Crew(agents=[_agent("lead"), writer], tasks=[TaskSpec("Write", "Write it.", agent=writer)])
    assert crew.tasks[0].assignee == "writer"


def test_task_spec_context_is_an_alias_for_depends_on() -> None:
    spec = TaskSpec("Second", "Do it.", context=["First"])
    assert spec.depends_on == ["First"]


def test_sequential_process_chains_tasks() -> None:
    crew = Crew(agents=[_agent("lead")], tasks=["one", "two", "three"], process="sequential")
    assert [spec.depends_on for spec in crew.tasks] == [[], ["one"], ["two"]]


def test_dependency_process_leaves_tasks_independent() -> None:
    crew = Crew(agents=[_agent("lead")], tasks=["one", "two"])
    assert [spec.depends_on for spec in crew.tasks] == [[], []]


def test_explicit_dependency_survives_sequential_process() -> None:
    crew = Crew(
        agents=[_agent("lead")],
        tasks=["one", "two", TaskSpec("three", "do it", depends_on=["one"])],
        process="sequential",
    )
    assert crew.tasks[2].depends_on == ["one"]


def test_unknown_assignee_is_rejected() -> None:
    with pytest.raises(CrewError, match="not in this crew"):
        Crew(agents=[_agent("lead")], tasks=[TaskSpec("T", "D", assignee="ghost")])


def test_unsupported_task_type_is_rejected() -> None:
    with pytest.raises(CrewError, match="Cannot use int as a task"):
        Crew(agents=[_agent("lead")], tasks=[3])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


async def test_crew_runs_dependent_tasks(fake_adapter: list[list[dict[str, object]]]) -> None:
    researcher, writer = _agent("researcher"), _agent("writer")
    crew = Crew(
        agents=[researcher, writer],
        tasks=[
            TaskSpec("Research", "Find sources.", agent=researcher),
            TaskSpec("Write", "Write a brief.", agent=writer, depends_on=["Research"]),
        ],
    )
    result = await crew.run()
    await crew.close()

    assert isinstance(result, CrewResult)
    assert result.success
    assert set(result.outputs) == {"researcher", "writer"}
    assert str(result) == result.output == "ok"


async def test_expected_output_reaches_the_agent_prompt(fake_adapter: list[list[dict[str, object]]]) -> None:
    crew = Crew(
        agents=[_agent("lead")],
        tasks=[TaskSpec("Summarize", "Summarize the notes.", expected_output="Three bullet points.")],
    )
    await crew.run()
    await crew.close()

    prompts = [str(message) for conversation in fake_adapter for message in conversation]
    assert any("Expected output: Three bullet points." in prompt for prompt in prompts)


async def test_autonomous_mode_delegates_to_run_team(fake_adapter: list[list[dict[str, object]]]) -> None:
    crew = Crew(agents=[_agent("lead"), _agent("helper")])
    result = await crew.run("Produce a brief.")
    await crew.close()
    assert result.success
    assert "lead" in result.outputs


async def test_goal_with_declared_tasks_is_rejected(fake_adapter: list[list[dict[str, object]]]) -> None:
    crew = Crew(agents=[_agent("lead")], tasks=["one"])
    with pytest.raises(CrewError, match="takes no goal"):
        await crew.run("a goal")
    await crew.close()


async def test_missing_goal_without_tasks_is_rejected(fake_adapter: list[list[dict[str, object]]]) -> None:
    crew = Crew(agents=[_agent("lead")])
    with pytest.raises(CrewError, match="no tasks"):
        await crew.run()
    await crew.close()


async def test_prebuilt_agents_keep_their_custom_tools(fake_adapter: list[list[dict[str, object]]]) -> None:
    @tool
    def special(value: str) -> str:
        """A tool only this agent has."""
        return value

    specialist = Agent(name="specialist", model="m", provider="openai", tools=[special])
    crew = Crew(agents=[specialist], tasks=["use the tool"])
    await crew.run()
    await crew.close()

    pooled = crew.engine._pool.get("specialist")
    assert pooled is specialist
    assert "special" in {definition.name for definition in specialist.tools}


async def test_injected_engine_is_not_closed_by_the_crew() -> None:
    engine = AnyCode()
    crew = Crew(agents=[_agent("lead")], engine=engine)
    await crew.close()
    assert crew.engine is engine
    await engine.close()


def test_run_sync_inside_a_loop_is_rejected(fake_adapter: list[list[dict[str, object]]]) -> None:
    crew = Crew(agents=[_agent("lead")], tasks=["one"])
    result = crew.run_sync()
    assert result.success


async def test_run_sync_from_async_context_is_rejected(fake_adapter: list[list[dict[str, object]]]) -> None:
    crew = Crew(agents=[_agent("lead")], tasks=["one"])
    with pytest.raises(RuntimeError, match="await crew.run"):
        crew.run_sync()
    await crew.close()


async def test_crew_is_an_async_context_manager(fake_adapter: list[list[dict[str, object]]]) -> None:
    async with Crew(agents=[_agent("lead")], tasks=["one"]) as crew:
        result = await crew.run()
    assert result.success
