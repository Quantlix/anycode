"""Crew — a one-import multi-agent entry point over the wavefront scheduler.

A crew owns an :class:`~anycode.core.orchestrator.AnyCode` engine and a
:class:`~anycode.collaboration.team.Team`; it contributes no scheduling logic of its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from anycode.collaboration.team import Team
from anycode.core.agent import Agent
from anycode.core.orchestrator import AnyCode, TaskSpec
from anycode.helpers.sync_runner import run_coroutine_blocking
from anycode.types import (
    AgentConfig,
    CostReport,
    OrchestratorConfig,
    OrchestratorEvent,
    TeamConfig,
    TeamRunResult,
    TokenUsage,
)

AgentInput = Agent | AgentConfig | dict[str, object]
TaskInput = TaskSpec | dict[str, object] | str
Process = Literal["dependency", "sequential"]


class CrewError(ValueError):
    """Raised when crew construction arguments are invalid."""


class CrewResult(BaseModel):
    """Ergonomic view over a :class:`TeamRunResult`, with the full result still attached."""

    model_config = ConfigDict(frozen=True)

    success: bool
    output: str
    outputs: dict[str, str]
    usage: TokenUsage
    cost: CostReport | None = None
    team_result: TeamRunResult

    def __str__(self) -> str:
        return self.output


class Crew:
    """A team of agents working through a list of tasks."""

    def __init__(
        self,
        agents: Sequence[AgentInput],
        tasks: Sequence[TaskInput] | None = None,
        *,
        name: str = "crew",
        process: Process = "dependency",
        max_concurrency: int | None = None,
        shared_memory: bool = False,
        verbose: bool = False,
        engine: AnyCode | None = None,
        **orchestrator_options: Any,
    ) -> None:
        if not agents:
            raise CrewError("Crew needs at least one agent. Pass agents=[Agent(...), ...].")
        if process not in ("dependency", "sequential"):
            raise CrewError(f'Unknown process "{process}". Use "dependency" or "sequential".')

        self.name = name
        self.process: Process = process
        self._prebuilt = [agent for agent in agents if isinstance(agent, Agent)]
        self._agent_configs = [_as_agent_config(agent) for agent in agents]
        self._specs = _resolve_task_specs(tasks, self._agent_configs, process)
        self._owns_engine = engine is None
        self._engine = engine or AnyCode(_build_orchestrator_config(orchestrator_options, max_concurrency, verbose))

        for agent in self._prebuilt:
            self._engine.register_agent(agent)

        self._team = self._engine.create_team(
            name,
            TeamConfig(
                name=name,
                agents=self._agent_configs,
                shared_memory=shared_memory or None,
                max_concurrency=max_concurrency,
            ),
        )

    @property
    def engine(self) -> AnyCode:
        """The underlying orchestrator, for durability, MCP, plugins, and everything else."""
        return self._engine

    @property
    def team(self) -> Team:
        """The underlying team."""
        return self._team

    @property
    def tasks(self) -> list[TaskSpec]:
        return list(self._specs)

    async def run(self, goal: str | None = None) -> CrewResult:
        """Execute the crew's tasks, or decompose *goal* when no tasks were declared."""
        await self._engine.connect_mcp_servers()
        if self._specs:
            if goal is not None:
                raise CrewError("Crew was built with tasks, so run() takes no goal. Drop the goal or drop the tasks.")
            team_result = await self._engine.run_tasks(self._team, self._specs)
        else:
            if goal is None:
                raise CrewError('Crew has no tasks. Pass tasks=[...] at construction, or a goal: await crew.run("...").')
            team_result = await self._engine.run_team(self._team, goal)
        return _build_crew_result(team_result, self._specs)

    def run_sync(self, goal: str | None = None) -> CrewResult:
        """Blocking form of :meth:`run`. Closes the engine this crew created."""
        return run_coroutine_blocking(
            self._run_and_close(goal),
            sync_call="Crew.run_sync()",
            async_call="await crew.run(...)",
        )

    async def _run_and_close(self, goal: str | None) -> CrewResult:
        try:
            return await self.run(goal)
        finally:
            await self.close()

    async def close(self) -> None:
        """Release engine resources. An engine passed in via ``engine=`` is left alone."""
        if self._owns_engine:
            await self._engine.close()

    async def __aenter__(self) -> Crew:
        await self._engine.connect_mcp_servers()
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        await self.close()

    def __repr__(self) -> str:
        return f"Crew(name={self.name!r}, agents={len(self._agent_configs)}, tasks={len(self._specs)}, process={self.process!r})"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _as_agent_config(agent: AgentInput) -> AgentConfig:
    if isinstance(agent, Agent):
        return agent.config
    if isinstance(agent, AgentConfig):
        return agent
    if isinstance(agent, dict):
        return AgentConfig.model_validate(agent)
    raise CrewError(f"Cannot use {type(agent).__name__} as a crew member. Pass an Agent, an AgentConfig, or a config dict.")


def _as_task_spec(task: TaskInput) -> TaskSpec:
    if isinstance(task, TaskSpec):
        return task
    if isinstance(task, str):
        return TaskSpec(task)
    if isinstance(task, dict):
        return TaskSpec(**task)  # type: ignore[arg-type]
    raise CrewError(f"Cannot use {type(task).__name__} as a task. Pass a TaskSpec, a dict of TaskSpec arguments, or a string title.")


def _resolve_task_specs(
    tasks: Sequence[TaskInput] | None,
    agents: Sequence[AgentConfig],
    process: Process,
) -> list[TaskSpec]:
    if not tasks:
        return []

    specs = [_as_task_spec(task) for task in tasks]
    known = {config.name for config in agents}
    default_assignee = agents[0].name

    previous_title: str | None = None
    for spec in specs:
        if spec.assignee is None:
            spec.assignee = default_assignee
        elif spec.assignee not in known:
            raise CrewError(f'Task "{spec.title}" is assigned to "{spec.assignee}", who is not in this crew. Members: {", ".join(sorted(known))}.')
        if process == "sequential" and not spec.depends_on and previous_title is not None:
            spec.depends_on = [previous_title]
        previous_title = spec.title

    return specs


def _build_orchestrator_config(options: dict[str, Any], max_concurrency: int | None, verbose: bool) -> OrchestratorConfig:
    unknown = sorted(set(options) - set(OrchestratorConfig.model_fields))
    if unknown:
        valid = ", ".join(sorted(OrchestratorConfig.model_fields))
        raise CrewError(f"Crew received unknown option(s): {', '.join(unknown)}. Valid orchestrator options are: {valid}.")

    settings = dict(options)
    if max_concurrency is not None:
        settings.setdefault("max_concurrency", max_concurrency)
    if verbose and settings.get("on_progress") is None:
        settings["on_progress"] = _print_progress
    return OrchestratorConfig.model_validate(settings)


def _print_progress(event: OrchestratorEvent) -> None:
    if event.type == "agent_start":
        print(f"[crew] {event.agent} started")
    elif event.type == "agent_complete":
        print(f"[crew] {event.agent} finished")
    elif event.type == "task_complete":
        print(f"[crew] task {event.task} complete")
    elif event.type == "error":
        print(f"[crew] error from {event.agent or 'crew'}: {event.data}")


def _build_crew_result(team_result: TeamRunResult, specs: Sequence[TaskSpec]) -> CrewResult:
    outputs = {name: result.output for name, result in team_result.agent_results.items()}

    output = ""
    if specs:
        for spec in reversed(specs):
            if spec.assignee and spec.assignee in outputs:
                output = outputs[spec.assignee]
                break
    if not output and outputs:
        output = next(reversed(outputs.values()))

    return CrewResult(
        success=team_result.success,
        output=output,
        outputs=outputs,
        usage=team_result.total_token_usage,
        cost=team_result.cost_report,
        team_result=team_result,
    )
