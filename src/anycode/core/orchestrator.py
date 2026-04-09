"""AnyCode — top-level coordinator for agents, teams, and task execution pipelines."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any

from pydantic import BaseModel

from anycode.checkpoint.manager import CheckpointManager
from anycode.collaboration.team import Team
from anycode.constants import (
    COORDINATOR_ROLE_PREVIEW_LENGTH,
    DEFAULT_MAX_CONCURRENCY,
    DEPENDENCY_CONTEXT_MAX_LENGTH,
    MCP_TOOL_PREFIX,
    ORCH_EVENT_AGENT_COMPLETE,
    ORCH_EVENT_AGENT_START,
    ORCH_EVENT_ERROR,
    ORCH_EVENT_TASK_COMPLETE,
    ORCH_EVENT_TASK_START,
)
from anycode.core.agent import Agent
from anycode.core.pool import AgentPool
from anycode.core.scheduler import Scheduler
from anycode.handoff.executor import HandoffExecutor
from anycode.handoff.tool import HANDOFF_TOOL_DEF
from anycode.helpers.usage_tracker import EMPTY_USAGE, merge_usage
from anycode.hitl.approval import ApprovalManager
from anycode.mcp.bridge import discover_and_register
from anycode.mcp.client import MCPClient
from anycode.routing.router import DefaultRouter
from anycode.tasks.queue import TaskQueue
from anycode.tasks.task import create_task, get_task_dependency_order, validate_task_dependencies
from anycode.telemetry.tracer import Tracer
from anycode.tools.built_in import register_built_in_tools
from anycode.tools.executor import ToolExecutor
from anycode.tools.registry import ToolRegistry
from anycode.types import (
    AgentConfig,
    AgentRunResult,
    GuardrailConfig,
    OrchestratorConfig,
    OrchestratorEvent,
    OutputValidator,
    Task,
    TeamConfig,
    TeamRunResult,
    TokenUsage,
    TraceConfig,
    TurnHook,
)

logger = logging.getLogger(__name__)


class TaskSpec:
    def __init__(self, title: str, description: str, assignee: str | None = None, depends_on: list[str] | None = None) -> None:
        self.title = title
        self.description = description
        self.assignee = assignee
        self.depends_on = depends_on or []


class AnyCode:
    """Top-level orchestration engine for agents, teams, and task pipelines."""

    def __init__(self, config: OrchestratorConfig | dict[str, object] | None = None) -> None:
        self._config = OrchestratorConfig.model_validate(config) if isinstance(config, dict) else (config or OrchestratorConfig())
        self._pool = AgentPool(self._config.max_concurrency or DEFAULT_MAX_CONCURRENCY)
        self._scheduler = Scheduler("dependency-first")
        self._teams: dict[str, Team] = {}
        self._trace_config: TraceConfig | None = None
        self._guardrail_config: GuardrailConfig | None = None
        self._hooks: list[TurnHook] | None = None
        self._output_validators: list[OutputValidator] | None = None
        self._tracer: Tracer | None = None
        self._checkpoint_manager: CheckpointManager | None = None
        self._approval_manager: ApprovalManager | None = None

        # MCP clients for external tool servers
        self._mcp_clients: dict[str, Any] = {}
        self._mcp_tool_registry: ToolRegistry = ToolRegistry()

        # Agent-to-agent handoff orchestration
        self._handoff_executor = HandoffExecutor(max_depth=self._config.max_handoff_depth)

        # Intelligent task routing
        self._router: DefaultRouter | None = None
        if self._config.routing and self._config.routing.enabled:
            self._router = DefaultRouter(self._config.routing)

        if self._config.checkpoint and self._config.checkpoint.enabled:
            self._checkpoint_manager = CheckpointManager(self._config.checkpoint)
        if self._config.approval and self._config.approval.enabled and self._config.approval_handler:
            self._approval_manager = ApprovalManager(self._config.approval, self._config.approval_handler)

    def configure(
        self,
        *,
        trace: TraceConfig | None = None,
        guardrails: GuardrailConfig | None = None,
        hooks: list[TurnHook] | None = None,
        output_validators: list[OutputValidator] | None = None,
    ) -> None:
        """Set telemetry, guardrails, hooks, and output validators for all agents."""
        if trace is not None:
            self._trace_config = trace
            self._tracer = Tracer(trace)
        if guardrails is not None:
            self._guardrail_config = guardrails
        if hooks is not None:
            self._hooks = hooks
        if output_validators is not None:
            self._output_validators = output_validators

    # -- MCP lifecycle --

    async def connect_mcp_servers(self) -> None:
        """Connect to all configured MCP servers and register their tools."""
        if not self._config.mcp_servers:
            return

        for server_config in self._config.mcp_servers:
            try:
                client = MCPClient(server_config)
                await client.connect()
                tools = await discover_and_register(client, server_config.name, self._mcp_tool_registry)
                self._mcp_clients[server_config.name] = client
                logger.info("MCP server '%s': registered %d tools", server_config.name, len(tools))
            except Exception as e:
                logger.error("Failed to connect MCP server '%s': %s", server_config.name, e)

    async def disconnect_mcp_servers(self) -> None:
        """Disconnect from all MCP servers and clean up tools."""
        for name, client in self._mcp_clients.items():
            try:
                if hasattr(client, "disconnect"):
                    await client.disconnect()
            except Exception as e:
                logger.warning("Error disconnecting MCP server '%s': %s", name, e)
        self._mcp_clients.clear()
        self._mcp_tool_registry = ToolRegistry()

    async def __aenter__(self) -> AnyCode:
        await self.connect_mcp_servers()
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        await self.disconnect_mcp_servers()

    def build_agent(
        self,
        config: AgentConfig | dict[str, object],
        *,
        output_schema: type[BaseModel] | None = None,
    ) -> Agent:
        """Construct a fully wired Agent instance with all default tools."""
        typed_config = AgentConfig.model_validate(config) if isinstance(config, dict) else config
        registry = ToolRegistry()
        register_built_in_tools(registry)

        # Register handoff tool if agent opts in
        if typed_config.tools and "handoff" in typed_config.tools:
            if not registry.has(HANDOFF_TOOL_DEF.name):
                registry.register(HANDOFF_TOOL_DEF)

        # Register MCP tools for this agent
        if self._mcp_clients:
            mcp_filter = set(typed_config.mcp_servers) if typed_config.mcp_servers else None
            for tool in self._mcp_tool_registry.list():
                if mcp_filter is not None:
                    # Only include tools from servers the agent has access to
                    server_prefix = f"{MCP_TOOL_PREFIX}_"
                    if not any(tool.name.startswith(f"{server_prefix}{s.replace('-', '_').replace('.', '_')}_") for s in mcp_filter):
                        continue
                if not registry.has(tool.name):
                    registry.register(tool)

        executor = ToolExecutor(registry)
        return Agent(
            typed_config,
            registry,
            executor,
            tracer=self._tracer,
            guardrail_config=self._guardrail_config,
            hooks=self._hooks,
            output_validators=self._output_validators,
            output_schema=output_schema,
        )

    def create_team(self, name: str, config: TeamConfig) -> Team:
        """Instantiate a team and enrol its agents into the shared pool."""
        team = Team(config)
        self._teams[name] = team
        for agent_cfg in config.agents:
            if not self._pool.get(agent_cfg.name):
                self._pool.add(self.build_agent(agent_cfg))
        return team

    async def run_agent(
        self,
        config: AgentConfig | dict[str, object],
        prompt: str,
        *,
        output_schema: type[BaseModel] | None = None,
    ) -> AgentRunResult:
        """Create and run a single agent on one prompt."""
        typed_config = AgentConfig.model_validate(config) if isinstance(config, dict) else config

        root_span = None
        if self._tracer and self._tracer.enabled:
            root_span = self._tracer.start_span(f"anycode.run_agent.{typed_config.name}")

        agent = self.build_agent(typed_config, output_schema=output_schema)
        self._emit(OrchestratorEvent(type=ORCH_EVENT_AGENT_START, agent=typed_config.name))
        try:
            result = await agent.run(prompt)
            self._emit(OrchestratorEvent(type=ORCH_EVENT_AGENT_COMPLETE, agent=typed_config.name, data=result))
            return result
        except Exception as e:
            self._emit(OrchestratorEvent(type=ORCH_EVENT_ERROR, agent=typed_config.name, data=e))
            return AgentRunResult(success=False, output=str(e), messages=[], token_usage=EMPTY_USAGE, tool_calls=[])
        finally:
            if root_span and self._tracer:
                self._tracer.end_span(root_span)

    async def run_team(self, team: Team, goal: str) -> TeamRunResult:
        """Decompose a high-level goal into tasks using a coordinator agent, then execute."""
        root_span = None
        if self._tracer and self._tracer.enabled:
            root_span = self._tracer.start_span(f"anycode.run_team.{team.name}")

        try:
            agents = team.get_agents()
            if not agents:
                return TeamRunResult(success=False, agent_results={}, total_token_usage=EMPTY_USAGE)

            coordinator_cfg = agents[0]
            coordinator = self.build_agent(coordinator_cfg)
            coordinator_prompt = self._build_coordinator_prompt(agents, goal)

            self._emit(OrchestratorEvent(type=ORCH_EVENT_AGENT_START, agent=coordinator_cfg.name))
            plan_result = await coordinator.run(coordinator_prompt)
            self._emit(OrchestratorEvent(type=ORCH_EVENT_AGENT_COMPLETE, agent=coordinator_cfg.name, data=plan_result))

            if not plan_result.success:
                return TeamRunResult(
                    success=False,
                    agent_results={coordinator_cfg.name: plan_result},
                    total_token_usage=plan_result.token_usage,
                )

            task_specs = self._parse_task_specs(plan_result.output, agents)
            if not task_specs:
                return TeamRunResult(
                    success=True,
                    agent_results={coordinator_cfg.name: plan_result},
                    total_token_usage=plan_result.token_usage,
                )

            task_results = await self._execute_tasks(team, task_specs)
            combined = {coordinator_cfg.name: plan_result, **task_results.agent_results}
            total = plan_result.token_usage
            for r in task_results.agent_results.values():
                total = merge_usage(total, r.token_usage)

            return TeamRunResult(success=task_results.success, agent_results=combined, total_token_usage=total)
        finally:
            if root_span and self._tracer:
                self._tracer.end_span(root_span)

    async def run_tasks(self, team: Team, task_specs: list[TaskSpec], *, resume_from: str | None = None) -> TeamRunResult:
        """Run an explicit set of task specs with full dependency resolution."""
        return await self._execute_tasks(team, task_specs, resume_from=resume_from)

    async def _execute_tasks(self, team: Team, specs: list[TaskSpec], *, resume_from: str | None = None) -> TeamRunResult:
        resolved = self._resolve_specs(specs)
        queue = TaskQueue()
        agent_results: dict[str, AgentRunResult] = {}
        total_usage: TokenUsage = EMPTY_USAGE
        all_succeeded = True
        start_wave = 0

        workflow_id = self._compute_workflow_id(resolved)

        if resume_from and self._checkpoint_manager:
            resolved, agent_results, total_usage, start_wave = await self._resume_from_checkpoint(workflow_id, resume_from, resolved)

        for task in resolved:
            queue.add(task)

        self._scheduler.auto_assign(queue, team.get_agents())
        ordered = get_task_dependency_order(queue.list())
        waves = self._build_waves(ordered, queue)

        # Intelligent routing — apply after scheduling, before execution
        route_decisions: dict[str, object] = {}
        if self._router:
            agents = team.get_agents()
            for task in ordered:
                decision = await self._router.route(task, agents)
                if decision:
                    route_decisions[task.id] = decision
                    self._emit(OrchestratorEvent(type=ORCH_EVENT_TASK_START, data={"routing": decision.model_dump()}))

        for wave_idx, wave in enumerate(waves):
            if wave_idx < start_wave:
                continue

            outcomes = await asyncio.gather(*[self._run_wave_task(t, queue, team) for t in wave])
            for outcome in outcomes:
                if outcome is None:
                    all_succeeded = False
                    continue
                assignee, result = outcome
                agent_results[assignee] = result
                total_usage = merge_usage(total_usage, result.token_usage)
                if not result.success:
                    all_succeeded = False

            # Save checkpoint after each wave
            if self._checkpoint_manager:
                await self._checkpoint_manager.auto_save(
                    workflow_id=workflow_id,
                    tasks=queue.list(),
                    agent_results=agent_results,
                    wave_index=wave_idx,
                    total_usage=total_usage,
                )

        return TeamRunResult(success=all_succeeded, agent_results=agent_results, total_token_usage=total_usage)

    async def _resume_from_checkpoint(
        self,
        workflow_id: str,
        resume_from: str,
        resolved: list[Task],
    ) -> tuple[list[Task], dict[str, AgentRunResult], TokenUsage, int]:
        """Resume execution from a checkpoint. Returns (tasks, results, usage, start_wave)."""
        assert self._checkpoint_manager is not None
        checkpoint = None
        if resume_from == "latest":
            checkpoint = await self._checkpoint_manager.load_latest(workflow_id)
        else:
            checkpoint = await self._checkpoint_manager.store.load(resume_from)

        if not checkpoint:
            return resolved, {}, EMPTY_USAGE, 0

        if self._checkpoint_manager.detect_spec_change(resolved, checkpoint):
            raise ValueError("AnyCode: task specs changed since checkpoint was created — cannot resume.")

        self._emit(OrchestratorEvent(type=ORCH_EVENT_TASK_START, data={"checkpoint_resume": checkpoint.id, "wave": checkpoint.wave_index + 1}))
        return checkpoint.tasks, dict(checkpoint.agent_results), checkpoint.total_token_usage, checkpoint.wave_index + 1

    async def _run_wave_task(
        self,
        task: Task,
        queue: TaskQueue,
        team: Team,
    ) -> tuple[str, AgentRunResult] | None:
        """Execute a single task within a wave. Returns (assignee, result) or None if unassigned."""
        assignee = task.assignee
        if not assignee:
            queue.fail(task.id, "Unassigned task — no agent available.")
            return None

        if self._approval_manager:
            response = await self._approval_manager.check_and_request(
                request_type="task",
                agent=assignee,
                description=f"Execute task: {task.title}",
                context={"task_id": task.id, "title": task.title, "description": task.description},
            )
            if response and not response.approved:
                reason = f"Approval denied: {response.reason or 'rejected'}"
                queue.fail(task.id, reason)
                return (
                    assignee,
                    AgentRunResult(
                        success=False,
                        output=reason,
                        messages=[],
                        token_usage=EMPTY_USAGE,
                        tool_calls=[],
                    ),
                )

        self._emit(OrchestratorEvent(type=ORCH_EVENT_TASK_START, task=task.id, agent=assignee, data=task))
        queue.update(task.id, status="in_progress")

        prompt = self._build_task_prompt(task, queue)
        agent = self._pool.get(assignee) or self._build_agent_for_team(assignee, team)

        self._emit(OrchestratorEvent(type=ORCH_EVENT_AGENT_START, agent=assignee))
        try:
            result = await agent.run(prompt)
            self._emit(OrchestratorEvent(type=ORCH_EVENT_AGENT_COMPLETE, agent=assignee, data=result))
            if result.success:
                queue.complete(task.id, result.output)
                self._emit(OrchestratorEvent(type=ORCH_EVENT_TASK_COMPLETE, task=task.id, agent=assignee))
            else:
                queue.fail(task.id, result.output)
            return (assignee, result)
        except Exception as e:
            queue.fail(task.id, str(e))
            return (assignee, AgentRunResult(success=False, output=str(e), messages=[], token_usage=EMPTY_USAGE, tool_calls=[]))

    def _build_agent_for_team(self, agent_name: str, team: Team) -> Agent:
        config = team.get_agent(agent_name)
        if not config:
            raise ValueError(f'AnyCode: "{agent_name}" is not part of team "{team.name}".')
        agent = self.build_agent(config)
        try:
            self._pool.add(agent)
        except ValueError:
            pass
        return agent

    def _resolve_specs(self, specs: list[TaskSpec]) -> list[Task]:
        title_map: dict[str, str] = {}
        pending: list[tuple[Task, list[str]]] = []

        for spec in specs:
            task = create_task(title=spec.title, description=spec.description, assignee=spec.assignee)
            title_map[spec.title] = task.id
            pending.append((task, spec.depends_on))

        resolved: list[Task] = []
        for task, raw_deps in pending:
            dep_ids = [title_map[d] for d in raw_deps if d in title_map]
            final = task.model_copy(update={"depends_on": dep_ids}) if dep_ids else task
            resolved.append(final)

        valid, errors = validate_task_dependencies(resolved)
        if not valid:
            raise ValueError("AnyCode: task dependency errors:\n" + "\n".join(errors))

        return resolved

    def _build_waves(self, ordered: list[Task], queue: TaskQueue) -> list[list[Task]]:
        if not ordered:
            return []
        done: set[str] = set()
        remaining = list(ordered)
        waves: list[list[Task]] = []

        while remaining:
            wave = [t for t in remaining if all(d in done for d in (t.depends_on or []))]
            if not wave:
                for t in remaining:
                    queue.fail(t.id, "Unresolvable or circular dependency.")
                break
            waves.append(wave)
            done.update(t.id for t in wave)
            remaining = [t for t in remaining if t.id not in done]

        return waves

    def _build_coordinator_prompt(self, agents: list[AgentConfig], goal: str) -> str:
        roster = "\n".join(f"- {a.name}: {(a.system_prompt or 'general-purpose assistant')[:COORDINATOR_ROLE_PREVIEW_LENGTH]}" for a in agents)
        return (
            f"You are the lead coordinator of a team. Here are your team members:\n{roster}\n\n"
            f"Objective: {goal}\n\n"
            "If you can accomplish this objective entirely by yourself with the tools you have, go ahead and do so now.\n"
            "Otherwise, decompose the objective into concrete tasks and list them in the following format (one per line):\n\n"
            "ASSIGN: <agent_name> | <task_title> | <task_description>\n\n"
            "Only output ASSIGN: lines when you need to delegate. Pick the best-suited team member for each task.\n"
            "List tasks in execution order — later tasks may depend on earlier ones."
        )

    def _build_task_prompt(self, task: Task, queue: TaskQueue) -> str:
        dep_context_parts = []
        for dep_id in task.depends_on or []:
            dep_task = next((t for t in queue.list() if t.id == dep_id), None)
            if dep_task and dep_task.result:
                dep_context_parts.append(f"[{dep_task.title}]: {dep_task.result[:DEPENDENCY_CONTEXT_MAX_LENGTH]}")

        context = "\n\nRelevant output from prerequisite tasks:\n" + "\n\n".join(dep_context_parts) if dep_context_parts else ""
        return f"Task: {task.title}\n\n{task.description}{context}"

    def _parse_task_specs(self, output: str, agents: list[AgentConfig]) -> list[TaskSpec]:
        lines = [ln for ln in output.split("\n") if ln.strip().upper().startswith("ASSIGN:")]
        if not lines:
            return []

        valid_names = {a.name for a in agents}
        specs: list[TaskSpec] = []
        titles: list[str] = []

        for line in lines:
            body = line.split(":", 1)[1].strip() if ":" in line else ""
            segments = [s.strip() for s in body.split("|")]
            if len(segments) < 3:
                continue
            assignee, title, *desc_parts = segments
            description = " | ".join(desc_parts)
            if assignee not in valid_names:
                continue
            depends_on = [titles[-1]] if titles else None
            specs.append(TaskSpec(title=title, description=description, assignee=assignee, depends_on=depends_on))
            titles.append(title)

        return specs

    def _emit(self, event: OrchestratorEvent) -> None:
        if self._config.on_progress:
            self._config.on_progress(event)

    @staticmethod
    def _compute_workflow_id(tasks: list[Task]) -> str:
        content = "|".join(f"{t.title}:{t.description}" for t in sorted(tasks, key=lambda t: t.title))
        return hashlib.sha256(content.encode()).hexdigest()[:16]
