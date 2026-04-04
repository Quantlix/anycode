"""Agent registry with semaphore-bounded concurrency and round-robin dispatch."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from anycode.helpers.concurrency_gate import Semaphore
from anycode.helpers.usage_tracker import EMPTY_USAGE
from anycode.types import AgentRunResult, PoolStatus

if TYPE_CHECKING:
    from anycode.core.agent import Agent


class AgentPool:
    """Manages named agents with bounded concurrency."""

    def __init__(self, max_concurrency: int = 5) -> None:
        self._agents: dict[str, Agent] = {}
        self._semaphore = Semaphore(max_concurrency)
        self._rr_index = 0

    def add(self, agent: Agent) -> None:
        if agent.name in self._agents:
            raise ValueError(f'Pool: agent "{agent.name}" already exists in the registry.')
        self._agents[agent.name] = agent

    def remove(self, name: str) -> None:
        if name not in self._agents:
            raise ValueError(f'Pool: no agent named "{name}" is registered.')
        del self._agents[name]

    def get(self, name: str) -> Agent | None:
        return self._agents.get(name)

    def list(self) -> list[Agent]:
        return list(self._agents.values())

    async def run(self, agent_name: str, prompt: str) -> AgentRunResult:
        agent = self._require(agent_name)
        await self._semaphore.acquire()
        try:
            return await agent.run(prompt)
        finally:
            self._semaphore.release()

    async def run_parallel(self, jobs: list[dict[str, str]]) -> dict[str, AgentRunResult]:
        results: dict[str, AgentRunResult] = {}

        async def _job(agent_name: str, prompt: str) -> None:
            try:
                results[agent_name] = await self.run(agent_name, prompt)
            except Exception as e:
                results[agent_name] = AgentRunResult(success=False, output=str(e), messages=[], token_usage=EMPTY_USAGE, tool_calls=[])

        await asyncio.gather(*[_job(j["agent"], j["prompt"]) for j in jobs])
        return results

    async def run_any(self, prompt: str) -> AgentRunResult:
        agents = self.list()
        if not agents:
            raise ValueError("Pool: run_any() requires at least one registered agent.")
        self._rr_index = self._rr_index % len(agents)
        agent = agents[self._rr_index]
        self._rr_index = (self._rr_index + 1) % len(agents)

        await self._semaphore.acquire()
        try:
            return await agent.run(prompt)
        finally:
            self._semaphore.release()

    def get_status(self) -> PoolStatus:
        idle = running = completed = error = 0
        for agent in self._agents.values():
            s = agent.get_state().status
            if s == "idle":
                idle += 1
            elif s == "running":
                running += 1
            elif s == "completed":
                completed += 1
            elif s == "error":
                error += 1
        return PoolStatus(total=len(self._agents), idle=idle, running=running, completed=completed, error=error)

    async def shutdown(self) -> None:
        for agent in self._agents.values():
            agent.reset()

    def _require(self, name: str) -> Agent:
        agent = self._agents.get(name)
        if agent is None:
            available = ", ".join(self._agents.keys())
            raise ValueError(f'Pool: "{name}" not found. Available agents: [{available}]')
        return agent
