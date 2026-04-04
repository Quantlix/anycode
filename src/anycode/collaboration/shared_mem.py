"""Namespaced shared memory for coordinated agent teams."""

from __future__ import annotations

from typing import Any

from anycode.collaboration.kv_store import InMemoryStore
from anycode.types import MemoryEntry, MemoryStore


class SharedMemory:
    """Keys are scoped as <agent_name>/<key> to avoid collisions."""

    def __init__(self) -> None:
        self._store = InMemoryStore()

    async def write(self, agent_name: str, key: str, value: str, metadata: dict[str, Any] | None = None) -> None:
        namespaced = f"{agent_name}/{key}"
        await self._store.set(namespaced, value, {**(metadata or {}), "agent": agent_name})

    async def read(self, key: str) -> MemoryEntry | None:
        return await self._store.get(key)

    async def list_all(self) -> list[MemoryEntry]:
        return await self._store.list()

    async def list_by_agent(self, agent_name: str) -> list[MemoryEntry]:
        prefix = f"{agent_name}/"
        return [e for e in await self._store.list() if e.key.startswith(prefix)]

    async def get_summary(self) -> str:
        """Build a markdown summary grouped by agent."""
        all_entries = await self._store.list()
        if not all_entries:
            return ""

        by_agent: dict[str, list[tuple[str, str]]] = {}
        for entry in all_entries:
            slash_idx = entry.key.find("/")
            agent = entry.key[:slash_idx] if slash_idx != -1 else "_unknown"
            local_key = entry.key[slash_idx + 1 :] if slash_idx != -1 else entry.key
            by_agent.setdefault(agent, []).append((local_key, entry.value))

        lines = ["## Collective Agent Knowledge", ""]
        for agent, entries in by_agent.items():
            lines.append(f"### {agent}")
            for local_key, value in entries:
                display = f"{value[:177]}\u2026" if len(value) > 180 else value
                lines.append(f"- {local_key}: {display}")
            lines.append("")

        return "\n".join(lines).rstrip()

    def get_store(self) -> MemoryStore:
        return self._store  # type: ignore[return-value]
