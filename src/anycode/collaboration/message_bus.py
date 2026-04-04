"""Agent-to-agent message bus with pub/sub and read-state tracking."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import uuid4

from anycode.types import Message


class MessageBus:
    """Pub/sub messaging between agents."""

    def __init__(self) -> None:
        self._messages: list[Message] = []
        self._read_state: dict[str, set[str]] = {}
        self._subscribers: dict[str, dict[int, Callable[[Message], None]]] = {}
        self._next_id = 0

    def send(self, from_agent: str, to_agent: str, content: str) -> Message:
        msg = Message(id=str(uuid4()), from_agent=from_agent, to_agent=to_agent, content=content, timestamp=datetime.now(UTC))
        self._dispatch(msg)
        return msg

    def broadcast(self, from_agent: str, content: str) -> Message:
        return self.send(from_agent, "*", content)

    def get_unread(self, agent_name: str) -> list[Message]:
        read = self._read_state.get(agent_name, set())
        return [m for m in self._messages if _is_recipient(m, agent_name) and m.id not in read]

    def get_all(self, agent_name: str) -> list[Message]:
        return [m for m in self._messages if _is_recipient(m, agent_name)]

    def mark_read(self, agent_name: str, message_ids: list[str]) -> None:
        if not message_ids:
            return
        read = self._read_state.setdefault(agent_name, set())
        read.update(message_ids)

    def get_conversation(self, agent1: str, agent2: str) -> list[Message]:
        return [m for m in self._messages if (m.from_agent == agent1 and m.to_agent == agent2) or (m.from_agent == agent2 and m.to_agent == agent1)]

    def subscribe(self, agent_name: str, callback: Callable[[Message], None]) -> Callable[[], None]:
        subs = self._subscribers.setdefault(agent_name, {})
        sub_id = self._next_id
        self._next_id += 1
        subs[sub_id] = callback
        def _unsub() -> None:
            subs.pop(sub_id, None)

        return _unsub

    def _dispatch(self, message: Message) -> None:
        self._messages.append(message)
        if message.to_agent == "*":
            for agent_name, subs in self._subscribers.items():
                if agent_name != message.from_agent:
                    for cb in subs.values():
                        cb(message)
        else:
            for cb in self._subscribers.get(message.to_agent, {}).values():
                cb(message)


def _is_recipient(message: Message, agent_name: str) -> bool:
    if message.to_agent == "*":
        return message.from_agent != agent_name
    return message.to_agent == agent_name
