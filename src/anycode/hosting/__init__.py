"""Framework-neutral managed-host integration primitives."""

from anycode.hosting.agent_card import (
    A2A_AGENT_CARD_PATH,
    A2A_PROTOCOL_VERSION,
    A2AAgentCard,
    AgentCapabilities,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    build_deployment_agent_card,
)
from anycode.hosting.lifecycle import DrainResult, HostAdmissionResult, HostLifecycle, HostLifecycleSnapshot

__all__ = [
    "A2A_AGENT_CARD_PATH",
    "A2A_PROTOCOL_VERSION",
    "A2AAgentCard",
    "AgentCapabilities",
    "AgentInterface",
    "AgentProvider",
    "AgentSkill",
    "DrainResult",
    "HostAdmissionResult",
    "HostLifecycle",
    "HostLifecycleSnapshot",
    "build_deployment_agent_card",
]
