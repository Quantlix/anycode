"""Read-only harness component registry.

The registry takes an :class:`OrchestratorConfig` (or any combination of
``AgentConfig`` / ``ToolRegistry`` / ``VerificationSensorConfig`` instances) and
materialises every editable artifact as a :class:`HarnessComponent`. Manifest
generation is intentionally side-effect-free and safe to run in production: it
only inspects in-memory configuration and does not mutate any state.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

from anycode.harness.component import (
    REDACTED_CHECKSUM_MARKER,
    compute_checksum,
    make_component,
)
from anycode.tools.registry import ToolRegistry
from anycode.types import (
    AgentConfig,
    ContextPolicy,
    HarnessComponent,
    HarnessComponentKind,
    HarnessManifest,
    OrchestratorConfig,
    RoutingConfig,
    TeamConfig,
    ToolDefinition,
    VerificationSensorConfig,
)


class HarnessRegistry:
    """In-memory store of :class:`HarnessComponent` records.

    The registry is append-only and indexed by component id. Components must have
    unique ids; collisions raise ``ValueError`` so that drift detection has a
    stable basis. Registration order is preserved for deterministic manifests.
    """

    def __init__(self) -> None:
        self._components: dict[str, HarnessComponent] = {}
        self._order: list[str] = []

    def register(self, component: HarnessComponent) -> None:
        if component.id in self._components:
            raise ValueError(f"HarnessRegistry: component '{component.id}' is already registered")
        self._components[component.id] = component
        self._order.append(component.id)

    def get(self, component_id: str) -> HarnessComponent | None:
        return self._components.get(component_id)

    def list(self, *, kind: HarnessComponentKind | None = None, editable: bool | None = None) -> list[HarnessComponent]:
        components = [self._components[cid] for cid in self._order]
        if kind is not None:
            components = [c for c in components if c.kind == kind]
        if editable is not None:
            components = [c for c in components if c.editable is editable]
        return components

    def __contains__(self, component_id: object) -> bool:
        return isinstance(component_id, str) and component_id in self._components

    def __len__(self) -> int:
        return len(self._components)


def register_component(registry: HarnessRegistry, component: HarnessComponent) -> HarnessComponent:
    """Add *component* to *registry* and return it for fluent registration."""

    registry.register(component)
    return component


def _agent_payload(agent: AgentConfig) -> dict[str, object]:
    return {
        "name": agent.name,
        "provider": agent.provider,
        "model": agent.model,
        "system_prompt": agent.system_prompt,
        "tools": list(agent.tools or ()),
        "max_turns": agent.max_turns,
        "max_tokens": agent.max_tokens,
        "temperature": agent.temperature,
    }


def _context_policy_payload(policy: ContextPolicy) -> dict[str, object]:
    return {
        "enabled": policy.enabled,
        "mode": policy.mode,
        "max_context_tokens": policy.max_context_tokens,
        "trim_ratio": policy.trim_ratio,
        "offload_ratio": policy.offload_ratio,
        "compact_ratio": policy.compact_ratio,
        "handoff_ratio": policy.handoff_ratio,
        "reserved_response_tokens": policy.reserved_response_tokens,
        "sections": {kind: budget.model_dump() for kind, budget in policy.sections.items()},
    }


def _routing_payload(routing: RoutingConfig) -> dict[str, object]:
    return {
        "enabled": routing.enabled,
        "default_model": routing.default_model,
        "default_provider": routing.default_provider,
        "rules": [rule.model_dump() for rule in routing.rules or ()],
    }


def _sensor_payload(sensor: VerificationSensorConfig) -> dict[str, object]:
    return {
        "name": sensor.name,
        "kind": sensor.kind,
        "phases": list(sensor.phases),
        "block_on_failure": sensor.block_on_failure,
        "retry_on_failure": sensor.retry_on_failure,
        "options": dict(sensor.options),
    }


def _tool_payload(tool: ToolDefinition) -> dict[str, object]:
    schema = tool.input_model.model_json_schema()
    schema.pop("title", None)
    return {"name": tool.name, "description": tool.description, "input_schema": schema}


def _register_agent_components(registry: HarnessRegistry, agent: AgentConfig) -> None:
    if agent.system_prompt:
        register_component(
            registry,
            make_component(
                id=f"prompt:agent:{agent.name}",
                kind="prompt",
                source=f"AgentConfig({agent.name}).system_prompt",
                owner="config",
                description=f"System prompt for agent '{agent.name}'",
                payload={"system_prompt": agent.system_prompt},
            ),
        )
    register_component(
        registry,
        make_component(
            id=f"provider:agent:{agent.name}",
            kind="provider",
            source=f"AgentConfig({agent.name})",
            owner="config",
            description=f"Provider/model profile for '{agent.name}'",
            payload=_agent_payload(agent),
            metadata={"agent": agent.name},
        ),
    )
    if agent.context_policy is not None:
        register_component(
            registry,
            make_component(
                id=f"context_policy:agent:{agent.name}",
                kind="context_policy",
                source=f"AgentConfig({agent.name}).context_policy",
                owner="config",
                description=f"Context policy for agent '{agent.name}'",
                payload=_context_policy_payload(agent.context_policy),
            ),
        )
    for sensor in agent.verification:
        register_component(
            registry,
            make_component(
                id=f"verification:agent:{agent.name}:{sensor.name}",
                kind="verification",
                source=f"AgentConfig({agent.name}).verification[{sensor.name}]",
                owner="config",
                description=f"Verification sensor '{sensor.name}' for '{agent.name}'",
                payload=_sensor_payload(sensor),
            ),
        )


def _register_tool_components(registry: HarnessRegistry, tools: Iterable[ToolDefinition]) -> None:
    for tool in tools:
        register_component(
            registry,
            make_component(
                id=f"tool:{tool.name}",
                kind="tool",
                source=f"ToolDefinition({tool.name})",
                owner="core",
                description=tool.description,
                editable=False,
                payload=_tool_payload(tool),
            ),
        )


def _register_orchestrator_components(registry: HarnessRegistry, orchestrator: OrchestratorConfig) -> None:
    if orchestrator.routing is not None:
        register_component(
            registry,
            make_component(
                id="routing_policy:global",
                kind="routing_policy",
                source="OrchestratorConfig.routing",
                owner="config",
                description="Global routing rules and defaults",
                payload=_routing_payload(orchestrator.routing),
            ),
        )
    for sensor in orchestrator.verification:
        register_component(
            registry,
            make_component(
                id=f"verification:global:{sensor.name}",
                kind="verification",
                source=f"OrchestratorConfig.verification[{sensor.name}]",
                owner="config",
                description=f"Global verification sensor '{sensor.name}'",
                payload=_sensor_payload(sensor),
            ),
        )
    if orchestrator.rag is not None and orchestrator.rag.enabled:
        register_component(
            registry,
            make_component(
                id="memory:rag",
                kind="memory",
                source="OrchestratorConfig.rag",
                owner="config",
                description="RAG retrieval policy",
                payload=orchestrator.rag.model_dump(),
            ),
        )


def build_default_registry(
    *,
    team: TeamConfig | None = None,
    orchestrator: OrchestratorConfig | None = None,
    tools: Sequence[ToolDefinition] | ToolRegistry | None = None,
    agents: Sequence[AgentConfig] | None = None,
    verification: Sequence[VerificationSensorConfig] | None = None,
) -> HarnessRegistry:
    """Materialise harness components from the configured run state."""

    registry = HarnessRegistry()

    agent_list: list[AgentConfig] = []
    if team is not None:
        agent_list.extend(team.agents)
    if agents:
        agent_list.extend(agents)
    seen: set[str] = set()
    for agent in agent_list:
        if agent.name in seen:
            continue
        seen.add(agent.name)
        _register_agent_components(registry, agent)

    if tools is not None:
        tool_list = tools.list() if isinstance(tools, ToolRegistry) else list(tools)
        _register_tool_components(registry, tool_list)

    if orchestrator is not None:
        _register_orchestrator_components(registry, orchestrator)

    if verification:
        for sensor in verification:
            cid = f"verification:global:{sensor.name}"
            if cid in registry:
                continue
            register_component(
                registry,
                make_component(
                    id=cid,
                    kind="verification",
                    source=f"verification[{sensor.name}]",
                    owner="config",
                    description=f"Verification sensor '{sensor.name}'",
                    payload=_sensor_payload(sensor),
                ),
            )

    return registry


def build_manifest(
    registry: HarnessRegistry,
    *,
    notes: str | None = None,
    created_at: datetime | None = None,
    manifest_version: str = "1",
) -> HarnessManifest:
    """Capture a deterministic snapshot of *registry* as a :class:`HarnessManifest`."""

    components = tuple(registry.list())
    payload = {
        "version": manifest_version,
        "components": [c.model_dump() for c in components],
    }
    checksum = compute_checksum(payload)
    return HarnessManifest(
        manifest_version=manifest_version,
        components=components,
        created_at=created_at or datetime.now(UTC),
        checksum=checksum,
        notes=notes,
    )


__all__ = [
    "REDACTED_CHECKSUM_MARKER",
    "HarnessRegistry",
    "build_default_registry",
    "build_manifest",
    "register_component",
]
