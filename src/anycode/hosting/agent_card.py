"""A2A 1.0 Agent Card generation for concrete deployment endpoints."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from anycode.contracts.models import CapabilityDescriptor

A2A_PROTOCOL_VERSION = "1.0"
A2A_AGENT_CARD_PATH = "/.well-known/agent-card.json"


class AgentCardModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class AgentInterface(AgentCardModel):
    url: str = Field(min_length=1)
    protocol_binding: str = Field(default="HTTP+JSON", alias="protocolBinding")
    protocol_version: str = Field(default=A2A_PROTOCOL_VERSION, alias="protocolVersion")


class AgentSkill(AgentCardModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    tags: tuple[str, ...]
    examples: tuple[str, ...] = ()
    input_modes: tuple[str, ...] = Field(default=("application/json", "text/plain"), alias="inputModes")
    output_modes: tuple[str, ...] = Field(default=("application/json", "text/plain"), alias="outputModes")


class AgentCapabilities(AgentCardModel):
    streaming: bool
    push_notifications: bool = Field(default=False, alias="pushNotifications")
    extended_agent_card: bool = Field(default=False, alias="extendedAgentCard")


class AgentProvider(AgentCardModel):
    organization: str = Field(min_length=1)
    url: str | None = None


class A2AAgentCard(AgentCardModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    supported_interfaces: tuple[AgentInterface, ...] = Field(alias="supportedInterfaces", min_length=1)
    version: str = Field(min_length=1)
    capabilities: AgentCapabilities
    default_input_modes: tuple[str, ...] = Field(alias="defaultInputModes", min_length=1)
    default_output_modes: tuple[str, ...] = Field(alias="defaultOutputModes", min_length=1)
    skills: tuple[AgentSkill, ...] = Field(min_length=1)
    provider: AgentProvider | None = None
    security_schemes: dict[str, JsonValue] = Field(default_factory=dict, alias="securitySchemes")
    security: tuple[dict[str, tuple[str, ...]], ...] = ()


def build_deployment_agent_card(
    capability: CapabilityDescriptor,
    *,
    endpoint: str,
    description: str,
    organization: str = "Quantlix",
    organization_url: str | None = None,
    openid_connect_url: str | None = None,
) -> A2AAgentCard:
    """Create one public card for one externally reachable deployment endpoint."""
    base = endpoint.rstrip("/")
    operations = capability.operations or ("task-lifecycle",)
    skills = tuple(
        AgentSkill(
            id=operation.replace("_", "-").replace(".", "-"),
            name=operation.replace("_", " ").replace(".", " ").title(),
            description=f"AnyCode operation: {operation}",
            tags=("anycode", operation),
        )
        for operation in operations
    )
    security_schemes: dict[str, JsonValue] = {}
    security: tuple[dict[str, tuple[str, ...]], ...] = ()
    if openid_connect_url:
        security_schemes = {"openid": {"openIdConnectSecurityScheme": {"openIdConnectUrl": openid_connect_url}}}
        security = ({"openid": ()},)
    return A2AAgentCard(
        name=capability.name,
        description=description,
        supportedInterfaces=(AgentInterface(url=f"{base}/a2a", protocolBinding="HTTP+JSON"),),
        version=capability.implementation_version,
        capabilities=AgentCapabilities(streaming=capability.supports_event_stream),
        defaultInputModes=("application/json", "text/plain"),
        defaultOutputModes=("application/json", "text/plain"),
        skills=skills,
        provider=AgentProvider(organization=organization, url=organization_url),
        securitySchemes=security_schemes,
        security=security,
    )
