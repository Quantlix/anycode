"""Parse YAML or TOML config files into typed AnyCode configurations."""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from anycode.types import (
    AgentConfig,
    ContextPolicy,
    ContextSectionBudget,
    ContextSectionKind,
    CostConfig,
    GuardrailConfig,
    ModelContextProfile,
    OrchestratorConfig,
    ProviderResilienceConfig,
    RAGConfig,
    ReflectionConfig,
    RoutingConfig,
    TeamConfig,
    ToolIdempotencyConfig,
    VerificationSensorConfig,
)

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


class TaskSpecConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    title: str
    description: str
    assignee: str | None = None
    depends_on: list[str] | None = None


@dataclass
class LoadedConfig:
    """Result of loading a config file."""

    team: TeamConfig
    tasks: list[TaskSpecConfig] | None = None
    guardrails: GuardrailConfig | None = None
    routing: RoutingConfig | None = None
    cost: CostConfig | None = None
    reflection: ReflectionConfig | None = None
    rag: RAGConfig | None = None
    verification: tuple[VerificationSensorConfig, ...] = ()
    context_policy: ContextPolicy | None = None
    max_handoff_depth: int | None = None
    provider_resilience: ProviderResilienceConfig | None = None
    tool_idempotency: ToolIdempotencyConfig = ToolIdempotencyConfig()

    def to_orchestrator_config(self) -> OrchestratorConfig:
        if self.max_handoff_depth is not None:
            return OrchestratorConfig(
                max_concurrency=self.team.max_concurrency,
                routing=self.routing,
                cost=self.cost,
                reflection=self.reflection,
                rag=self.rag,
                verification=self.verification,
                max_handoff_depth=self.max_handoff_depth,
                provider_resilience=self.provider_resilience,
                tool_idempotency=self.tool_idempotency,
            )
        return OrchestratorConfig(
            max_concurrency=self.team.max_concurrency,
            routing=self.routing,
            cost=self.cost,
            reflection=self.reflection,
            rag=self.rag,
            verification=self.verification,
            provider_resilience=self.provider_resilience,
            tool_idempotency=self.tool_idempotency,
        )


def _substitute_env(value: Any) -> Any:
    """Recursively substitute ${ENV_VAR} placeholders in strings."""
    if isinstance(value, str):

        def _replace(match: re.Match[str]) -> str:
            var = match.group(1)
            return os.environ.get(var, match.group(0))

        return _ENV_VAR_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _substitute_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env(v) for v in value]
    return value


def _read_raw(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as e:
            raise ImportError('PyYAML is required for YAML config files. Install it with: pip install "anycode-py[cli]"') from e
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    elif suffix == ".toml":
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    else:
        raise ValueError(f"Unsupported config file extension: {suffix!r}. Use .yaml, .yml, or .toml.")
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a top-level mapping.")
    return data


def _parse_context_policy(raw: dict[str, Any]) -> ContextPolicy | None:
    """Parse an optional top-level `context_engineering` block into a `ContextPolicy`."""
    block = raw.get("context_engineering")
    if not block:
        return None
    if not isinstance(block, dict):
        raise ValueError("`context_engineering` must be a mapping.")

    enabled_raw = block.get("enabled", True)
    if isinstance(enabled_raw, str) and enabled_raw == "auto":
        enabled = True
        mode = "auto"
    elif isinstance(enabled_raw, bool):
        enabled = enabled_raw
        mode = block.get("mode") or ("manual" if enabled else "disabled")
    else:
        enabled = bool(enabled_raw)
        mode = block.get("mode", "manual")

    window = block.get("window") or {}
    reserved = int(window.get("reserved_response_tokens", block.get("reserved_response_tokens", 0)))
    if mode == "auto" and "mode" not in block:
        mode = "auto"

    sections_raw = block.get("sections") or {}
    sections: dict[ContextSectionKind, ContextSectionBudget] = {}
    for kind, payload in sections_raw.items():
        if not isinstance(payload, dict):
            raise ValueError(f"`context_engineering.sections.{kind}` must be a mapping.")
        sections[kind] = ContextSectionBudget(kind=kind, **payload)

    profiles_raw = block.get("model_profiles") or []
    custom_profiles = tuple(ModelContextProfile.model_validate(p) for p in profiles_raw)

    return ContextPolicy(
        enabled=enabled,
        mode=mode,  # type: ignore[arg-type]
        reserved_response_tokens=reserved,
        sections=sections,
        custom_profiles=custom_profiles,
        max_context_tokens=int(block.get("max_context_tokens", 100_000)),
    )


def load_config(path: str | os.PathLike[str]) -> LoadedConfig:
    """Load a YAML or TOML config file and return a typed LoadedConfig."""
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = _substitute_env(_read_raw(config_path))

    agents_raw = raw.get("agents", [])
    if not isinstance(agents_raw, list) or not agents_raw:
        raise ValueError("Config must define a non-empty 'agents' list.")

    global_verification = raw.get("verification")
    global_sensors: tuple[VerificationSensorConfig, ...] = ()
    if global_verification:
        if not isinstance(global_verification, list):
            raise ValueError("Top-level 'verification' must be a list of sensor configs.")
        global_sensors = tuple(VerificationSensorConfig.model_validate(item) for item in global_verification)

    global_context_policy = _parse_context_policy(raw)

    typed_agents: list[AgentConfig] = []
    for raw_agent in agents_raw:
        agent_data = dict(raw_agent)
        agent_verification_raw = agent_data.pop("verification", None)
        agent_context_raw = agent_data.pop("context_policy", None)
        agent = AgentConfig.model_validate(agent_data)
        sensors: tuple[VerificationSensorConfig, ...] = global_sensors
        if agent_verification_raw is not None:
            if not isinstance(agent_verification_raw, list):
                raise ValueError(f"Agent '{agent.name}' verification must be a list.")
            agent_sensors = tuple(VerificationSensorConfig.model_validate(item) for item in agent_verification_raw)
            sensors = agent_sensors if agent_sensors else global_sensors
        if sensors:
            agent = agent.model_copy(update={"verification": sensors})
        if agent_context_raw is not None:
            if not isinstance(agent_context_raw, dict):
                raise ValueError(f"Agent '{agent.name}' context_policy must be a mapping.")
            agent = agent.model_copy(update={"context_policy": ContextPolicy.model_validate(agent_context_raw)})
        elif global_context_policy is not None and agent.context_policy is None:
            agent = agent.model_copy(update={"context_policy": global_context_policy})
        typed_agents.append(agent)
    agents = typed_agents

    team = TeamConfig(
        name=raw.get("name", "team"),
        agents=agents,
        shared_memory=raw.get("shared_memory"),
        max_concurrency=raw.get("max_concurrency"),
    )

    tasks_raw = raw.get("tasks")
    tasks = [TaskSpecConfig.model_validate(t) for t in tasks_raw] if tasks_raw else None

    guardrails = GuardrailConfig.model_validate(raw["guardrails"]) if "guardrails" in raw else None
    routing = RoutingConfig.model_validate(raw["routing"]) if "routing" in raw else None
    cost = CostConfig.model_validate(raw["cost"]) if "cost" in raw else None
    reflection = ReflectionConfig.model_validate(raw["reflection"]) if "reflection" in raw else None
    rag = RAGConfig.model_validate(raw["rag"]) if "rag" in raw else None
    provider_resilience = ProviderResilienceConfig.model_validate(raw["provider_resilience"]) if "provider_resilience" in raw else None
    tool_idempotency = ToolIdempotencyConfig.model_validate(raw.get("tool_idempotency", {}))

    raw_handoff_depth = raw.get("max_handoff_depth")
    if raw_handoff_depth is not None and not isinstance(raw_handoff_depth, int):
        raise ValueError("Top-level 'max_handoff_depth' must be an integer.")

    return LoadedConfig(
        team=team,
        tasks=tasks,
        guardrails=guardrails,
        routing=routing,
        cost=cost,
        reflection=reflection,
        rag=rag,
        verification=global_sensors,
        context_policy=global_context_policy,
        max_handoff_depth=raw_handoff_depth,
        provider_resilience=provider_resilience,
        tool_idempotency=tool_idempotency,
    )
