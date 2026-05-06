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
    CostConfig,
    GuardrailConfig,
    OrchestratorConfig,
    RAGConfig,
    ReflectionConfig,
    RoutingConfig,
    TeamConfig,
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

    def to_orchestrator_config(self) -> OrchestratorConfig:
        return OrchestratorConfig(
            max_concurrency=self.team.max_concurrency,
            routing=self.routing,
            cost=self.cost,
            reflection=self.reflection,
            rag=self.rag,
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
            raise ImportError("PyYAML is required for YAML config files. Install with: pip install anycode[cli]") from e
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

    typed_agents: list[AgentConfig] = []
    for raw_agent in agents_raw:
        agent_data = dict(raw_agent)
        agent_verification_raw = agent_data.pop("verification", None)
        agent = AgentConfig.model_validate(agent_data)
        sensors: tuple[VerificationSensorConfig, ...] = global_sensors
        if agent_verification_raw is not None:
            if not isinstance(agent_verification_raw, list):
                raise ValueError(f"Agent '{agent.name}' verification must be a list.")
            agent_sensors = tuple(VerificationSensorConfig.model_validate(item) for item in agent_verification_raw)
            sensors = agent_sensors if agent_sensors else global_sensors
        if sensors:
            agent = agent.model_copy(update={"verification": sensors})
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

    return LoadedConfig(
        team=team,
        tasks=tasks,
        guardrails=guardrails,
        routing=routing,
        cost=cost,
        reflection=reflection,
        rag=rag,
    )
