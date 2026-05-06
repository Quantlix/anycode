"""Resolve VerificationSensorConfig entries to runnable Sensor instances.

Used by the runner/orchestrator to wire declarative (YAML/TOML) sensor configs
into the live execution pipeline. Custom sensor factories can be registered at
runtime so user-defined sensors are addressable by name from config files.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Final

from anycode.types import VerificationResult, VerificationSensorConfig
from anycode.verification.builtins import (
    pyright_sensor,
    pytest_sensor,
    ruff_sensor,
)
from anycode.verification.sensor import Sensor, SensorContext

SensorFactory = Callable[[VerificationSensorConfig], Sensor]


def _str_option(config: VerificationSensorConfig, key: str, default: str) -> str:
    raw = config.options.get(key)
    return str(raw) if raw is not None else default


def _build_ruff(config: VerificationSensorConfig) -> Sensor:
    target = _str_option(config, "target", "src/")
    return ruff_sensor(target, phases=config.phases)


def _build_pyright(config: VerificationSensorConfig) -> Sensor:
    target = _str_option(config, "target", "src/")
    return pyright_sensor(target, phases=config.phases)


def _build_pytest(config: VerificationSensorConfig) -> Sensor:
    target = _str_option(config, "target", "tests/")
    return pytest_sensor(target, phases=config.phases)


def _build_regex_sensor(config: VerificationSensorConfig) -> Sensor:
    """Pure-python sensor: pass when output matches/doesn't match a regex."""
    pattern = _str_option(config, "pattern", "")
    expect = config.options.get("expect", "match")
    severity_raw = config.options.get("severity", "error")
    severity = str(severity_raw)
    compiled = re.compile(pattern, re.MULTILINE | re.DOTALL) if pattern else None

    def _run(ctx: SensorContext) -> VerificationResult:
        text = ctx.output or ""
        if compiled is None:
            return VerificationResult(
                sensor_name=config.name,
                kind=config.kind,
                passed=False,
                severity="error",
                message="regex sensor missing 'pattern' option",
            )
        match = compiled.search(text)
        if expect == "match":
            passed = match is not None
            msg = "pattern matched" if passed else "pattern not found"
        else:
            passed = match is None
            msg = "pattern absent (as required)" if passed else "forbidden pattern present"
        return VerificationResult(
            sensor_name=config.name,
            kind=config.kind,
            passed=passed,
            severity="info" if passed else severity,  # type: ignore[arg-type]
            message=msg,
            evidence={"pattern": pattern, "expect": str(expect)},
            feedback_for_agent=None if passed else f"Verification '{config.name}' failed: {msg}.",
        )

    return Sensor(config=config, fn=_run)


_BUILTIN_FACTORIES: Final[dict[str, SensorFactory]] = {
    "ruff": _build_ruff,
    "pyright": _build_pyright,
    "pytest": _build_pytest,
    "regex": _build_regex_sensor,
}

_CUSTOM_FACTORIES: dict[str, SensorFactory] = {}


def register_sensor_factory(name: str, factory: SensorFactory) -> None:
    """Register a user-defined sensor factory addressable by name from config."""
    _CUSTOM_FACTORIES[name] = factory


def build_sensor(config: VerificationSensorConfig) -> Sensor:
    """Build a Sensor from a VerificationSensorConfig declared in YAML/TOML or code."""
    factory = _CUSTOM_FACTORIES.get(config.name) or _BUILTIN_FACTORIES.get(config.name)
    if factory is None:
        raise ValueError(
            f"Unknown verification sensor '{config.name}'. Register with register_sensor_factory or use one of: {sorted(_BUILTIN_FACTORIES)}"
        )
    return factory(config)


def build_sensors(configs: tuple[VerificationSensorConfig, ...] | list[VerificationSensorConfig]) -> list[Sensor]:
    return [build_sensor(c) for c in configs]
