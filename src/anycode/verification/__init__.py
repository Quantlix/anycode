"""Verification sensors and quality gates."""

from anycode.verification.builtins import (
    pyright_sensor,
    pytest_sensor,
    ruff_sensor,
    schema_sensor,
)
from anycode.verification.gate import QualityGate, decide_gate
from anycode.verification.registry import (
    SensorFactory,
    build_sensor,
    build_sensors,
    register_sensor_factory,
)
from anycode.verification.sensor import Sensor, SensorContext, SensorFn

__all__ = [
    "QualityGate",
    "Sensor",
    "SensorContext",
    "SensorFactory",
    "SensorFn",
    "build_sensor",
    "build_sensors",
    "decide_gate",
    "pyright_sensor",
    "pytest_sensor",
    "register_sensor_factory",
    "ruff_sensor",
    "schema_sensor",
]
