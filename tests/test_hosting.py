"""Managed-host lifecycle, Agent Card, and deployment profile tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

import yaml

from anycode.backends import Admission, SQLiteDurabilityBackend, WorkItem
from anycode.contracts import CapabilityDescriptor, Event, Run
from anycode.hosting import A2A_AGENT_CARD_PATH, HostLifecycle, build_deployment_agent_card

ROOT = Path(__file__).parents[1]


async def test_host_stops_admission_then_drains_inflight_work() -> None:
    lifecycle = HostLifecycle(max_inflight=2)
    await lifecycle.start()
    assert (await lifecycle.admit("work-1")).accepted
    assert await lifecycle.begin("work-1")
    drain = asyncio.create_task(lifecycle.drain(timeout_seconds=1))
    await asyncio.sleep(0)

    rejected = await lifecycle.admit("work-2")
    assert not rejected.accepted and rejected.error is not None and rejected.error.code == "host_draining"
    assert lifecycle.live() and not lifecycle.ready()
    await lifecycle.complete("work-1")
    result = await drain

    assert result.drained and not result.timed_out
    assert lifecycle.snapshot().state == "stopped" and not lifecycle.live()


async def test_drain_timeout_durably_returns_work() -> None:
    lifecycle = HostLifecycle()
    await lifecycle.start()
    await lifecycle.admit("work-1")
    await lifecycle.begin("work-1")
    returned: list[str] = []

    async def durable_return(work: tuple[str, ...]) -> None:
        returned.extend(work)

    result = await lifecycle.drain(timeout_seconds=0, durable_return=durable_return)

    assert result.drained and result.timed_out and result.durably_returned == ("work-1",)
    assert returned == ["work-1"] and lifecycle.snapshot().state == "stopped"


def test_agent_card_is_generated_per_endpoint_in_a2a_1_shape() -> None:
    capability = CapabilityDescriptor(
        name="AnyCode Research",
        implementation_version="0.8.0",
        operations=("send_message", "get_task", "cancel_task"),
    )
    production = build_deployment_agent_card(
        capability,
        endpoint="https://agents.example.com",
        description="Research workflow",
        openid_connect_url="https://identity.example.com/.well-known/openid-configuration",
    )
    canary = build_deployment_agent_card(
        capability,
        endpoint="https://canary-agents.example.com",
        description="Research workflow",
    )
    payload = production.model_dump(mode="json", by_alias=True, exclude_none=True)

    assert A2A_AGENT_CARD_PATH == "/.well-known/agent-card.json"
    assert payload["supportedInterfaces"][0] == {  # type: ignore[index]
        "url": "https://agents.example.com/a2a",
        "protocolBinding": "HTTP+JSON",
        "protocolVersion": "1.0",
    }
    assert payload["capabilities"]["streaming"]  # type: ignore[index]
    assert len(payload["skills"]) == 3  # type: ignore[arg-type]
    assert production.supported_interfaces[0].url != canary.supported_interfaces[0].url
    assert "credential" not in production.model_dump_json().casefold()


async def test_rolling_host_replacement_preserves_backend_history_and_ready_work(tmp_path: Path) -> None:
    path = tmp_path / "rolling.db"
    old_backend = SQLiteDurabilityBackend(path)
    run = Run(id="run-1", correlation_id="corr-1")
    initial = Event(id="event-1", run_id=run.id, sequence=1, type="run.accepted", correlation_id=run.correlation_id)
    assert (await old_backend.admit(Admission(admission_key="admit-1", run=run, initial_event=initial))).admitted
    progress = Event(id="event-2", run_id=run.id, sequence=2, type="task.queued", correlation_id=run.correlation_id)
    assert (await old_backend.append_event(progress, expected_sequence=1)).accepted
    await old_backend.enqueue(WorkItem(id="work-1", run_id=run.id, task_id="task-1"))
    old_host = HostLifecycle()
    await old_host.start()
    assert (await old_host.drain(timeout_seconds=0)).drained

    new_backend = SQLiteDurabilityBackend(path)
    new_host = HostLifecycle()
    await new_host.start()

    assert [event.sequence for event in await new_backend.read_events(run.id, after_sequence=0)] == [1, 2]
    assert [event.sequence for event in await new_backend.read_events(run.id, after_sequence=1)] == [2]
    assert (await new_backend.claim("new-worker")).claimed
    assert new_host.ready()


def test_deployment_profiles_include_probes_resources_identity_and_no_plaintext_secrets() -> None:
    deployment = yaml.safe_load((ROOT / "deploy" / "kubernetes" / "deployment.yaml").read_text(encoding="utf-8"))
    config = yaml.safe_load((ROOT / "deploy" / "kubernetes" / "configmap.yaml").read_text(encoding="utf-8"))
    account = yaml.safe_load((ROOT / "deploy" / "kubernetes" / "serviceaccount.yaml").read_text(encoding="utf-8"))
    compose = yaml.safe_load((ROOT / "deploy" / "container" / "compose.yaml").read_text(encoding="utf-8"))
    container = deployment["spec"]["template"]["spec"]["containers"][0]

    assert deployment["spec"]["strategy"]["rollingUpdate"]["maxUnavailable"] == 0
    assert deployment["spec"]["template"]["metadata"]["annotations"]["dapr.io/enabled"] == "true"
    assert container["livenessProbe"]["httpGet"]["path"] == "/health/live"
    assert container["readinessProbe"]["httpGet"]["path"] == "/health/ready"
    assert container["lifecycle"]["preStop"]["httpGet"]["path"] == "/health/drain"
    assert container["resources"]["requests"] and container["resources"]["limits"]
    assert account["metadata"]["annotations"]
    assert config["data"]["ANYCODE_WORKLOAD_IDENTITY_REF"].startswith("kubernetes:")
    assert compose["services"]["anycode"]["volumes"]
    serialized = "\n".join(
        (ROOT / "deploy" / path).read_text(encoding="utf-8").casefold()
        for path in (
            "kubernetes/deployment.yaml",
            "kubernetes/configmap.yaml",
            "kubernetes/serviceaccount.yaml",
            "container/compose.yaml",
        )
    )
    assert "api_key:" not in serialized and "password:" not in serialized and ":latest" not in serialized
