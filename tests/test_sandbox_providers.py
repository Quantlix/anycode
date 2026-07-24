"""Factory and conformance tests for the expanded sandbox provider catalog."""

from __future__ import annotations

import pytest

from anycode.sandbox import SANDBOX_PROVIDER_EXTRAS, DaytonaSandboxProvider, create_sandbox_provider


def test_factory_builds_daytona_provider() -> None:
    provider = create_sandbox_provider("daytona")
    assert isinstance(provider, DaytonaSandboxProvider)
    assert provider.capabilities().provider == "daytona"


def test_factory_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown sandbox provider: 'firecracker'"):
        create_sandbox_provider("firecracker")


def test_every_factory_provider_declares_an_extra() -> None:
    for name, extra in SANDBOX_PROVIDER_EXTRAS.items():
        provider = create_sandbox_provider(name)
        assert provider.capabilities().provider == name
        assert extra.startswith("sandbox")
