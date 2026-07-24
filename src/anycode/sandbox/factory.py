"""Factory for constructing sandbox providers by name."""

from __future__ import annotations

from typing import Any

from anycode.sandbox.protocol import SandboxProvider

# Provider name -> install extra carrying its SDK. SDKs load lazily on first
# use, so construction never imports them.
SANDBOX_PROVIDER_EXTRAS: dict[str, str] = {
    "daytona": "sandbox",
    "e2b": "sandbox-e2b",
    "modal": "sandbox-modal",
    "runloop": "sandbox-runloop",
}


def create_sandbox_provider(name: str, **kwargs: Any) -> SandboxProvider:
    """Build a sandbox provider by name.

    Construction is cheap and offline; the provider's SDK is imported on first
    use and a missing SDK surfaces the matching ``pip install "anycode-py[...]"``
    guidance.
    """
    if name == "daytona":
        from anycode.sandbox.daytona import DaytonaSandboxProvider

        return DaytonaSandboxProvider(**kwargs)
    if name == "e2b":
        from anycode.sandbox.e2b import E2BSandboxProvider

        return E2BSandboxProvider(**kwargs)
    if name == "modal":
        from anycode.sandbox.modal import ModalSandboxProvider

        return ModalSandboxProvider(**kwargs)
    if name == "runloop":
        from anycode.sandbox.runloop import RunloopSandboxProvider

        return RunloopSandboxProvider(**kwargs)

    raise ValueError(f"Unknown sandbox provider: {name!r}. Available: {sorted(SANDBOX_PROVIDER_EXTRAS)}")
