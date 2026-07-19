---
title: "Engineer the AnyCode Context Window as History Grows"
description: "Control AnyCode context with policies that trim, mask, offload, compact, or hand off history under pressure while preserving task and verification state."
keywords: anycode context engineering, ContextPolicy, context window, context pressure, offload_text, model context profile, token budget, compaction, context manager
---

# Engineer the Context Window

Long runs accumulate history until it no longer fits the model's context window. Rather than blindly truncating, AnyCode applies a **context policy**: as pressure rises it trims, masks, offloads, compacts, and finally hands off — while always preserving the things a downstream agent can't lose, like task state and verification failures. This guide shows how to shape that behavior.

## The pressure ladder

A `ContextPolicy` defines ratios of the effective window at which each strategy kicks in. As the assembled context grows, AnyCode escalates through them:

| Strategy | Default ratio | What it does |
| --- | --- | --- |
| `trim` | `0.65` | Drop the oldest low-priority messages |
| `mask` | `0.70` | Replace bulky content with placeholders |
| `offload` | `0.75` | Write oversized tool output to disk, leave a handle |
| `compact` | `0.85` | Summarize archived history |
| `handoff` | `0.95` | Serialize context for a fresh agent to resume |

```mermaid
flowchart LR
    N["normal"] --> T["trim 0.65"] --> M["mask 0.70"] --> O["offload 0.75"] --> C["compact 0.85"] --> H["handoff 0.95"]
```

## Attach a policy to an agent

Set `context_policy` on an `AgentConfig`. The runner honors it every turn and records what it did in `AgentRunResult.context_manifests`.

```python title="policy.py"
from anycode import AgentConfig
from anycode.types import ContextPolicy

policy = ContextPolicy(
    enabled=True,
    max_context_tokens=100_000,
    keep_recent_messages=6,
    max_tool_output_tokens=4_000,
    preserved_task_state={"objective": "Migrate the billing module"},
    preserved_verification_failures=("pytest: 2 failing in tests/test_billing.py",),
)

agent = AgentConfig(
    name="migrator",
    provider="anthropic",
    model="claude-sonnet-5",
    tools=["file_read", "file_edit", "grep"],
    context_policy=policy,
)
```

| `ContextPolicy` field | Default | Effect |
| --- | --- | --- |
| `enabled` | `False` | Policy is inert until you set this |
| `mode` | `"manual"` | `disabled`, `manual` (use the cap), or `auto` (use the model's window) |
| `max_context_tokens` | `100_000` | Manual-mode window cap |
| `keep_recent_messages` | `6` | Recent messages always kept intact |
| `max_tool_output_tokens` | `4_000` | Tool output above this is offloaded |
| `redact_sensitive_data` | `True` | Scrub recognized credentials from persisted artifacts |
| `preserved_task_state` | `{}` | Key facts that survive every reduction |
| `preserved_verification_failures` | `()` | Failures that survive so they aren't forgotten |

!!! warning "It's off by default"
    `ContextPolicy.enabled` defaults to `False`, so no pressure management happens until you turn it on, and `mode` defaults to `"manual"` (the 100k cap) rather than the model's true window. Set `mode="auto"` to size the window from the model's own profile.

## Let the model's profile size the window

In `"auto"` mode, AnyCode sizes the window from a `ModelContextProfile`. It ships profiles for current Anthropic, OpenAI, and Google models; register a `custom_profiles` entry for anything it doesn't know.

```python title="auto_mode.py"
from anycode.types import ContextPolicy, ModelContextProfile

giga = ModelContextProfile(
    provider="myvendor", model="giga-1m",
    max_context_tokens=1_000_000, max_output_tokens=64_000,
)
policy = ContextPolicy(enabled=True, mode="auto", custom_profiles=(giga,))
```

## Inspect and offload manually

`ContextManager` is the engine behind the policy — you can drive it directly to see what a policy would do, and read a per-section usage report.

```python title="inspect.py"
from anycode import ContextManager
from anycode.context.reporting import render_usage_report_table

manager = ContextManager(policy, provider="anthropic", model="claude-sonnet-5")
prepared, manifest = manager.assemble(messages)
print(manifest.pressure)                                 # normal .. handoff
print(render_usage_report_table(manifest.usage_report))  # per-section token table
```

The low-level `offload_text` / `restore_text` helpers move a large blob to disk and back with an integrity check, and `rebuild_from_handoff` restores a serialized context for a fresh agent. By default, offloaded text, archived history, and handoff files replace recognized credentials with `<redacted-secret>` before writing. The digest verifies the redacted artifact, so restoration is exact for the persisted form rather than necessarily the original input.

Set `ContextPolicy(redact_sensitive_data=False)` or pass `redact_sensitive_data=False` to `offload_text` only when the artifact directory has its own encryption, access control, and retention policy.

!!! tip "tiktoken sharpens token counts"
    By default AnyCode counts tokens heuristically. Install `anycode-py[tokens]` and the OpenAI-family profiles use `tiktoken` for exact counts, which makes the pressure ladder trigger at the right moments.

## The complete, runnable program

The fragments above are pieces of one file. Here is a complete `context_engineering.py` that registers a custom huge-context model profile, runs a policy in `"auto"` mode, assembles a long history through `ContextManager`, and prints the resulting pressure, the preserved task state and open failures, and a per-section usage table. It makes no LLM calls, so it runs offline with no API key.

```python title="context_engineering.py"
from anycode import ContextManager, LLMMessage, TextBlock
from anycode.context.reporting import render_usage_report_table
from anycode.types import ContextPolicy, ModelContextProfile


def make_history(n: int) -> list[LLMMessage]:
    """Build a long conversation so the context grows under pressure."""
    return [
        LLMMessage(
            role="user" if i % 2 == 0 else "assistant",
            content=[TextBlock(text=f"turn {i}: " + ("lorem ipsum " * 80))],
        )
        for i in range(n)
    ]


def main() -> None:
    # Auto mode sizes the window from a model profile instead of a fixed cap.
    # Register a custom profile for any model AnyCode does not already know.
    giga = ModelContextProfile(
        provider="myvendor",
        model="giga-1m",
        max_context_tokens=1_000_000,
        max_output_tokens=64_000,
    )
    policy = ContextPolicy(
        enabled=True,
        mode="auto",
        keep_recent_messages=6,
        max_tool_output_tokens=4_000,
        custom_profiles=(giga,),
        preserved_task_state={"objective": "Migrate the billing module"},
        preserved_verification_failures=("pytest: 2 failing in tests/test_billing.py",),
    )

    # ContextManager.assemble is synchronous — it reports what the policy would do.
    manager = ContextManager(policy, provider="myvendor", model="giga-1m")
    prepared, manifest = manager.assemble(make_history(40))

    print(f"pressure:          {manifest.pressure}")
    print(f"prepared messages: {len(prepared)}")
    print(f"preserved state:   {manifest.preserved_task_state}")
    print(f"open failures:     {manifest.preserved_verification_failures}")
    if manifest.usage_report is not None:
        print()
        print(render_usage_report_table(manifest.usage_report))


if __name__ == "__main__":
    main()
```

Run it from the project root:

```bash
uv run python context_engineering.py
```

!!! tip "Tested copy"
    See [`examples/26_context_engineering.py`](https://github.com/Quantlix/anycode/blob/main/examples/26_context_engineering.py) for section-aware budgets on a huge-context model, plus [`examples/19_adaptive_context.py`](https://github.com/Quantlix/anycode/blob/main/examples/19_adaptive_context.py) and [`examples/23_context_pressure.py`](https://github.com/Quantlix/anycode/blob/main/examples/23_context_pressure.py) for the offload, compaction, and handoff steps of the pressure ladder in action.

## Next steps

- [Give agents memory and RAG](memory-and-rag.md) — the retrieved context this policy budgets a section for.
- [Durable and resumable runs](durability.md) — where context handoff serialization pays off.
- [Production controls](production-controls.md) — context policy in the broader hardening picture.
- [Configuration reference](../reference/configuration.md) — every `ContextPolicy` field.
