---
title: "Evaluate AnyCode Agents with a Scenario Suite"
description: "Define AnyCode eval scenarios, run a suite deterministically or live, score by success criteria and stop reason, and compare reports to catch regressions in CI."
keywords: anycode eval, evaluation suite, EvalScenario, run_suite, deterministic eval, compare_reports, regression testing, anycode eval cli, agent testing
---

# Evaluate Agents

Prompt and config changes can quietly make an agent worse. An evaluation suite turns "seems fine" into a repeatable, scored check: define scenarios with expected outcomes, run them, and compare against a baseline to catch regressions before they ship. This guide covers writing scenarios, running a suite, and wiring it into CI.

## Write scenarios

An `EvalScenario` describes a prompt and what a good answer looks like — substrings that must appear, substrings that must not, and the expected stop reason. Scenarios live in YAML or JSON.

```yaml title="scenarios.yaml"
scenarios:
  - name: greeting_completes
    prompt: Provide a one-line greeting that includes the word completed.
    deterministic: true
    fake_responses: ["Hello — task completed successfully."]
    success_criteria: [completed]
    forbidden_substrings: [error]
    expected_stop_reason: success
    max_turns: 2
    model: fake-model
```

Set `deterministic: true` with `fake_responses` to run entirely offline against a `FakeAdapter` — no provider key, no network, byte-for-byte repeatable. Drop `deterministic` and set a real `provider`/`model` to evaluate against a live model.

| `EvalScenario` field | Default | Purpose |
| --- | --- | --- |
| `success_criteria` | `()` | Substrings that must all appear (case-insensitive) |
| `forbidden_substrings` | `()` | Substrings that must not appear |
| `expected_stop_reason` | `None` | Required stop-reason code (e.g. `success`) |
| `deterministic` | `False` | Replay `fake_responses` offline |
| `max_turns` | `4` | Turn cap for the scenario |

## Run a suite

Load the scenarios, run the suite, and write a report. `detect_provider` picks a live provider from your environment when a scenario doesn't pin one.

```python title="run_eval.py"
from anycode import load_scenarios, run_suite, write_report, render_markdown

scenarios = load_scenarios("scenarios.yaml")
report = await run_suite(scenarios, suite_name="reliability", harness_variant="baseline")

print(f"{report.passed}/{report.total_scenarios} passed")
write_report(report, "artifacts/eval/report.json")
```

`write_report` and `render_markdown` redact recognized credentials by default, including secrets in model output and failure reasons. Pass `redact_sensitive_data=False` only when the destination is independently protected and the exact output is required.

Scoring is deterministic string-and-stop-reason matching: a scenario passes only if the run succeeded, every `success_criteria` substring is present, no `forbidden_substrings` appear, and the stop reason matches. There is no model-graded scoring in the eval layer — for semantic judgment, add an inferential [verification sensor](verification-gates.md) to the run instead.

!!! note "Suites run sequentially"
    `run_suite` executes scenarios one at a time. Keep suites focused, and prefer `deterministic` scenarios for the bulk of CI so runs are fast, free, and stable.

## Catch regressions

`compare_reports` diffs a candidate report against a baseline and flags regressions — scenarios that passed before and fail now.

```python title="compare.py"
from anycode import read_report, compare_reports

diff = compare_reports(read_report("baseline.json"), read_report("candidate.json"))
print(diff["regressions"], diff["improvements"])
```

## Run it from the CLI

The `anycode eval` commands make this a one-liner in CI. `run` exits non-zero if any scenario fails; `compare` exits non-zero on a regression.

```bash title="CI"
anycode eval run scenarios.yaml --output artifacts/eval/report.json --markdown
anycode eval compare baseline.json artifacts/eval/report.json
```

Because both commands set their exit code, a failed scenario or a regression fails the build without any extra glue.

## Next steps

- [Verify output with quality gates](verification-gates.md) — the sensors that add semantic checks to a run.
- [Add self-reflection](reflection.md) — improve output before you evaluate it.
- [CLI reference](../reference/cli.md) — full `anycode eval` options.
- [Configuration reference](../reference/configuration.md) — every `EvalScenario` field.
