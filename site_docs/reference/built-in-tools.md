---
title: "AnyCode Built-in Tools Reference"
description: "Reference all six AnyCode agent tools—bash, file_read, file_write, file_edit, grep, and list_files—with parameters, defaults, outputs, and safety limits."
keywords: AnyCode built-in tools, bash tool, file_read, file_write, file_edit, grep tool, list_files, tool parameters, tool limits
---

# Built-in Tools

AnyCode ships six built-in tools, registered in this order: `bash`, `file_read`, `file_write`, `file_edit`, `grep`, `list_files`. Every agent built through `AnyCode.build_agent` gets all six registered automatically; an agent's `tools` allowlist in `AgentConfig` controls which it may call. All tools validate input with Pydantic before executing and return a text `ToolResult` — validation failures and exceptions come back to the model as error results, never as raised exceptions.

Add `"handoff"` to an agent's `tools` list to also register the built-in handoff tool.

## `bash`

Run a shell command and capture stdout and stderr.

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `command` | `str` | required | Runs through the system shell |
| `timeout` | `float` | `30` s | Per-call override |
| `cwd` | `str` | `None` | Working directory, passed through as-is |
| `max_output_bytes` | `int` | `200_000` | Cap per stream (stdout and stderr each) |

Output is stdout, with stderr appended under a `-- stderr --` header and `(exit code N)` on non-zero exit. `is_error` is true whenever the exit code is non-zero.

**Hardening.** The child runs in its own process group (`start_new_session` on POSIX, `CREATE_NEW_PROCESS_GROUP` on Windows). On timeout the entire process tree is killed (`taskkill /F /T` on Windows, `killpg` + `SIGKILL` on POSIX) and the call returns exit code `124`; a command that cannot be spawned returns `127`. Output streams are fully drained even past the cap — no pipe-buffer deadlock — and truncation is explicit: `[output truncated: showing first N of TOTAL bytes; D dropped]`.

!!! warning "No sandbox at this layer"
    `bash` executes through the real shell with the parent's environment and no command allowlist. Scope it with the agent `tools` allowlist, guardrail `blocked_tools`, or an approval gate.

## `file_read`

Read a file with optional line windowing. Output lines are numbered `N<TAB>line`.

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `path` | `str` | required | Absolute path |
| `offset` | `int` | `None` | 1-based start line |
| `limit` | `int` | `None` | Max lines |

Truncated reads append `(showing lines A–B of TOTAL)`. An offset past end-of-file is an error stating the total line count.

## `file_write`

Create or overwrite a file, creating parent directories as needed.

| Parameter | Type | Default |
| --- | --- | --- |
| `path` | `str` | required |
| `content` | `str` | required |

Writes are **atomic** (same-directory temp file + `os.replace`) and **byte-exact** — line endings are written as given, not translated to the host convention. Returns `Created "…" (N lines, B bytes).` or `Overwrote "…" (…)`.

## `file_edit`

Exact-string replacement in an existing file. Same atomic, byte-exact write path as `file_write`.

| Parameter | Type | Default |
| --- | --- | --- |
| `path` | `str` | required |
| `old_string` | `str` | required |
| `new_string` | `str` | required |
| `replace_all` | `bool` | `False` |

Zero matches is an error. More than one match without `replace_all=True` is an error telling the model to use a more specific string.

## `grep`

Regex search returning `path:line:text` matches.

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `pattern` | `str` | required | Regular expression |
| `path` | `str` | cwd | File or directory |
| `glob` | `str` | `None` | e.g. `"*.py"` |
| `max_results` | `int` | `100` | Match cap |

Uses ripgrep when `rg` is on `PATH`, otherwise a pure-Python walk that skips `.git`, `.svn`, `.hg`, `node_modules`, `.next`, `dist`, `build`, `__pycache__`, and `.venv`. No matches returns `No matches.`; an invalid regex is an error.

## `list_files`

List files fast, preferring native tooling and falling back gracefully.

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `path` | `str` | cwd | Root directory |
| `glob` | `str` | `None` | Filename glob, case-insensitive (e.g. `"*.py"`) |
| `max_results` | `int` | `1000` | Result cap |

Backends are tried in order — `git ls-files` (respects `.gitignore`), `rg --files`, `fd`, then a Python `os.walk` that skips the same ignored directories as `grep` — and each native backend gets a 15-second timeout before falling through to the next. Output is newline-separated relative paths plus a footer like `(K of TOTAL file(s), backend: git)`, with a `capped` note when results were truncated. A missing path is an error; an empty directory is not.

## Execution engine guarantees

These apply to built-in, custom, and MCP tools alike:

- **Parallel execution.** Independent tool calls in one turn run concurrently (`asyncio.gather`), bounded by a semaphore of 4. Results are returned in the original call order.
- **Errors are data.** Unknown tool, invalid input, or an exception inside a tool all produce an error `ToolResult` the model can react to — the run does not crash.
- **Approval hooks.** When an `ApprovalManager` is configured, tool calls can be denied (error result) or approved with modified input before execution.
- **Telemetry.** Each invocation runs inside a tracer span named `anycode.tool.<name>` with an `is_error` attribute.

## See also

- [Work with tools](../guides/tools.md) — building custom tools with `define_tool`
- [Connect MCP servers](../guides/mcp.md) — external tools over stdio and HTTP
