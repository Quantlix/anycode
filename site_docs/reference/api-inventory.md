---
title: "Complete AnyCode Python API Inventory"
description: "Browse every supported name exported by the AnyCode Python package, generated from anycode.__all__ and source docstrings at documentation build time."
keywords: AnyCode API inventory, anycode exports, Python agent framework API, anycode __all__, mkdocstrings
---

# Complete API Inventory

This inventory is generated from the `anycode` package root at build time. It covers every supported name declared by `anycode.__all__`, including protocols, configuration and result models, helpers, constants, exceptions, and subsystem entry points.

Use `from anycode import ...` for these names. The [curated public API guide](public-api.md) explains the primary entry points by workflow, while this page is the exhaustive lookup surface. Internal modules and private names remain implementation details under the [compatibility policy](compatibility.md).

## Module exports

- `stop_reasons` exposes the canonical stop-reason constants as a module namespace for comparisons and integrations.

::: anycode
    options:
      members: true
      members_order: source
      show_root_heading: false
      show_root_full_path: false
      show_source: false
