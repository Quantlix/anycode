---
title: "AnyCode Documentation Strategy — MkDocs Material, Diátaxis, SEO"
description: How AnyCode documents a multi-agent Python framework — MkDocs Material, Diátaxis, versioned docs with mike, a language switcher, JSON-LD, and an SEO plan.
keywords: documentation strategy, mkdocs material vs docusaurus, diataxis, llms.txt, docs seo, versioned docs mike, ai answer engines docs
---

# Documentation Strategy and Case Studies

AnyCode's documentation aims to be easy to maintain in a Python repository, fast to publish, friendly to search engines, and readable by coding agents. This page records the decisions behind the docs system, the case studies that shaped them, and the plan that keeps the site useful as the framework grows.

The stack is **MkDocs Material** with Markdown source under `site_docs/`, published to GitHub Pages. Much of what earlier revisions of this page proposed is now shipped — versioning, structured data, social cards, and a language switcher are live, not aspirational.

## What ships today

| Capability | Tool / location | Status |
| --- | --- | --- |
| Static site + local search | MkDocs Material | Live |
| Python API reference | `mkdocstrings` from `src/` | Live |
| Versioned site + version selector | `mike` (`latest` alias) | Live (via CI) |
| Python / TypeScript language switcher | `extra.alternate` in `mkdocs.yml` | Live (TS page is a placeholder) |
| Social card images (Open Graph) | Material `social` plugin | Live (CI only) |
| Structured data + canonical + robots | `overrides/main.html` (JSON-LD) | Live |
| Custom hero + brand theme | `overrides/home.html`, `stylesheets/extra.css` | Live |
| Agent entry point | `site_docs/llms.txt` | Live |
| Analytics | Google, via `GOOGLE_ANALYTICS_KEY` | Configured |

!!! note "Alpha honesty"
    AnyCode is alpha-stage. Docs should say so plainly and steer readers toward experiments, prototypes, and evaluation harnesses unless a maintainer has confirmed production readiness for a specific workload.

## Tooling decision

MkDocs Material is the best fit for this repository: AnyCode is a Python package, the project already uses `uv`, and contributors write docs in Markdown without adding a Node build chain. MkDocs produces static HTML, ships local search, generates a `sitemap.xml` from `site_url`, and documents the Python API through `mkdocstrings`.

The historical gap versus Docusaurus was versioning. That gap is now closed: **`mike`** publishes each release as a selectable version with a `latest` alias, so we get versioned docs without introducing React or MDX.

Alternatives considered:

| Tool | Strength | Tradeoff for AnyCode |
| --- | --- | --- |
| **MkDocs Material** | Python-native, fast static output, built-in search, strong API docs, versioning via `mike` | Heavy interactive/MDX components need custom work |
| Docusaurus | Mature platform with versioning, MDX, React customization | Adds Node and React maintenance to a Python library |
| Mintlify | Polished hosted docs with strong AI and API features | Hosted dependency and less repository ownership |
| Nextra / VitePress | Good Markdown developer experience | Adds a JavaScript framework dependency |
| Docsify | Simple setup | Client-rendered pages are weaker for SEO than static HTML |

## Case study: LangChain

LangChain's docs lead with a crisp product definition, a runnable agent example, provider tabs, and clear paths across LangChain, LangGraph, Deep Agents, and LangSmith. The site also exposes `https://docs.langchain.com/llms.txt`, which explicitly tells agents to fetch the documentation index before exploring.

Lessons for AnyCode:

- Put a working agent example near the top of the docs journey.
- Explain product boundaries early: AnyCode orchestrates agent teams and provider adapters, while model providers and tools stay replaceable.
- Keep observability, evaluation, and runtime control pages visible — agent frameworks are hard to debug without traceable results.
- Maintain `/llms.txt` as an agent entry point.

## Case study: CrewAI

CrewAI groups its homepage into practical journeys — get started, build the basics, enterprise journey, and what's new — and includes a copyable coding-agent setup prompt plus a link to `https://docs.crewai.com/llms.txt`.

Lessons for AnyCode:

- Group docs around reader intent, not package modules.
- Add agent-facing setup instructions instead of assuming a human will browse every page.
- Keep examples and cookbooks easy to reach from the first screen.
- Make operational topics — deployment, triggers, monitoring, guardrails — discoverable as the project matures.

## Case study: LlamaIndex

LlamaIndex opens by defining core concepts (agents, workflows, context augmentation), then offers a short quickstart and routes readers by use case, serving both beginners and advanced users.

Lessons for AnyCode:

- Keep the first concept page short and orienting.
- Explain agents, workflows, tools, and context in separate sections.
- Give advanced users a clear path to providers, plugins, memory, and verification rather than hiding those behind the quickstart.

## Content model

AnyCode uses [Diátaxis](https://diataxis.fr/) as its information architecture:

- **Tutorials** — install and quickstart.
- **How-to guides** — tools, YAML config, teams, production controls.
- **Reference** — public API and CLI.
- **Explanation** — concepts and architecture.

Keeping these types distinct focuses each page and makes retrieved sections more useful to coding agents.

## SEO plan

The SEO strategy follows Google Search Central guidance: publish useful, well-organized content, use descriptive URLs, write clear titles and snippets, link relevant pages, keep content current, and avoid keyword stuffing.

Target topics for AnyCode:

- Python multi-agent framework.
- AI agent orchestration in Python.
- Dependency-aware agent task scheduling.
- LLM tool use with Pydantic.
- Multi-provider agent framework.
- Agent verification gates and evaluation harnesses.

Each major page should answer one of those topics directly in its opening paragraph. The mechanical layer is already automated: `overrides/main.html` injects a canonical URL, robots directives, and JSON-LD (`WebSite`, `Organization`, `SoftwareSourceCode`, and per-page `TechArticle`), and Material generates Open Graph and Twitter cards — so authors focus on clear titles, descriptions, and structure.

## Agent-friendly plan

The [`llms.txt`](https://llmstxt.org/) proposal recommends a root `/llms.txt` file with a short project summary and curated links. AnyCode ships this file at `site_docs/llms.txt`; it stays short, links core docs and examples, and points at the `latest` release alias.

Agent-friendly pages should:

- Use stable headings and self-contained sections.
- Include copyable commands and runnable code.
- Avoid hiding essential content behind images or client-side interactions.
- Link from overview pages to task-specific pages.

## Publishing checklist

Before and after each release:

1. Confirm the docs URL and update `site_url` in `mkdocs.yml` if the project moves to a custom domain.
2. Configure GitHub Pages to serve from the `gh-pages` branch that `mike` publishes to.
3. Run `uv run python -m mkdocs build --strict` locally; CI runs the same check on every pull request.
4. Submit the generated `sitemap.xml` to Google Search Console after the first deployment.
5. Confirm `/llms.txt`, `/robots.txt`, and the main pages are reachable on the published `latest` alias.
6. Review page titles, descriptions, and headings, and refresh `/llms.txt` links when pages change.

## Next steps

- [Documentation contributor guide](docs-guide.md)
- [TypeScript SDK (planned)](../typescript/index.md)
- [Quickstart](../getting-started/quickstart.md)
