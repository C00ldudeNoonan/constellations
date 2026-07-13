# AGENTS.md — dbt-ml

## What this is

dbt-ml is "dbt for unstructured data." v0.1 (merged) is the pure-Python PoC:
DuckDB warehouse, 6 backends (json/markdown/pdf/html/email/llm), profiles,
manifest artifacts, dbt-shaped selectors, schema tests, incremental
materialization. A full Rust+Python design exists as a private working note
(`docs/private/`, gitignored) and is deferred to a later v2.

v0.2 is being scoped: see the GitHub issue tagged `roadmap` for the live plan.
The headline shifts are RAG support (chunking, embeddings, vector storage)
and broadening warehouse support to match the dbt-core adapter set.

## Scope discipline

- **No Rust, no PyO3.** Still pure Python through v0.2.
- **Match dbt-core warehouses over time.** v0.2 starts with DuckDB + LanceDB
  (lakehouse-style vector store); subsequent versions add Postgres, then
  Snowflake / BigQuery / Databricks / Redshift via a warehouse adapter
  pattern. Avoid changes that hard-code DuckDB; route through the adapter.
- **Metaxy is deferred** — incremental state lives in the warehouse (today
  `state.py` with DuckDB). When we move to a real adapter pattern, state
  needs to follow the adapter too.

## Build & run

```
uv sync                                # install deps
uv run dbt-ml init my_project           # scaffold (templates: json/pdf/markdown/html)
uv run dbt-ml seed --count 20           # synthetic data
uv run dbt-ml run                       # extract + materialize
uv run dbt-ml test                      # schema + custom python tests
uv run dbt-ml docs generate             # static HTML site from manifest.json
uv run pytest                          # dbt-ml package's own tests
```

## Codex GitHub auth note

Codex desktop sandbox commands can run as `codexsandboxoffline`, which may not
see the Windows keyring entry that `gh auth login` creates for the interactive
`alexn` user. If `gh auth status` fails inside the sandbox but the user says
they reauthed, retry read/write GitHub CLI commands with
`sandbox_permissions=require_escalated`; the elevated command can access the
Windows keyring. Do not ask the user to paste tokens into chat.

## Conventions

- Python 3.12+, type hints everywhere, ruff clean.
- Pydantic v2 for config models.
- Click for CLI.
- No comments explaining what code does — only why, and only when non-obvious.

## Key files

- `dbt-ml/src/dbt_ml/config/` — Pydantic models for project/source/model/profile YAML
- `dbt-ml/src/dbt_ml/dag.py` — graphlib-based DAG + selectors + Mermaid render
- `dbt-ml/src/dbt_ml/state.py` — DuckDB-backed incremental state (per-adapter in v0.2+)
- `dbt-ml/src/dbt_ml/runner.py` — extract → materialize orchestration
- `dbt-ml/src/dbt_ml/backends/` — extraction backends
- `dbt-ml/src/dbt_ml/profile.py` — profile discovery + resolution
- `dbt-ml/src/dbt_ml/manifest.py` / `docs.py` / `dbt_export.py` — artifacts
- `dbt-ml/examples/*/` — runnable example projects

## Private working notes

Anything under `docs/research/`, `docs/private/`, or `_scratch/` is gitignored.
Internal industry research and design sketches live there; never commit them.

## Brand note

"dbt" is always lowercase, even at the start of a sentence. The project name
here is "dbt-ml" (lowercase). dbt Labs is the company (capital L).
