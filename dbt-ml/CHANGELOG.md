# Changelog

## Unreleased

### Security

- Local `source.file_pattern` values must be relative and cannot contain `..`.
  Discovery never follows matched file or directory symlinks, hashes through
  verified file descriptors where supported, and gives parsers a hash-checked
  scratch snapshot to close discovery/fetch races. Project, source, and model
  YAML and project Python modules are likewise confined and cannot escape via
  symlinks.
- `llm.api_key_env` is now honored by synchronous extraction, Message Batches,
  and reusable helpers. The profile-owned variable wins deterministically,
  secrets never enter options/artifacts/errors, model YAML cannot select an
  arbitrary environment variable, and secret-value interpolation in
  `api_key_env` is rejected.
- `redact_pii` omits matched substrings from entity evidence by default,
  automatically drops a separately named input text column, and adds explicit
  `keep_fields` / `drop_fields`, `include_raw_text`, and `retain_input_text`
  controls. The customer-facing support-ticket example now uses an allow-list
  projection.
- `clean` no longer invokes adapter database/schema/dataset deletion. It
  removes only named local artifacts, preserves warehouses/caches/unknown
  files, rejects project-root or config-root overlap, and refuses every
  symlink component. The old `--force` option was removed.

### Reliability and correctness

- Incremental inputs now reject missing, NULL, or duplicate keys before
  mutation. DuckDB performs delete+insert and full-table publication in
  transactions; BigQuery stages each write under a unique name and uses one
  atomic `MERGE`. Cleanup errors no longer mask the primary warehouse error.
- Full extraction models publish only after every document succeeds; any
  parser/backend error preserves the prior target and state. Backend and
  zero-match warnings are retained in CLI output and `run_results.json`.
- Extraction `fields[].data_type` defines a typed output contract and enables
  successful zero-document runs to create real zero-row relations on DuckDB
  and BigQuery. Supported logical types are string, integer, float, boolean,
  date, timestamp, and JSON; contract changes invalidate incremental state.
- Selection now limits source discovery and watch paths to the selected graph,
  so unrelated GCS branches do not construct clients or consume credentials.
  Run artifacts record `sources_considered`.

### Compiler and blocker fixes

- All public configuration models reject unknown keys, source/model YAML is
  fixed at schema version 2, configured and dynamic extraction fields cannot
  overwrite lineage columns, and LLM numeric settings are bounded before any
  file, cloud, or API access.
- A shared preflight for compile/run/build/test/watch validates backend names,
  edge kinds, dependency counts, materializations, transform/custom-test
  modules and call signatures, built-in test specifications, and relationship
  targets. Relationship targets are DAG predecessors, so `build` orders them
  before the referencing test.
- GCS sources accept an explicit `project:` and missing ADC/project inference
  now exits 2 with actionable guidance instead of a raw traceback (#105).
- Tests against a genuinely missing model relation return a structured
  `relation_exists` failure rather than leaking an adapter-specific 404 (#106).
- `show` safely replaces unsupported characters on narrow Windows console
  encodings such as cp1252 (#107).
- Custom `llm.api_key_env` resolution is consistent at compile and runtime
  (#116), and redacted outputs no longer silently persist raw evidence (#115).

### Upgrade notes

- Unknown configuration keys that were previously ignored now fail. Correct
  misspellings and keep source/model file `version: 2`.
- LLM extraction models require the configured credential variable even when
  their response cache is warm. Put `api_key_env` under profile `llm:`, not
  model options.
- Use `include_raw_text: true` or `retain_input_text: true` only when the
  resulting relation is intentionally sensitive. Use `keep_fields` for
  customer-facing redacted tables.
- `dbt-ml clean` preserves the local DuckDB warehouse; use an explicitly
  scoped administrative workflow when warehouse relations must be dropped.

### Extraction (issue #108)

- html backend: two opt-in heading detectors for corpora that style headings
  instead of using `<h1>`–`<h6>` (SEC inline-XBRL filings):
  `styled_headings: true` heuristically treats short, fully-bold leaf blocks
  as headings with levels ranked by font size, and `heading_selectors:`
  accepts explicit CSS selectors with selector order setting the level.
- `sections` entries now carry a `source` field: `"tag"`, `"selector"`, or
  `"style"`. Both detectors are off by default; existing extractions are
  unaffected.

### Observability

- Backend extraction warnings (missing json fields, empty pdf pages, html
  selectors matching nothing) are no longer dropped: the runner aggregates
  them per model as distinct message → document count, `dbt-ml run`/`build`
  print a WARNING section under each model (capped at 5 distinct messages),
  and `run_results.json` carries the full counts per model plus a run-level
  `counts.warnings` total. Warnings never change the exit code.

## v0.2.7 - 2026-07-10

### Security (issue #65)
- Project-YAML paths are now confined to the project directory: source
  `path:`, `ml.artifact.path`, the layout paths (`source-paths`,
  `model-paths`, `transform-paths`, `target-path`), and model-level llm
  `cache_path` error (exit 2) when they resolve outside it — including via
  `..` and symlinks. Sources and artifact blocks opt out explicitly with
  `external: true`; external llm caches belong in profiles.yml.
- profiles.yml paths stay trusted (operator-local config), but `dbt-ml clean`
  now requires `--force` to delete a warehouse file outside the project
  directory.
- New Trust model & filesystem boundaries section in the README.
- **Upgrade note:** projects whose sources point outside the repo must add
  `external: true` to those sources. `artifact.external` and the boundary
  checks never change `code_version` — incremental state is unaffected.

### Scale (issue #77)
- Extraction models stream rows to the warehouse every `flush_every` documents
  (default 5000) instead of accumulating the whole corpus in memory.
  Incremental models upsert rows and state per flush, so a killed run keeps
  completed chunks and the re-run processes only the remainder; full models
  stream through a `dbt_ml_staging__*` table swapped in atomically at the end.
- New `WarehouseAdapter.materialize_full_chunks` (DuckDB + BigQuery
  implementations); staging tables are hidden from `list_tables`.
- `flush_every` is excluded from `code_version`, so tuning it never
  invalidates incremental state. Empty-corpus full models now drop the target
  table on both adapters (previously DuckDB errored).

### Observability (issue #75, part 2)
- Opt-in `batch: true` on `llm` extraction options routes uncached documents
  through the Anthropic Message Batches API (50% token cost, minutes-latency;
  sync stays the default). Per-document errors stay isolated, responses land
  in the LLM cache, and `estimated_cost_usd` applies the batch discount.
- New `BaseBackend.extract_batch` hook (default: sequential loop with
  per-document error capture).

### Observability (issue #75, part 1)
- LLM extraction records token usage per model: API calls, response-cache hits,
  input/output tokens, and prompt-cache read/write tokens. Totals land on
  `ModelRunResult.metrics`, in `run_results.json`, and as a summary line after
  `dbt-ml run`.
- Optional `pricing:` block in the profile `llm:` config (USD per million
  tokens, user-supplied — no prices ship with dbt-ml) adds `estimated_cost_usd`
  to those metrics.
- New `extract_fields_with_usage` alongside `extract_fields_from_text` for
  transforms that want token accounting; the original keeps its signature.

### Orchestration (issue #87)
- `run`/`build` exit codes now distinguish success (`0`), run failure (`1`), and
  configuration/usage error (`2`) so an orchestrator can branch on the cause.
  Malformed YAML is now reported as a config error instead of an uncaught trace.
- `run`/`build` gain `--json`, printing the `run_results.json` payload to stdout
  (identical to the on-disk artifact) for machine consumption.
- `run_results.json` carries run-level metadata (warehouse target, counts,
  status, elapsed) and per-model `status` + fully-qualified output `relation`;
  `build` records skipped downstream models as `status: "skipped"`.
- `emit-dbt-sources --dagster-meta` stamps `meta.dagster.asset_key` on each
  emitted source table so dbt-ml tables map cleanly onto `dagster-dbt` assets
  (pure dbt ignores the meta).
- New `docs/orchestration-dagster.md`: native `dagster-dbt` wiring — dbt-ml
  materializes the dbt source assets a `@dbt_assets` graph depends on, via
  `get_asset_keys_by_output_name_for_source` and `dbt-ml run --json`.

## v0.1.0 (unreleased)

Initial public preview.

### Backends
- `json` — project keys from JSON objects (deterministic, no API)
- `markdown` — frontmatter + body + word count
- `pdf` — text extraction via pypdf, with empty-text warnings for scanned PDFs
- `html` — body text, CSS selectors, OpenGraph, meta tags via BeautifulSoup
- `llm` — Claude-backed structured extraction with response caching

### Pipeline mechanics
- Declarative YAML: project, sources, extraction models, transform models
- DAG via `graphlib`, `ref()` syntax, cycle detection
- Incremental materialization keyed on content + code version
- `full` / `incremental` materialization
- `target/manifest.json` and `target/run_results.json` artifacts on every run

### CLI
- `init` (with `--template {json,pdf,markdown,html}`)
- `seed`, `compile`, `graph`, `run` (with `--full-refresh`), `test`, `show`, `clean`
- `source freshness` — mtime-vs-threshold check
- `emit-dbt-sources` — write dbt-compatible `sources.yml`

### Selection + filtering
- `--select` / `--exclude` with dbt-shaped syntax: name, `name+`, `+name`, `+name+`
- `tag:` prefix for tag-based selection
- `tags:` on models and sources

### Testing
- Built-in: `not_null`, `unique`, `min_rows`, `not_empty`
- Severity: `severity: warn` downgrades fail → warn (exit 0)
- Custom Python tests: drop `tests/<module>.py` with `run(con, table_ref) -> str | None`

### Profiles
- dbt-shaped `profiles.yml` with per-target warehouse + llm config
- Lookup: `--profiles-dir` → `$DBT_ML_PROFILES_DIR` → `<project>/profiles.yml` → `~/.dbt_ml/profiles.yml`
- `--target` flag selects within active profile
- LLM cache and model id come from profile, with per-model overrides

### Composition
- `dbt-ml emit-dbt-sources` writes dbt-compatible `sources.yml` so a
  `dbt-duckdb` project can `{{ source(...) }}` dbt-ml-materialized tables in the same DuckDB file
- Worked example in `examples/dbt_consumer/` (verified end-to-end with `dbt build`)
