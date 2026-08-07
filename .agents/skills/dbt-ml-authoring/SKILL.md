---
name: dbt-ml-authoring
description: Author dbt-ml projects and declarative data pipelines for unstructured documents. Use when creating, configuring, explaining, or troubleshooting dbt-ml sources, extraction models, transforms, chunk/embed/search flows, ML models, profiles, tests, selectors, artifacts, or dbt handoff; especially for a new project or a user-facing pipeline change.
---

# dbt-ml Authoring

Author a small, declarative pipeline that matches the user's data, warehouse, and privacy boundary. Treat dbt-ml as a standalone, dbt-shaped CLI; do not present it as a dbt package or adapter.

## Ground the work in the installed contract

1. Inspect an existing project's `dbt_ml_project.yml`, `profiles.yml`, `sources/`, `models/`, and `transforms/` before changing it.
2. For a new project, start with `uv run dbt-ml init <name> --template json|pdf|markdown|html` and modify its generated files rather than inventing a layout.
3. Run `uv run dbt-ml --help` and, when a source checkout is available, use its package README and closest example as the authority for feature syntax. Match the installed version; do not rely on generic dbt configuration or an assumed provider feature.
4. Select the narrowest fitting pipeline shape:

| Need | Start with |
| --- | --- |
| Folder or GCS documents become rows | `source` + extraction model |
| Derive deterministic fields/tables | Python or SQL transform over `depends_on` |
| Retrieval units and vectors | chunk model, then embed model |
| Queryable retrieval index | chunk/embed output, then a `search:` resource |
| Deterministic text features/classifier | `ml:` model and a persisted artifact contract |
| Structured inference | LLM extraction or a transform that explicitly declares `uses_llm: true` |

Use the closest shipped example as the starting point: `invoice_pipeline` for basic JSON, `pdf_invoice_pipeline` for PDF plus inference, `rag_chunks_pipeline` for retrieval, `classic_text_ml` for text ML, `sql_governed_chunks` for SQL, and `dbt_consumer` or `dbt_embed_duckdb` for dbt composition.

## Establish the project boundary

Keep project identity, paths, and runtime selection separate:

- Put `name`, `version`, `profile`, and project-controlled paths in `dbt_ml_project.yml`.
- Use `version: 2` in every source and model YAML file.
- Define a profile target for each environment in `profiles.yml`; warehouse paths, remote source overrides, provider selection, caches, and credentials belong there.
- Install exactly the extras the pipeline needs (`pdf`, `html`, `text`, `pii`, `bigquery`, `gcs`, `vertex`, `lancedb`, or `mcp`).
- Use a credential reference or operator-owned authentication flow in the profile. Never put a secret value, API key, service-account JSON, or provider credential choice in model YAML.
- Keep project YAML paths inside the project. Declare `external: true` only for an intentional external source root; it never permits `..`, absolute file patterns, or symlink traversal.

## Author the pipeline in dependency order

### 1. Declare a source

Give every source a stable name, a narrow path and file pattern, and only metadata/freshness that users need. Do not make a model find files itself.

```yaml
version: 2
sources:
  - name: vendor_invoices
    path: "./data/invoices/"
    file_pattern: "*.json"
    recursive: true
```

Use `source_paths` in a profile target when dev and prod need different source roots. Use `gs://bucket/prefix` only after installing/configuring the GCS integration; bound listings for very large prefixes.

### 2. Add one model kind per model

Use `source: ref('source_name')` for extraction models. Use `depends_on: [ref('upstream_model')]` for every derived model. A model has exactly one executable kind; split mixed work into DAG stages.

```yaml
version: 2
models:
  - name: raw_invoices
    source: ref('vendor_invoices')
    extraction:
      backend: json
      options:
        fields: [invoice_id, vendor, total]
    materialization: incremental
    fields:
      - {name: invoice_id, data_type: string}
      - {name: vendor, data_type: string}
      - {name: total, data_type: float}
    tests:
      - not_null: [invoice_id, vendor, total]
      - unique: invoice_id
```

Declare `fields:` for a stable typed extraction payload, especially when a source can be empty. dbt-ml supplies lineage/provenance columns; do not restate them unless the current contract requests it. Layer parser-specific behavior under the backend's `options` block and put domain logic in a downstream transform.

### 3. Make transforms explicit and testable

- Use a Python transform for Polars-based logic and a SQL transform for warehouse-native derivations. Keep input names in `depends_on`, not hidden in code.
- Preserve a clear input/output grain. For one-to-many outputs, retain the parent key and generate stable child keys before considering incremental materialization.
- Keep Python transforms and custom tests trusted and reviewable. Add the smallest module/API contract that solves the task; do not use a transform to conceal source access, credentials, or provider selection.
- Use a documents spine when a token-derived transform must retain documents with no tokens. Preserve intended dtypes in empty output.

### 4. Choose materialization deliberately

- Use `full` for non-deterministic, whole-table, or cross-parent logic.
- Use `incremental` only when identity and replacement semantics are clear. It skips unchanged source/parent records, not arbitrary changed output rows.
- Run `--full-refresh` after intentional contract changes that require rebuilding existing relations, such as incompatible extraction schema/layout changes.
- Use `--source-filter` only for orchestrated, partitioned incremental extraction. It is additive/upsert-only and does not reconcile deletions; schedule an unfiltered run for deletion reconciliation.
- Keep BigQuery `insert_overwrite` for batches that contain every document in each touched partition. Use `merge` when that contract is not guaranteed.

## Protect data and operator controls

- Assume remote LLM and hosted embedding calls send selected document text to the configured service. State that boundary before enabling them on sensitive data.
- Treat caches, artifacts, retrieval indexes, and output tables as potentially sensitive stores. Project/drop raw PII rather than assuming a redacted derivative makes retained original columns safe.
- Keep prompts and schemas intentional, typed, and minimal. Review provider cost/budget, batching, and caching settings in profiles before bulk runs.
- Preserve `--json` stdout for automation; use stderr/log capture for human progress. Use `-v` for safe INFO-level progress rather than expecting debug traces.

## Compile, run, inspect, and test

Use a small feedback loop after each authored stage:

```bash
uv run dbt-ml compile
uv run dbt-ml run --select <model-or-selector>
uv run dbt-ml show <model>
uv run dbt-ml test --select <model-or-selector>
```

Use `dbt-ml graph` to verify lineage and `target/manifest.json` / `target/run_results.json` to inspect compiled configuration, run counts, warnings, and safe failure details. Use selectors to limit work; `state:modified+` is appropriate for CI when a prior manifest is available.

Before handing off an authored pipeline, provide the project files changed, required extras/environment variables, a command sequence that succeeds from a clean checkout, output relations/grain, schema tests, remote-data or PII boundaries, and the intended full versus incremental behavior.
