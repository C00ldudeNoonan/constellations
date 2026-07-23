# Warehouse-native SQL transform models

Status: accepted design (issue #141). Not yet implemented — the first
executable slice (full-refresh SQL models) lands in issue #143. This document is
the architecture decision that #143 implements without reopening design
questions. Incremental SQL models (#142) are a separate, later contract.

dbt-ml currently supports only `transform.type: python`: the runner reads each
upstream relation into Polars via `adapter.read_table`, calls a Python module,
and writes the returned DataFrame back with `adapter.materialize_full`. That is
the wrong shape for transformations that are naturally relational — joining
chunks to registries, tenants, ACLs, and taxonomies; normalization,
deduplication, effective dating, masking, aggregation; and processing large
warehouse tables without pulling them through Python.

This document defines a first-class, dbt-shaped SQL transform that runs **inside
the warehouse** and never moves rows through the dbt-ml process, without turning
dbt-ml into a dbt-core reimplementation.

## 1. Resource and file contract

SQL is a second implementation of the existing **transform** resource, not a new
top-level model kind. A transform already means "produce a tabular relation from
upstream models"; SQL and Python are two ways to express that, so selectors,
DAG, docs, model-kind, and materialization behavior are inherited unchanged.

```yaml
- name: governed_chunks
  transform:
    type: sql
    path: sql/governed_chunks.sql
  materialization: full          # only `full` in this slice (#143); see #142
  fields:
    - {name: chunk_id, data_type: string}
    - {name: access_groups}
    - {name: is_public, data_type: boolean}
  tests:
    - unique: chunk_id
    - not_null: [chunk_id, document_id]
```

```sql
-- sql/governed_chunks.sql
select
    c.*,
    p.tenant_id,
    p.access_groups,
    coalesce(p.is_public, false) as is_public
from {{ ref('document_chunks') }} as c
left join {{ ref('document_permissions') }} as p
  on c.document_id = p.document_id
```

Decisions:

- **`transform.type: sql`**, with a new `path` field on `TransformConfig`
  (`src/dbt_ml/config/model.py`). Exactly-one-of is enforced by a Pydantic
  `model_validator`: `type: python` requires `module` and forbids `path`;
  `type: sql` requires `path` and forbids `module`. Unknown `type` values keep
  failing in the compiler as they do today.
- **External `.sql` file only** in this slice. Inline SQL is deferred; reserving
  a future `query:` field is a non-breaking addition. The `path` is
  project-relative and routed through `resolve_within_project` (the same
  confinement `module`, artifacts, and layout paths already use in
  `paths.py`) — no parent traversal, no symlink escape, no absolute paths. The
  extension must be `.sql` (case-insensitive); a missing file, wrong extension,
  or directory is a compile-time `ConfigError` with file/line/column.
- **Files live beside models** by convention (`model-paths` or a project
  `sql/` subdir); dbt-ml does not impose a dedicated directory. The path in YAML
  is authoritative.
- **Model kind** in the manifest/docs is `transform` with an
  `implementation: sql` discriminator so tooling can tell SQL and Python
  transforms apart without a new node type.

## 2. Compilation surface

SQL is compiled with a **deliberately narrow, sandboxed** Jinja environment —
`jinja2.Environment(undefined=StrictUndefined, autoescape=False)` with **no**
globals, filters, macros, `include`/`import`, or Python attribute access beyond
the functions listed below. This is not dbt's Jinja; broad dbt macro/package
compatibility is an explicit non-goal (see Out of scope).

Exposed template API (v1):

- **`ref('model')`** — resolves through the active adapter to a safely quoted,
  fully-qualified target relation and records a lineage edge. `ref` takes one
  string literal naming another model in this project; a missing target or a
  cycle fails before any warehouse mutation. Non-literal / computed refs are
  rejected.
- **`target`** — a small read-only object exposing `target.name` and
  `target.type` (adapter type, e.g. `duckdb`/`bigquery`) only. No paths, no
  credentials, no profile values.

Deferred and explicitly **not** available in v1: `source()`, `var()`, `this`,
`is_incremental()`, and any macro. `this`/`is_incremental()` are meaningless
until incremental SQL (#142); `source()` and `var()` are additive later without
breaking v1 SQL. Referencing any undefined name is a compile error.

**Two-phase compilation** (mirrors the DAG discovery the runner already does for
Python `depends_on`):

1. **Parse phase** — before target resolution, statically discover every
   `ref('…')` call to build DAG edges and validate the graph. Because refs must
   be string literals, this is a safe AST/lexical scan of the rendered-with-
   stubbed-`ref` template, matching #143's "no dynamic refs" rule.
2. **Compile phase** — after the target adapter is resolved, render each `ref`
   into its quoted relation. Compilation is deterministic for a given project +
   target.

**`depends_on`:** derived entirely from `ref()`. An explicit `depends_on` on a
SQL model is optional; if present it must be a superset-agreement check (every
declared dep must also appear as a `ref`, and vice versa) — a mismatch is a
compile error. This keeps one source of truth (the SQL) while allowing the
existing field for readability.

Secrets, environment lookups, arbitrary Python objects, and profile values are
never reachable from a template. Compiled SQL is an audit artifact (§6) and must
contain no credential-bearing values.

## 3. SQL statement boundary

A SQL model is **exactly one `SELECT` that produces one relation**; dbt-ml owns
target creation and replacement. Model-authored DDL/DML, multiple statements,
transaction control, CTEs-plus-side-effects, and hooks are out of scope for the
first slice.

Validation is layered, cheapest first:

1. **Structural pre-check (core, dialect-agnostic):** after compilation, verify
   the statement is a single top-level statement and is a `SELECT`/`WITH … SELECT`
   (no trailing `;`-delimited second statement, no leading DDL/DML keyword). This
   is a lightweight lexical guard, not a full parser, and runs before any
   connection.
2. **Adapter dry-run (authoritative):** the adapter validates the compiled
   `SELECT` against the real dialect without materializing rows — DuckDB via
   `PREPARE`/`EXPLAIN` or a `LIMIT 0` describe; BigQuery via a `dryRun` job. The
   dry run also yields the output schema for optional contract checking (§7).

Project SQL is trusted repository code (consistent with Python transforms and
custom tests already executing as trusted code). The trust boundary this slice
enforces is not "sandbox the author's SQL" but "core never interpolates unsafe
values into identifiers, and never emits multiple/side-effecting statements on
the author's behalf." Target identifiers, `ref` relations, and `target.*` are
adapter-quoted; the author's `SELECT` body is passed to the adapter verbatim.

## 4. Adapter contract

Core must never assemble dialect-specific `CREATE TABLE AS` / replace SQL. A new
typed adapter operation owns quoting, staging, replacement, physical layout, and
dry run:

```python
def materialize_sql_full(
    self,
    table: str,
    select_sql: str,
    *,
    options: BaseModel | None = None,
) -> SqlMaterializationResult: ...

def dry_run_sql(self, select_sql: str) -> SqlRelationSchema: ...
```

- `select_sql` is the compiled single `SELECT`. The adapter wraps it in its own
  atomic replacement (DuckDB: `CREATE OR REPLACE TABLE … AS`; BigQuery: the same
  atomic `CREATE OR REPLACE TABLE … AS`, falling back to staged replace only for
  the partition-spec-change case already handled by `materialize_full` in the
  BigQuery adapter). `options` is the adapter's parsed `warehouse_options`,
  applied only on (re)create — identical to the existing `materialize_full`
  contract.
- `SqlMaterializationResult` carries `rows_written`, the fully-qualified target
  `relation`, and adapter/job metadata (e.g. BigQuery job id, bytes processed)
  for `run_results`.
- **New capability `SQL_MODEL_MATERIALIZATION`.** `SQL_QUERIES` (read queries) is
  insufficient: being able to run a `SELECT` does not imply safe, atomic
  CTAS/replacement. Preflight requires `SQL_MODEL_MATERIALIZATION`; DuckDB and
  BigQuery declare it, and a model is rejected at compile time on adapters that
  do not. Failure atomicity is gated on the adapter also declaring
  `ATOMIC_FULL_REPLACE` (both do) — a failed query then leaves the previous full
  target intact.

DuckDB and BigQuery may accept different dialect features in the author's
`SELECT`; the contract does not promise a universal SQL dialect, only a uniform
materialization/lineage/artifact boundary.

## 5. Governance and trust boundary

Two distinct governance layers, documented so SQL models are not mistaken for an
authorization mechanism:

1. **Build-time:** SQL produces deterministic **policy metadata** — tenant,
   access group, classification, effective dates, public/private flags — as
   ordinary typed columns. This is reproducible and auditable.
2. **Serve-time:** retrieval/serving still applies mandatory policy filters at
   query time. SQL preparation **does not** replace runtime authorization; a
   chunk carrying `access_groups` is not "safe" until the retrieval layer filters
   on it. This mirrors the fail-closed boundaries in the semantic-retrieval
   contract.

Compiled SQL and lineage are audit artifacts. Per the credential invariants in
`AGENTS.md`, source SQL, compiled SQL, manifest, and `run_results` must never
contain resolved secrets, credential-bearing profile values, or unsafe Jinja
globals — enforced by the narrow template surface (§2), which has no way to reach
them.

## 6. Versioning and artifacts

- **`code_version`** for a SQL model hashes the **raw source-SQL file content**
  plus the model's semantic config (fields, tests, materialization, tags,
  warehouse_options) — the SQL-model analogue of the Python module content hash.
  The **compiled**, target-specific SQL does **not** enter `code_version`:
  state/selection must not churn merely because the DuckDB vs BigQuery relation
  spelling differs. Changing the `.sql` file or referenced config re-selects the
  model under `--state`.
- **Manifest** per SQL model adds: `implementation: sql`, raw `path`, source
  hash, the compiled SQL (or a pointer to a compiled artifact), `adapter_type`,
  the resolved `ref` list, and the safe fully-qualified target relation. No
  secrets.
- **`run_results`** adds query/job metadata (rows written, and adapter job
  identifiers / bytes processed where available) alongside the existing
  per-model fields.
- **Docs** render the model as a transform with its compiled SQL viewable/
  downloadable and its `ref` lineage in the DAG, consistent with existing pages.
- **`dbt-ml compile`** writes the compiled SQL + dependencies so authors can
  inspect source SQL, compiled SQL, dialect/target identity, and refs **without
  running** the model.

## 7. Contracts and tests

SQL outputs are ordinary warehouse relations, so existing machinery applies
unchanged:

- schema contracts and model tests run **after** materialization exactly as they
  do for Python transforms;
- compile/preflight validates `SQL_MODEL_MATERIALIZATION` (and, for atomicity,
  `ATOMIC_FULL_REPLACE`) **before** opening a connection;
- declared `fields` can optionally be checked against the adapter dry-run output
  schema (§3.2) at compile time — a typed-empty/dry-run contract check that
  catches drift before materialization;
- when the adapter declares atomic replacement, a query failure preserves the
  previous full target.

## What this unblocks and what it defers

Implemented by #143 (first slice): the grammar above, `ref()` compilation +
lineage, the single-`SELECT` boundary, `materialize_sql_full` +
`SQL_MODEL_MATERIALIZATION` on DuckDB and BigQuery, the raw/compiled artifacts,
and `materialization: full` only.

Explicitly deferred (unchanged from #141's scope):

- Incremental `is_incremental()` / `this` semantics and merge/partition
  strategies (#142).
- Inline SQL, `source()`, `var()`, and any macro/package compatibility.
- Executing SQL against a retrieval store (LanceDB, turbopuffer, …).
- Warehouse-specific ML/embedding SQL functions.
- Author-owned DDL/DML, multiple statements, and pre/post hooks.
