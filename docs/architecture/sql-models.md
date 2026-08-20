# Warehouse-native SQL transform models

Status: accepted design (issue #141), implemented for full-refresh (#143) and
incremental (#142) SQL transforms. This document is the architecture decision
those issues implement without reopening design questions.

stel supports both `transform.type: python` and `transform.type: sql`. Python
transforms read upstream relations into Polars; SQL transforms keep relational
work in the warehouse. The latter is the right shape for joining chunks to
registries, tenants, ACLs, and taxonomies; normalization, deduplication,
effective dating, masking, aggregation; and processing large tables without
pulling them through Python.

This document defines a first-class, dbt-shaped SQL transform that runs **inside
the warehouse** and never moves rows through the stel process, without turning
stel into a dbt-core reimplementation.

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
  materialization: incremental
  unique_key: chunk_id
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
  (`src/stel/config/model.py`). Exactly-one-of is enforced by a Pydantic
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
  `sql/` subdir); stel does not impose a dedicated directory. The path in YAML
  is authoritative.
- **Model kind** in the manifest/docs is `transform` with an
  `implementation: sql` discriminator so tooling can tell SQL and Python
  transforms apart without a new node type.

## 2. Compilation surface

SQL is compiled with a **deliberately narrow, sandboxed** Jinja environment —
`jinja2.sandbox.SandboxedEnvironment(undefined=StrictUndefined, autoescape=False)`.
A plain `jinja2.Environment` is **not** sufficient: it ships default filters and
permits attribute traversal, so `{{ target.__class__.__init__.__globals__ }}` (or
similar) could reach Python internals from any exposed object. The
`SandboxedEnvironment` blocks access to underscore/internal attributes and unsafe
callables; on top of it the default globals and filters are cleared and only the
API below is added, and `include`/`import`/macros are unavailable. This is not
dbt's Jinja; broad dbt macro/package compatibility is an explicit non-goal (see
Out of scope). The exposed `target` object is a frozen, string-only value with no
methods, so even under the sandbox it carries nothing sensitive.

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

A SQL model is **exactly one `SELECT` that produces one relation**; stel owns
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
  plus the model's semantic config (fields, tests, materialization, tags) — the
  SQL-model analogue of the Python module content hash. It excludes
  `warehouse_options`, matching the existing `compute_code_version` contract
  (issue #91): partitioning/clustering/labels/TTL shape physical layout, not row
  content, and a layout change is applied with `--full-refresh` regardless — so
  including them would fire `state:modified` on layout-only edits. The
  **compiled**, target-specific SQL also does **not** enter `code_version`:
  state/selection must not churn merely because the DuckDB vs BigQuery relation
  spelling differs. Changing the `.sql` file or referenced semantic config
  re-selects the model under `--state`.
- **Manifest** per SQL model adds: `implementation: sql`, raw `path`, source
  hash, the compiled SQL (or a pointer to a compiled artifact), `adapter_type`,
  the resolved `ref` list, and the safe fully-qualified target relation. No
  secrets.
- **`run_results`** adds query/job metadata (rows written, and adapter job
  identifiers / bytes processed where available) alongside the existing
  per-model fields.
- **Docs** render the model as a transform with its compiled SQL viewable/
  downloadable and its `ref` lineage in the DAG, consistent with existing pages.
- **`stel compile`** writes the compiled SQL + dependencies so authors can
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

## 8. Incremental materialization (#142)

`materialization: incremental` extends the same `transform.type: sql` grammar
with a required `unique_key:` (validated at compile time; forbidden on `full`
SQL models and on python transforms, which have their own fixed-identity
incremental path). `on_schema_change` is the existing model-level field, applied
identically to the DataFrame-sourced incremental path.

**Template surface:** `is_incremental()` (zero-arg) and `this` (the target's own
quoted relation) are added to the sandbox, but only when the model is configured
incremental — a `full` model that references either gets a `StrictUndefined`
error, since neither is meaningful outside an incremental branch. Both are
discovered by the same AST scan as `ref()`, so a dynamic/argument-taking
`is_incremental(...)` is rejected before compilation.

**Compile-time truth, not a runtime branch inside the query:** the runner
decides, once per run, whether the model is *actually* running incrementally —
`is_incremental()` renders `True` only when the target already exists **and**
`--full-refresh` is not active — and compiles the `.sql` file accordingly. On
the first run or under `--full-refresh`, it renders `False` and the model is
materialized via the plain `materialize_sql_full` CTAS (the same path a `full`
model uses); the compiled SQL for that run is simply whatever the template's
non-incremental branch produces. Only when the target exists and the run isn't
a full refresh does the runner compile the incremental branch and call
`materialize_sql_incremental`. This keeps `is_incremental()` truthful — it never
diverges from what the run actually did — at the cost of compiling the SQL text
once per run rather than emitting both branches as separate artifacts (the
"full" and "incremental" compiled SQL are both recoverable by running
`stel compile` before and after the first materialization, so nothing is
lost, just not simultaneously present in one manifest).

**Adapter contract:** a new `SQL_INCREMENTAL_MATERIALIZATION` capability gates
`materialize_sql_incremental(table, select_sql, *, unique_key, on_schema_change,
options)`, called only when the target already exists (the adapter never needs
an `IF NOT EXISTS` fallback — the runner's branch above guarantees that). Before
mutating anything, the adapter validates `unique_key` is non-null and unique in
`select_sql`'s result with a single portable aggregate query
(`sql_models.build_key_check_sql`) that never returns row payloads to Python.
DuckDB stages the compiled query into a session-scoped temp table, then runs a
transactional delete-matching-keys + insert (the same idiom as the existing
DataFrame `materialize_incremental`); BigQuery issues one `MERGE ... USING
(select_sql) ...` statement, which is atomic as a single DML job. Both report a
best-effort `rows_inserted`/`rows_updated` split — "matched" counts as "updated"
even when column values happen to be unchanged, matching the existing
DataFrame-incremental convention rather than claiming a true row-level diff.

**Versioning:** `unique_key` and `on_schema_change` are folded into `code_version`
(scoped to the SQL-transform payload block only, so no other model kind's state
is affected) — changing either re-selects the model under `state:modified` even
if the `.sql` file is untouched. `warehouse_options` stays excluded, unchanged
from §6.

**Deferred from this slice:** composite unique keys; a configured deletion
strategy for source rows removed upstream (ordinary merge does not infer
deletions); `insert_overwrite`/partition-replace strategies; concurrent-run
serialization guarantees beyond each adapter's native transaction/DML atomicity.

## What this unblocks and what it defers

Implemented: the grammar in §1–§7 and full-refresh materialization (#143); the
incremental grammar and materialization in §8 (#142); `materialize_sql_full` /
`materialize_sql_incremental` and their capabilities on DuckDB and BigQuery.

Explicitly deferred (unchanged from #141's scope):

- Inline SQL, `source()`, `var()`, and any macro/package compatibility.
- Executing SQL against a retrieval store such as LanceDB.
- Warehouse-specific ML/embedding SQL functions.
- Author-owned DDL/DML, multiple statements, and pre/post hooks.
- Composite unique keys, CDC/deletion strategies, and partition-replace
  incremental strategies (see §8).
