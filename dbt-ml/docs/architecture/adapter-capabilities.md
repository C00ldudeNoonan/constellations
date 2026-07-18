# Adapter capability architecture

Status: accepted for issue #70; bounded snapshot reads implemented in issue
#140 and the first independent retrieval-store implementation in issue #134.

## Decision

Warehouse adapters and retrieval stores are separate roles.

A warehouse is the tabular system of record for model outputs, lineage columns,
incremental state, tests, transforms, and artifacts consumed by dbt. A retrieval
store owns serving-collection/index lifecycle and the typed vector, text,
filtered, and hybrid query modes it advertises. LanceDB therefore implements the
`RetrievalStore` contract; it does not implement `WarehouseAdapter` or
emulate SQL.

Search-index models will read their canonical rows and lineage from the
configured warehouse, then publish a serving projection to the configured
retrieval store. The warehouse remains usable when the retrieval store is
unavailable, and retrieval behavior is never inferred from warehouse behavior.

## Warehouse contract

`WarehouseAdapter` keeps the operations every tabular target must provide:

- lifecycle and target identity;
- full and keyed incremental materialization;
- bounded chunk writes and typed empty relations;
- relation listing and deletion;
- incremental-state CRUD.

Core reads small relations through typed operations such as `read_table()` and
`row_count()`. Serving-sink consumers use `table_snapshot()` for bounded
projected Arrow batches, typed predicate pushdown, same-snapshot key-domain
validation, and opaque safe snapshot and generation fingerprints. Raw SQL methods remain
available to SQL adapters, but callers must not assume that SQL is the only way
to implement a typed operation.

Incremental state uses a generic stable `record_key` within a `StateScope`
(model, stage, and safe target identity). Document-grain stages use
`document_id`; derived publication stages can use `chunk_id` or another stable
row key. Adapters atomically replace a complete scope and combine target-row
deletion with scoped state invalidation. Publication remains at-least-once:
materialize first, then advance only the successfully published record state.
Target-specific callers derive the stored identity from a semantic, non-secret
descriptor rather than persisting raw configuration or credentials.

Adapters declare a frozen set of `WarehouseCapability` values. Project
preflight checks the selected model workload before opening a connection. The
adapter also guards capability-dependent runtime methods, so direct library use
gets the same clear error rather than an attribute error or silent skip.

## Current capability matrix

| Capability | DuckDB | BigQuery |
| --- | --- | --- |
| SQL queries and references | yes | yes |
| typed tabular reads | yes | yes |
| immutable streaming tabular snapshots | yes | yes |
| typed predicate pushdown | yes | yes |
| SQL-backed schema tests | yes | yes |
| atomic full replacement | yes | no |
| atomic keyed upsert | yes | yes |
| multi-statement transactions | yes | no |
| typed empty relations | yes | yes |
| bounded chunk writes | yes | yes |
| additive schema evolution | yes | yes |

Capabilities describe implemented guarantees, not aspirations. Adapter sets are
explicit rather than derived from every enum member, so adding a future
capability cannot silently advertise unimplemented behavior.

## Bounded snapshot-read contract

`table_snapshot()` accepts a relation, optional projection, batch size, typed
predicates, and optional stable key column. It guarantees:

- memory bounded by warehouse result paging and the configured output batch,
  independently of total relation size;
- one Arrow schema available even for an empty relation, with every emitted
  batch checked against it;
- projection and AND-combined typed predicates compiled inside the adapter,
  with values passed as bound parameters;
- a warehouse-native NULL and uniqueness preflight over the same filtered
  snapshot when `key_column` is supplied; a projected read must include that
  key in its output columns;
- a one-shot iterator, deterministic context cleanup after exhaustion, error,
  or early close, and final generation validation after full consumption;
- an opaque 16-byte handle fingerprint plus a 16-byte generation fingerprint
  safe for state, cache, and artifact identity; and
- no ordering guarantee. Ordered state reconciliation is a separate capability
  tracked by issue #153.

DuckDB opens an independent cursor transaction. MVCC pins the relation version
through schema inspection, key validation, every Arrow batch, and the final
context validation; concurrent commits do not change the rows being read. A
bounded second scan hashes the current projected Arrow stream before successful
close, so a newer table version cannot silently become ready. The handle
fingerprint carries an opaque transaction-scoped identity and the generation
fingerprint hashes the schema and rows; neither exposes row values. This final
validation means a successful DuckDB snapshot reads the projected relation
twice. Its generation fingerprint is `None` until every primary batch has been
consumed; an early-close path cannot be published as a complete generation.

BigQuery executes one uncached query job and pages its immutable result through
the REST Arrow iterator. The configured page size also caps emitted batches.
Table `etag`/modification metadata is checked before the first page and after
full consumption; a changed generation fails rather than allowing a serving
publisher to mark the read ready. The fingerprint hashes the safe table
generation, query-job identity, and semantic read shape. A separate stable
generation fingerprint omits the query job. Query cost remains
subject to the active profile's project, priority, retry, timeout, and
`maximum_bytes_billed` settings.

`ReadPredicate` supports equality, inequality, ordered comparisons, membership,
and NULL checks over strict scalar values. Membership tuples are non-empty and
homogeneous. Reprs redact values, and adapter errors never include row or
predicate payloads.

The eager `read_table()` operation remains for small interactive and existing
model-runner paths. Issue #140 does not claim bounded memory for transform,
chunk, or classic-ML execution. Search-index publication introduced by issue
#134 must use `table_snapshot()` and keep it open until its upstream-generation
readiness check completes.

BigQuery does not declare atomic full replacement. A layout-changing replacement
must drop the existing target before renaming its staged replacement, so a rename
failure can temporarily leave the target unavailable.

## Unsupported workflows

- Transform, chunk, embed, and classic ML models require `tabular_reads`.
- Model tests require `sql_schema_tests` while the test engine remains SQL-based.
- Full and incremental models require their corresponding atomic publication
  capability.
- Extraction models require typed empty relations and bounded chunk writes.
- `on_schema_change: append_new_columns` requires schema evolution.
- `show` uses `read_table()` and reports an adapter capability error if typed
  reads are unavailable.
- Serving-sink publication requires `streaming_tabular_reads`; a pushed
  predicate additionally requires `tabular_predicate_pushdown`.

`compile` checks every model. Commands with selectors check the selected model
workload, so an unsupported unselected model does not block an otherwise valid
run.

## Retrieval store boundary

The original decision used `VectorStore` as a placeholder. The accepted
[semantic retrieval architecture](semantic-retrieval.md) broadens that role to
`RetrievalStore` so full-text, typed-filter, and hybrid capabilities do not have
to masquerade as vector behavior. The retrieval-store interface owns:

- collection and index lifecycle;
- atomic replacement and keyed record upsert;
- vector dimensions and distance metric validation;
- typed vector, text, filtered, and hybrid search when advertised;
- mandatory policy-prefilter enforcement when advertised; and
- deletion by stable model/record IDs.

It will not own schema tests, Python transforms, tabular incremental state, dbt
source emission, or arbitrary SQL. Provider and embedding model work can depend
on this boundary without adding vector-specific branches to warehouse code.
