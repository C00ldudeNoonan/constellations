# Adapter capability architecture

Status: accepted for issue #70.

## Decision

Warehouse adapters and vector stores are separate roles.

A warehouse is the tabular system of record for model outputs, lineage columns,
incremental state, tests, transforms, and artifacts consumed by dbt. A vector
store owns vector index lifecycle and nearest-neighbor search. LanceDB therefore
implements the future `VectorStore` contract; it does not implement
`WarehouseAdapter` or emulate SQL.

Embedding models will publish their tabular rows and lineage to the configured
warehouse, then publish vector ids and vectors to the configured vector store.
The warehouse remains usable when the vector store is unavailable, and vector
search is never inferred from warehouse behavior.

## Warehouse contract

`WarehouseAdapter` keeps the operations every tabular target must provide:

- lifecycle and target identity;
- full and keyed incremental materialization;
- bounded chunk writes and typed empty relations;
- relation listing and deletion;
- incremental-state CRUD.

Core reads relations through typed operations such as `read_table()` and
`row_count()`. Raw SQL methods remain available to SQL adapters, but callers
must not assume that SQL is the only way to implement a typed operation.

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
| SQL-backed schema tests | yes | yes |
| atomic full replacement | yes | no |
| atomic keyed upsert | yes | yes |
| multi-statement transactions | yes | no |
| typed empty relations | yes | yes |
| bounded chunk writes | yes | yes |
| additive schema evolution | yes | yes |

Capabilities describe implemented guarantees, not aspirations. Streaming reads
with projection and predicate pushdown are not declared until a typed streaming
read API exists.

BigQuery does not declare atomic full replacement. A layout-changing replacement
must drop the existing target before renaming its staged replacement, so a rename
failure can temporarily leave the target unavailable.

## Unsupported workflows

- Transform, chunk, and classic ML models require `tabular_reads`.
- Model tests require `sql_schema_tests` while the test engine remains SQL-based.
- Full and incremental models require their corresponding atomic publication
  capability.
- Extraction models require typed empty relations and bounded chunk writes.
- `on_schema_change: append_new_columns` requires schema evolution.
- `show` uses `read_table()` and reports an adapter capability error if typed
  reads are unavailable.

`compile` checks every model. Commands with selectors check the selected model
workload, so an unsupported unselected model does not block an otherwise valid
run.

## Vector store boundary

The future vector-store interface will own:

- collection and index lifecycle;
- atomic replacement and keyed vector upsert;
- vector dimensions and distance metric validation;
- nearest-neighbor search with explicit filter capability;
- deletion by stable model/vector ids.

It will not own schema tests, Python transforms, tabular incremental state, dbt
source emission, or arbitrary SQL. Provider and embedding model work can depend
on this boundary without adding vector-specific branches to warehouse code.
