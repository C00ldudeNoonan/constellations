# ADR-0051: Warehouse-native vector search is a query-time read, not a retrieval store

- **Status:** accepted
- **Date:** 2026-11-04
- **Prompted by:** #747 (ALE-301), #618

## Context

The [BigQuery-native retrieval design review](../architecture/bigquery-native-retrieval.md)
(#618) examined a `store: bigquery` retrieval store: stel directs the process
and the warehouse is the index. It found that a store which searches the live
upstream table cannot keep the serving-generation contract of
[ADR-0003](0003-reader-safe-online-publication.md) and
[ADR-0035](0035-session-query-pins-drain-without-expiring-readers.md). Time
travel lasts two to seven days and cannot be queried once a table is deleted.
A table clone per generation can, but the clone needs its own vector index, and
in the bounded experiment that index was still at 0% coverage after four
minutes on 50,000 rows while queries fell back to a full scan.

The question that review left open (Linear ALE-257) has since been closed, and
its practical outcome is narrower than "adopt a store". The consumer that
wants BigQuery vector search needs one thing: filtered semantic search over an
embeddings table stel already published, for cross-company questions that a
token search cannot answer. It does not need stel to publish, version, pin or
retire anything for that read.

## Decision

Warehouse-native vector search is a **query-time read over the embeddings
table**, owned by the consumer's query tool. It is not a `RetrievalStore`, and
stel does not add `type: bigquery` for it.

stel's part is the embeddings table: the output of the `embed:` model, with its
embedding model and dimensions recorded in lineage. The read must meet these
conditions, which an implementation states and tests:

- **Same model and width.** The query vector is produced by the embedding model
  and dimensions the table was built with. A mismatch is refused, never
  coerced.
- **Policy before ranking.** Every filter is part of the base-table query
  passed to `VECTOR_SEARCH`. An outer `WHERE` is a post-filter, can return
  fewer than `top_k`, and is not an acceptable substitute. A filter that
  BigQuery cannot apply before ranking is rejected when the tool is configured,
  not weakened silently.
- **No generation claim.** The read sees the table as it is when the query
  runs. It carries no pin, no ledger entry and no `FOR SYSTEM_TIME AS OF`
  timestamp used as a retention policy. A query that overlaps a publish can see
  rows from either side of it.
- **Exact or indexed approximate, from BigQuery's own state.** The index is on
  the embeddings table itself and is maintained by BigQuery. Readiness is
  reported from `INFORMATION_SCHEMA.VECTOR_INDEXES` coverage, and rows the index
  has not covered are searched by brute force. Bytes billed per query is
  recorded alongside latency, since an uncovered index is a cost, not an error.
- **Hybrid is not required.** Google documents `VECTOR_SEARCH` hybrid as
  Preview. The first version is vector search with the same metadata filters as
  the existing token search.
- **Verified before it replaces anything.** The retrieval path it would replace
  stays as the rollback until the read clears the replay rule in ALE-362.

Access to the embeddings table is governed by the warehouse's own IAM. This
path does not pass through stel's MCP grants and typed-filter surface.

## Alternatives considered

- **A `type: bigquery` retrieval store with a clone per generation:** rejected
  for now. It preserves the generation contract, but it needs an amendment to
  the [semantic retrieval boundary](../architecture/semantic-retrieval.md)
  (the warehouse is the system of record and a store owns a *separate* serving
  projection), and its central assumption, that a clone's index becomes ready
  in acceptable time, is unproven. It would be built for one consumer. Revisit
  it when more than one warehouse-backed store is wanted; see the follow-up
  issue below.
- **A live-table store with time-travel pins:** rejected by the design review.
  Time travel is not a durable generation identifier.
- **A `bigquery` store whose `upsert` and `delete` succeed without doing
  anything:** rejected. It would make the durable-receipt and
  state-advancement contracts untrue.
- **Keep serving cross-company questions from LanceDB:** the status quo. It
  remains the rollback until the read is verified, and it keeps a second copy
  of vectors the warehouse already holds.

## Consequences

The read is cheaper to build and to retire, and it can ship without touching
the store registry, the collection descriptor or the serving ledger. It gives
up what a store would provide: generation pinning, stel's grants on this path,
and one portable retrieval API across stores. A caller that needs any of those
on a warehouse-native index has not been served by this decision, and that is
the case the follow-up issue exists to examine.

Nothing in stel's shipped behavior changes. The consumer's tool, its
configuration and its verification live with the consumer; this record fixes
the conditions it must meet so a later `store: bigquery` is built against known
constraints rather than rediscovered ones.
