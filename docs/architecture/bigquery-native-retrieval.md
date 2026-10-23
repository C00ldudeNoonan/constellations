# BigQuery-native retrieval design review

Status: design review complete for issue #618; implementation is deferred
pending the adopt-or-defer comparison in [Linear ALE-257](https://linear.app/alex-noonan/issue/ALE-257/decide-whether-a-warehouse-native-search-store-bigquery-vector-search).
This review builds on the [semantic retrieval architecture](semantic-retrieval.md)
and checks the issue's proposal against the current serving and publication
contracts.

## Decision

Do not add `store: bigquery` under this design issue. The original live-table
plus time-travel proposal does not preserve stel's durable serving-generation
contract. A table clone is a viable generation boundary, but it needs its own
asynchronous vector index. In the bounded experiment, queries remained on the
full-scan fallback and billed the full synthetic table after four minutes.
That is not enough evidence to choose BigQuery over the current LanceDB path
for the 3.6-million-row corpus.

Keep BigQuery-native retrieval as a candidate. Linear ALE-257 now owns the
comparison and final adopt-or-defer decision; its text-versus-hybrid replay
prerequisite, ALE-227, is complete. If ALE-257 recommends adoption, file a
separate implementation issue with a concrete publication, filtering,
generation-retention, and cost contract. The existing stel MCP retrieval
surface remains the governed product interface either way.

## Goal

Let BigQuery search the embedding rows already materialized by stel, instead of
copying the whole corpus into a second search system. Keep BigQuery's managed
vector index and query engine responsible for search work, while stel continues
to own model lineage, query authorization, serving readiness, and the public
retrieval API.

## Findings

### BigQuery covers more of the query surface than the issue originally states

As of 2026-10-23, Google documents `VECTOR_SEARCH` for exact brute-force and
approximate indexed queries. It supports cosine, Euclidean, and dot-product
distance. Vector indexes refresh asynchronously, and `VECTOR_SEARCH` includes
rows not yet indexed by brute-force search; `INFORMATION_SCHEMA.VECTOR_INDEXES`
exposes status and coverage. This is a reasonable basis for correctness plus an
operator-visible index-freshness signal.

Google also documents a **Preview** single-vector hybrid form of
`VECTOR_SEARCH` that combines semantic and lexical search. It is distinct from
`SEARCH()`, which is a boolean token predicate rather than a ranked text
retriever. The hybrid feature is covered by Pre-GA terms, so it must not be a
required production capability in the first stable slice. The issue's claim
that BigQuery cannot do hybrid search needs this qualification.

For governed queries, policy predicates must be part of the base-table query
passed to `VECTOR_SEARCH`, before ranking. An outer `WHERE` is a post-filter
and may return fewer than `top_k`; it is not an acceptable substitute for
policy-prefiltering. BigQuery can pre-filter against indexed stored columns;
the compiler must reject a configuration that cannot honor its declared
filtering semantics rather than weakening them silently.

Primary references:

- [BigQuery vector search and hybrid search](https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/search_functions)
- [Manage BigQuery vector indexes](https://docs.cloud.google.com/bigquery/docs/vector-index)
- [BigQuery search indexes and `SEARCH()`](https://docs.cloud.google.com/bigquery/docs/search)

### The current store contract assumes a separate serving projection

The accepted [semantic retrieval architecture](semantic-retrieval.md) says the
warehouse is the canonical row store and a `RetrievalStore` owns a separate
serving projection. `execution/search.py` reads a bounded upstream snapshot,
reconciles row fingerprints, writes typed `IndexedRow` batches to the store,
then advances warehouse publication state after durable receipts. The store
contract also assumes physical collections, private generations, activation,
and retirement.

A BigQuery-native target would instead search the upstream relation itself.
Implementing it as another ordinary `RetrievalStore` would either copy the rows
again or claim mutation and generation guarantees it does not provide. It is
therefore an architectural exception to the accepted boundary and needs an
explicit amendment before code is added. The first implementation should not
add a BigQuery class that returns success from no-op `upsert` or `delete`
methods: that would make existing durable-receipt and state-advancement
contracts untrue.

### Time travel alone cannot preserve the serving generation

The serving protocol keeps the old generation available until a new one is
validated and activated, and query pins keep readers on the generation they
were admitted to. A live-table-only design cannot provide that property while
the upstream table changes before the search model runs.

`FOR SYSTEM_TIME AS OF` can read historical table versions, but BigQuery time
travel is configurable only from two to seven days, and a deleted table cannot
be queried by time travel. A timestamp in the serving ledger would therefore
not be a durable generation identifier. It cannot replace stel's generation
and reader-safety guarantees by itself.

Primary references:

- [Query historical BigQuery table versions](https://docs.cloud.google.com/bigquery/docs/access-historical-data)
- [BigQuery time-travel retention](https://docs.cloud.google.com/bigquery/docs/time-travel)
- [ADR-0003: reader-safe online publication](../adr/0003-reader-safe-online-publication.md)
- [ADR-0035: session query pins](../adr/0035-session-query-pins-drain-without-expiring-readers.md)

## Proposed direction

Keep the warehouse relation as the canonical row set and have BigQuery manage
the search index, but preserve a separately addressable serving generation.
The leading option to investigate is a BigQuery table clone per search
generation, with the vector index created on the clone before the serving
ledger activates it. A bounded sandbox experiment confirmed that a clone is an
independently addressable row snapshot and can be queried with
`VECTOR_SEARCH`. The clone did not inherit the base table's vector index; it
needed its own index. The design remains unaccepted because that index was
still pending after a four-minute poll on a 50,000-row, 768-dimensional table,
so indexed readiness and usage were not verified.

Treat the native target as a capability of the active BigQuery warehouse
connection, not as a second profile with a duplicate set of credentials and
dataset settings. This keeps endpoint, location, execution project, and
credential resolution under the existing warehouse profile contract. The
public configuration spelling can still be `retrieval.stores.<alias>.type:
bigquery`, but its physical target must resolve from the selected warehouse
target and the upstream relation. A separate project or dataset target is out
of scope for the first slice.

## Validation gate before implementation

The first live check ran on 2026-10-23 in `sandbox-440119`, using only
synthetic data in a temporary dataset that was deleted afterward. It created a
10,000-row table with 256-dimensional vectors (about 20.8 MB), indexed it,
cloned it, and created a separate vector index on the clone. The source index
was not present on the clone: creating `idx_clone` succeeded, and the metadata
view showed separate indexes for `base` and `clone`. A source-only insert
changed the base count to 10,001 while the clone stayed at 10,000 and did not
contain the inserted row. Clone `VECTOR_SEARCH` returned five results both
before and after the source mutation.

The clone index remained at 0% coverage and queries reported
`PENDING_INDEX_CREATION` / `UNUSED` throughout a bounded wait of about one
minute. A follow-up used 50,000 rows at 768 dimensions (about 308.8 MB),
matching the issue's embedding width and exceeding its stated 5,000-row
minimum. After a four-minute poll, the index still had 0% coverage and
uncached queries reported `PENDING_INDEX_CREATION` / `UNUSED`; each query
returned ten results via fallback and billed 308,281,344 bytes. This is a
meaningful readiness and cost signal: the index DDL completing does not mean
the index is ready, and the fallback can scan the full table while it builds.

The experiments verify clone independence, separate index creation, and
queryability during index build, but not eventual completion time or indexed
query use. The query fallback is consistent with BigQuery's documented
behavior; production readiness must wait for positive coverage and query job
evidence that the index was used. Both scratch datasets and their tables were
deleted after the runs.

Use deterministic, credential-gated integration coverage to answer the
remaining questions:

1. Does a clone index reach the declared coverage/readiness threshold in a
   bounded and operationally acceptable time, and does an uncached query use
   it without a corpus-sized row copy?
2. Does the clone retain an independently addressable, queryable generation
   after later upstream materialization, and can it be retired safely after
   query pins drain?
3. Which index metadata, permissions, DDL retries, and asynchronous readiness
   states are observable through supported BigQuery APIs and
   `INFORMATION_SCHEMA` views?
4. Can policy predicates be applied before ranking for every supported
   attribute type, with the required fields stored in the vector index where
   BigQuery needs them for pre-filtering?
5. Can an exact query, an approximate query, and an explicitly opt-in Preview
   hybrid query preserve stel's result shape and scoring contract?
6. How are source-table replacement, failed index creation, interrupted
   publication, and reader pins coordinated so an unsuccessful run never
   exposes a partial generation?

If clone-plus-index cannot satisfy the existing generation contract, stop and
choose explicitly between retaining a second store copy and revising the
serving contract. Do not use the seven-day time-travel window as an implicit
retention policy.

## Initial supported surface, if the gate passes

- BigQuery warehouse targets only; no cross-project or cross-warehouse
  routing.
- Exact vector search and indexed approximate vector search, with
  configuration preflight against BigQuery's documented types and limits.
- Metadata and mandatory policy filtering only where the compiled query can
  apply predicates before ranking.
- Index readiness reported from BigQuery status and coverage metadata; an
  asynchronous index build is not reported as fully ready before the declared
  readiness condition is met.
- Hybrid search only behind an explicit opt-in while Google's implementation
  remains Preview. Standalone ranked text search is unsupported unless a
  separate ranked contract is established.
- Safe diagnostics expose stable operation/error categories and index status,
  never native error text, query literals, row data, or credential details.

This scope does not change shipped behavior. User-facing reference and profile
examples should be updated only when the implementation contract is accepted.

