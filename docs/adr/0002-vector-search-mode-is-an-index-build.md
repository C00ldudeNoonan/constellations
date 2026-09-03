# ADR-0002: The vector search mode is an index build, not an index identity

- **Status:** accepted
- **Amended by:** [ADR-0003](0003-reader-safe-online-publication.md) — the choice to
  apply the switch in place; it now builds into a private generation. The
  classification (search mode is an index build, not an identity) stands.
- **Date:** 2026-09-02
- **Prompted by:** #461

## Context

`search: exact` builds no vector index, so every query reads the whole vector
column. That is invisible at small scale and disqualifying at large. A
production collection of 3,613,979 rows x 768 float32 published cleanly, passed
post-publication validation, and reported `ready` — and then answered no
governed query at all, because a single scan cost ~275s and the context server
bounds `timeout_seconds` at 600s with a 30s default. The index was structurally
unservable by the serving path that stel itself ships.

Two things made that expensive rather than merely wrong. Nothing in the build
path mentioned the cost, at any scale. And `vector` was classified as one
opaque descriptor field, so changing only `search: exact` -> `approximate`
counted as whole-index invalidation: a new collection name, a full re-embed,
and a consumer cutover — 4h35m for this corpus — to change what is, physically,
an index-build flag over vectors that were already published and unchanged.

The cost of learning the choice was wrong therefore scaled with the corpus that
made it wrong, which is the worst possible shape for a default.

## Decision

Classify a change confined to `vector.search` as `COMPATIBLE`. Under
`on_index_change: online` it is applied to the live collection: the rows are
republished from the warehouse (no provider calls, no re-embed, no new
collection name) and `ensure_indexes` builds the ANN structure. Because `exact`
is implemented by the *absence* of an index, `ensure_indexes` now reconciles in
both directions and drops a vector index when the mode is `exact`.

Separately, warn at publish time. `retrieval/servability.py` estimates the
per-query scan from row count and dimensions and reports it when it exceeds the
default serving timeout — escalating for a governed index, which is reachable
only through a context server, and again past the server's absolute ceiling,
where no permitted setting can answer the query. A TIMEOUT raised at that
ceiling says so in its message, but stays retryable: see below.

### Reporting a ceiling timeout as `retryable: false`

#461 asked whether `retryable: true` is right for a timeout no retry can
satisfy. It is, from the serving layer. The limiter observes a deadline elapse
and nothing else, and the same expiry at 600s covers an oversized deterministic
scan, an approximate search behind a congested store, and a warehouse read that
was merely unlucky — two of which succeed on a retry. The serving layer knows
neither the collection's row count nor its search mode, so it cannot tell them
apart. Marking them all permanent to catch the first would trade a misleading
flag for a wrong one.

So the message carries what the server does know — whether a larger deadline is
still configurable — and the structural claim is made at publish time, where
the row count and the search mode are in hand.

## Alternatives considered

### Keep `vector` opaque and tell operators to republish

The status quo, and defensible on the grounds that vector configuration is
where correctness lives. It lost because it is not true of this sub-field:
`search` selects an access path over bytes that are already written, and no
stored row's meaning depends on it. Charging a re-embed for it is charging the
most expensive operation in the system for the cheapest kind of change.

### Exclude `vector.search` from `config_fingerprint` as well

Tempting, and strictly better in the steady state: a row's `input_fingerprint`
includes the config digest, so leaving `search` in it means an `online` switch
rewrites every row from the warehouse rather than skipping all of them and only
building the index.

It lost on migration cost. Removing a key from the hashed descriptor changes
the fingerprint of *every* published collection, whether or not anyone ever
switches modes — so every existing index would take exactly the full row
rewrite this change exists to avoid, once, on upgrade. The cost is the same;
the alternative just charges it to everybody. Worth revisiting only alongside
another change that already invalidates the stamp.

### Refuse `search: exact` above a row threshold

Rejected as overreach. `exact` is correct at any size, the threshold is an
estimate anchored to a single measurement on object storage, and a local NVMe
store is materially faster. A refusal would encode that estimate as a contract.
The warning conveys the same information and leaves the judgment where it
belongs.

### Raise the context server's `timeout_seconds` ceiling

Would have unblocked this one index and made the next one worse. Ten minutes is
already far outside what an interactive agent tool can wait for; the problem is
a query that costs eleven gigabytes of reads, not a limit set too low.

## Consequences

- `on_index_change: online` becomes the answer to "I picked the wrong search
  mode", which raises its profile. It still applies only to changes the
  classifier calls compatible.
- The switch still streams every row from the warehouse to recompute
  fingerprints, so it is not free — minutes to tens of minutes at millions of
  rows, against hours for a re-embed. The advisory says so rather than implying
  the switch is instant.
- `ensure_indexes` is now destructive in one narrow case: it drops a vector
  index when the mode is `exact`. That is required for `exact` to mean what it
  says, but it means a mode typo now costs an index rebuild on the next run.
- **A store that refuses an index must say so at compile time.** This was
  survivable while a vector-search change forced a rebuild into a private
  generation: the live collection was untouched. Now the change is applied in
  place, so a refusal discovered at `ensure_indexes` arrives after every row
  has been republished and the in-place claim has cleared the serving pointer.
  `RetrievalStore.index_config_refusal` is the seam — asked of the constructed
  store rather than its capability set, because DuckDB's
  `hnsw_experimental_persistence` is a property of the resolved config, not of
  the store type. A store adding a config-dependent refusal must implement it
  there, not only at index time.
- The throughput constant in `servability.py` is a single measurement, not a
  model. It will drift as stores and hardware move; it is stated as an estimate
  everywhere it surfaces, and the advisory names its source.

## Evidence

Measured on stel 0.15.4, LanceDB over GCS, BigQuery warehouse, reported in #461
on 2026-09-02:

```
connect + open_table :     0.6s
count_rows           :     0.0s   (3,613,979 rows)
vector search cold   :   279.5s
vector search warm #1:   272.8s
vector search warm #2:   273.4s
```

Warm is not faster, so this is scan cost, not cold start: 3,613,979 x 768 x 4B
= ~11.1 GB read per query, or ~40 MB/s effective. The same corpus was servable
at 1.04M rows and unservable at 3.6M with no signal in between, which matches
the scale-dependent failure described in #418. `lancedb.py` confirmed five
`BTree` scalar indexes and one `FTS` on the published table, and no vector
index.

The 4h35m republish figure is the reporter's measured full rebuild of this
corpus under `on_index_change: fail`.

## Amendments

- **#476 — the index *type* is an index-build field too.** `vector.index`
  (`ivf_hnsw_flat` | `ivf_hnsw_sq` | `ivf_pq`) joins `vector.search` in the
  set of fields classified `COMPATIBLE`, for the same reason: it selects a
  structure over vectors already published. The "exclude from the fingerprint"
  alternative this ADR rejected on migration cost is what forced the shape of
  the new field: its default is *omitted* by the model's own serializer rather
  than written, so no dump — descriptor, fingerprint, manifest, code version —
  moves, and only a deliberate choice is recorded. The measured build-memory ratios that motivated the field are in
  `docs/reference.md` under "Choosing the index `approximate` builds".
