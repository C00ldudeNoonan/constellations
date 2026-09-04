# ADR-0004: An index-only change fills its private generation from the store

- **Status:** accepted
- **Date:** 2026-09-04
- **Prompted by:** #495

## Context

Adding an ANN index over vectors that were already published and unchanged
re-read the whole corpus from the warehouse into a fresh generation. For
`sec_chunk_search` that was 3,613,979 rows across 145 pages, ~4.2 hours and a
full BigQuery read, to change one descriptor field — while the embed model
alongside it reported `documents_skipped: 3613979, rows_written: 0`. The
vectors were confirmed unchanged and cached; the search index rewrote all of
them anyway.

The constraint that shaped the fix is #474, recorded as
[ADR-0003](0003-reader-safe-online-publication.md). It made *every* compatible change
build into a private generation, because `online` had been mutating the live
collection under readers and taking search down (#473). That contract is
pinned by `tests/test_online_publication.py`, whose fixture transition is
exactly the `exact` -> `approximate` switch this issue is about. Any fix had
to keep the generation and the reader guarantees; the only thing on the table
was where the generation's rows come from.

## Decision

Keep the private generation. Classify a change confined to `vector.search` or
`vector.index` as reaching no row, and when a store advertises
`COLLECTION_SEEDING`, fill the new generation by copying the collection it
replaces — DuckDB with one streamed `INSERT ... SELECT`, LanceDB in bounded
batches — instead of streaming the warehouse. Three things move with the rows:

- the collection stamps a `row_fingerprint` that advances only when a change
  reaches a row, so carried rows keep the digest they were written under; a
  stamp without one falls back to `config_fingerprint`, which is what those
  rows were written with, so no existing collection is forced to republish;
- seeded publication state is restamped to the current `code_version`, which
  reconciliation compares alongside the fingerprint;
- a seeded rebuild reconciles deletions, which a plain rebuild skips only
  because it rewrites every row from upstream.

Seed only from a collection the ledger vouches for — one with an active,
validated generation — and never when the operator asked for the rebuild
outright (`--full-refresh`, `materialization: full`).

## Alternatives considered

### Build in place for index-only changes

The issue's first suggestion, and it looked right: no row is mutated, adding
an index does not degrade readers, and #491's evidence says an index build
alone peaks at 7 GiB of 26 GiB. It was implemented and it broke 13 of 17 tests
in `test_online_publication.py`. With only the routing change disabled and
every other piece intact, 17 of 17 passed, so the routing was the sole
conflict — and what it conflicted with was #474's contract, established the
day before, with #473's OOM half still open. Narrowing it to index *addition*
(never dropping or swapping an index in place, which would hand live readers
the ~275s full scan of #461) collided with the same fixture. Reversing a
day-old reader-safety contract to save a copy was not a trade worth making.

### Exclude index-only fields from `config_fingerprint`

ADR-0002's rejected alternative, rejected again for the same reason: removing
a key from the hashed descriptor changes every published collection's digest,
so every existing index would take one full row rewrite on upgrade. The
stamped `row_fingerprint` gets the steady-state benefit — carried rows keep
their digest — with a fallback that costs nothing, because a stamp without the
field can only have been written under `config_fingerprint`.

### Skip the warehouse entirely on an index-only change

Tempting, and it would make the switch a pure store operation. It lost because
an index switch can coincide with upstream changes, and only the warehouse
knows which rows moved or disappeared. `test_online_can_replace_with_an_empty_snapshot`
caught the concrete failure: without stale reconciliation, a document deleted
upstream came back on the next index change. One read pass, with no writes, is
the price of not resurrecting deletions.

## Consequences

- `append` is no longer the only write a rebuild performs. A seeded generation
  already holds every id it was copied with, so a row that changed upstream
  during the switch must `upsert` or it lands twice. Any new write path on the
  rebuild branch has to respect `seed_from`.
- Restamping seeded state to the new `code_version` is load-bearing and does
  not look it. Without it every carried row is classified `changed` and the
  build rewrites the corpus it just copied — the symptom is a doubled row
  count, two steps away from the cause.
- The seeding gate is the ledger's `active_generation`. Anyone changing what
  `failed` or `degraded` mean, or when the pointer is cleared, has to revisit
  it: seeding from a half-rewritten collection would launder the damage into a
  generation that activates as sound.
- The store contract grows `COLLECTION_SEEDING` and `seed_collection()`. A
  store without it keeps the warehouse path silently — correct, only slow.
- Tests that hooked `append` to observe or fail a private build now hook
  `seed_collection`, because on this path `append` is never reached.

## Evidence

Reported in #495 against stel 0.16.0 on 2026-09-04: 145 pages / 3,613,979
rows written into a new generation over ~4.2h before the index step, with the
embed model reporting `documents_processed: 0, documents_skipped: 3613979`.

The build-in-place attempt, 2026-09-04, in this repository: 13 of 17 failures
in `tests/test_online_publication.py` with the routing change applied; 17 of
17 with only that change disabled and the digest plumbing left in place.

Index build cost from #491, same corpus: peak RSS 7.0 GiB of 26 GiB, all seven
indices rebuilt by hand in 173s.
