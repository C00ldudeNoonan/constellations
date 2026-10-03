# ADR-0021: an index behind on rows is extended; a rebuild is for a shape change

- **Status:** accepted
- **Date:** 2026-10-03
- **Prompted by:** #619

## Context

The LanceDB store asked Lance to maintain nothing. `retrieval/lancedb.py` held
no call to `Table.optimize()`, `compact_files` or `cleanup_old_versions`, so
every incremental page committed a new table version and new fragments, none
of which were ever consolidated, and new rows stayed unindexed until
`ensure_indexes` ran at the end of the publish. One private generation was
observed at 395 table versions and the default collection at 237.

The index step itself was the expensive half. LanceDB's `create_index`
replaces: an index that was merely *behind* on rows was retrained over the
whole collection, so a monthly increment of tens of thousands of rows into a
3.64M-row corpus paid for the 3.64M. That retrain is also the operation that
exhausts Lance's DataFusion memory pool, which is what [ADR-0013](0013-a-merge-page-is-bounded-by-bytes-in-the-store.md)
and issue #636 are both about.

## Decision

Where every declared index merely lags the rows, extend them: `ensure_indexes`
calls `Table.optimize()` once and re-reads the listing, so the build loops find
nothing to do. Where an index has to change shape — a vector index of a newly
declared type, or an ANN index under `search: exact` — reach the build path
with the unindexed count intact and retrain. Extend nothing when nothing is
behind, so an unchanged rerun stays a metadata check. An extension that fails
falls back to the rebuild rather than failing the publish; an extension that
exhausts the memory pool is refused outright, because the rebuild it would
fall back to sorts strictly more rows through the same pool.

## Alternatives considered

### Compact periodically during the page loop

What #619 asked for first, on the theory that unbounded fragments were the
mechanism behind #616's per-page cost decay. Measured and declined. Compacting
every 100,000 rows did bound the table — 21 fragments and 21 versions became 2
and 2 over 20 pages — but per-page merge cost was identical with and without
it: 0.04s rising to 0.06s in both arms, across a four-fold difference in
fragment count. The cost it did add was real: 2.3x the page wall locally.
#616's decay had already been isolated to index maintenance on a *resumed*
generation and fixed by dropping the indices first ([ADR-0018](0018-a-resume-drops-its-indices-and-rebuilds-once.md)),
which is the better explanation of the same curve. A local filesystem is the
limit of this evidence: many small fragments on object storage mean many
object reads, and that penalty would not appear here.

### `optimize(retrain=True)`

The flag #619 named as the choice to make. It is deprecated in lancedb 0.34.0
and documented as "no longer used" — passing it does nothing, so there is no
in-place retrain to choose. `--full-refresh` is the retrain.

### `compact_files` and `cleanup_old_versions`

The methods #619 grepped for. Both were deprecated in lancedb 0.21.0 and
delegate through `to_lance()`, which needs pylance — not a dependency of this
project. `optimize()` is the only route to any of the three operations.

### Extend unconditionally, shape changes included

Simpler, and wrong. An extension cannot change an index's type, and `exact` is
implemented by the absence of an ANN index (issue #461), so extending would
quietly keep serving approximate results under a configuration that promises
exact ones.

### Prune every version but the latest

`cleanup_older_than=timedelta(0)` would collect the 395 versions immediately.
Declined for a collection that may be serving: it deletes files a reader could
still hold open. Lance's default window bounds growth across publishes instead
— a publish's own versions are hours old and survive it, and the next publish
collects them.

## Consequences

- **An approximate index that is only ever extended drifts.** New vectors are
  assigned to centroids an earlier build trained; as the corpus moves away
  from that sample, recall degrades without anything reporting it.
  `--full-refresh` is the only retrain, and nothing yet measures the drift or
  prompts for one. That is the sharp edge of this decision.
- **The in-place build advisory is gone**, because the build it predicted no
  longer happens. It survives for the two cases that do build from scratch: a
  private generation, and a first publish.
- **stel now prunes**, one publish behind. Storage no longer grows without
  bound, but a generation's own versions are not collected until the next
  publish is at least Lance's default window later.
- The rebuild path stays reachable through the fallback, so it keeps its
  retry budget, its pool-exhaustion refusal and its tests.

## Evidence

Measured 2026-10-03 on lancedb 0.34.0 (the pinned version), local filesystem,
Windows, against synthetic tables; see the probes recorded on #619.

- **Extension works.** A BTree and an FTS index over 20,000 rows, then 100,000
  rows appended: `num_unindexed_rows` went 100,000 to 0 after one `optimize()`,
  with no `create_index` call, and the version history collapsed from 11 to 3.
- **Extension is cheaper.** A 10,000-row increment onto bases of 50,000,
  100,000 and 200,000 rows with three indices (BTree, FTS, IvfPq): extension
  took 0.38s, 0.47s and 0.73s against 5.45s, 5.14s and 6.17s to rebuild all
  three — 14.5x, 11.0x and 8.4x — and both left nothing unindexed.
- **Fragments did not drive page cost.** 40 `merge_insert` pages of 10,000 rows
  with the `count_rows` acknowledgement, compacting every 100,000 rows and not:
  40 fragments against 10, and per-page time 0.04s to 0.06s in both arms.
- **`retrain` is deprecated** and `compact_files`/`cleanup_old_versions` were
  deprecated in lancedb 0.21.0, read from the installed package's source at
  0.34.0.
- Not reproduced at the scale that matters: the corpus behind #619 is 3.64M
  rows on GCS, and none of the above exceeded 210,000 rows locally.
