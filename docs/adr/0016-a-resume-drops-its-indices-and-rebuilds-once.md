# ADR-0016: A resumed generation drops its indices and rebuilds them once

- **Status:** accepted
- **Date:** 2026-10-02
- **Prompted by:** #616

## Context

A fresh private generation never maintains an index while it is being
written. The page loop is `while True:` at `execution/search.py:523` and
`ensure_indexes` is at `:817`, *after* it, so every page merges into a table
with no indices and the whole set is built once at the end.

Resuming broke that order without anyone deciding to. `resumable_generation`
adopts the collection an earlier attempt left behind, and nothing dropped the
indices that attempt had built — so a resumed publish paid to maintain them on
every page, for the rest of the run.

#616 measured a prod publish: per-page cost rising monotonically, roughly
doubling by page 73 of 146, with memory flat. It listed three candidate
causes — fragment accumulation, the acknowledgement scan, incremental index
maintenance — and measured none.

A local probe separated them. 30 pages × 1,500 rows, 768-dim, BTree on the id
field plus two attribute BTrees plus FTS, same data and page order in all four
conditions:

| condition | first quarter | last quarter | ratio |
| --- | --- | --- | --- |
| no indices, with ack scan | 0.05s | 0.05s | **1.09x** |
| no indices, no ack scan | 0.04s | 0.05s | **1.17x** |
| indices, with ack scan | 0.08s | 0.34s | **4.14x** |
| indices, no ack scan | 0.09s | 0.33s | **3.60x** |

Table version count rose identically in all four (2 → 34). That rules out
fragment and manifest accumulation as the driver: the same accumulation is
flat in one pair and 3.6x in the other. Removing the acknowledgement scan
moved the ratio 4.14x → 3.60x, so it is a term and not the cause.

The only difference between the pairs is whether indices exist.

## Decision

**A resume drops its indices before the page loop, and `ensure_indexes`
rebuilds them after the last page** — the order a fresh build already uses.
`RetrievalStore.drop_indexes(collection)` is the new seam; both in-tree stores
implement it.

Measured against doing nothing, same pages and same final row count:

| order | merges | index build | total |
| --- | --- | --- | --- |
| keep indices (resume before this) | 6.18s | 0.27s up front | **6.45s** |
| drop, rebuild at the end | 1.39s | 2.65s at end | **4.04s** |

37% faster over 29 pages, and the gap widens with page count: maintenance is
per page, the rebuild is once.

**A complete resume does not drop.** When the rows are already complete and
only the index build remains (#508), dropping first would turn a metadata
check into a full rebuild — making the cheapest resume the most expensive one.

**A fresh build does not drop.** It is created empty and unindexed, so the
call would be a pointless store round trip on the publish path.

## Alternatives considered

### Compaction every N pages

#616's first hypothesis, and the one the data rules out. Versions and
fragments accumulated identically in the flat conditions and the decaying
ones. Compaction would cost I/O on every page group to address something that
was measured not to be the cause. Still available if a *fresh* build is ever
observed decaying, which is the shape that would implicate it.

### Drop the acknowledgement scan, or sample it

Also from #616. It is a real term — 4.14x → 3.60x — but it is not what makes
the curve. And it is the thing that earns LanceDB its
`EXACT_MUTATION_RECEIPTS` claim (ADR-0013): the publish gates state on a
receipt that is never ahead of the store, and the scan is how that is
established. Removing it to save the smaller half of a cost would trade a
correctness proof for a performance term.

Worth revisiting on its own terms: 25,000 literals against 3.6M rows is not
represented by the probe's scale, so it may be larger in prod than measured
here.

### Keep the indices and accept the cost

The status quo, and defensible if a resume were rare. It is not: a resume is
what happens after *any* failed attempt at a multi-hour publish, so the runs
that most need to be fast are exactly the ones paying this. At 146 pages the
extrapolated publish is ≥14.7h against ~9h flat.

### Do not defer on a fresh build either — build indices up front everywhere

The symmetric option, and worse in the same way for the same reason. The
measurement says maintenance per page costs more than one rebuild; applying
that to fresh builds too would make every publish slower.

## Consequences

A resumed generation is briefly unindexed, between the drop and
`ensure_indexes`. This is safe because such a generation is always private and
never the active one — `resumable_generation` excludes the active collection
from its candidates — so nothing is serving reads from it. A run that dies in
that window leaves an unindexed generation, which the next resume adopts,
drops (a no-op) and rebuilds; no state is lost, because publication state
advances per page and is unaffected by index presence.

`drop_indexes` is abstract rather than defaulted to a no-op. A store with
index structures that silently skipped this would show the decay with nothing
in the code to point at, which is the failure mode that made #616 take a prod
reproduction to diagnose.

One convergence worth recording: #598 hypothesises that a resume may adopt a
*physically damaged* table, specifically from an attempt that died during an
index build. Dropping the indices on resume discards exactly that structure.
This is not evidence for that hypothesis, and #598 should still not be fixed
without it — but the mitigation is now shared.

## Evidence

Probe run 2026-10-02 against lancedb 0.34.0 on local disk, 45,000 rows over
29 pages, `ivf_pq` excluded because it needs more rows than the probe writes.
Shape reproduced, magnitude not: prod is 3.6M rows on GCS with `ivf_pq`
present, which the probe's omission makes an under-estimate rather than an
over-estimate. Prod numbers are #616's, measured 2026-10-01.

Read at `6fbf50b`: the page loop at `execution/search.py:523` with
`ensure_indexes` at `:817`; `resumed` set only where `resumable_generation`
returns a private generation; `upsert` acknowledging with
`count_rows(id_field IN (...))` while `append` counts rows without a
predicate.
