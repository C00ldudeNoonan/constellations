# ADR-0013: A merge page is bounded by bytes, in the store, before it is sent

- **Status:** accepted
- **Date:** 2026-09-20
- **Prompted by:** #592

## Context

`search.batch_size` counts rows. It is a good lever for what it was tuned
for: each published page costs two ledger reads and an `upsert_state` MERGE —
about 3.7s of BigQuery round trips — so page *count*, not row count, drives
publish cost, and it was deliberately raised from 2,000 to 25,000 to cut that.

`LanceDBStore.upsert` turned one page into one Arrow table and one
`merge_insert`. `merge_insert` reserves the whole payload for its join build
side out of a process-wide pool, fixed at 100 MB in lancedb 0.34.0 and not
reachable from configuration: `lancedb.Session` exposes
`index_cache_size_bytes` and `metadata_cache_size_bytes` and nothing else.

So a lever denominated in rows governed a limit denominated in bytes, and row
size varies with text length. Measured against the live 3.6M-row collection:

| rows | payload | result |
| --- | --- | --- |
| 20,000 | 91.8 MB | ok |
| 25,000 | 115.0 MB | Resources exhausted, needed 111.7 MB of a 100 MB pool |

Two consecutive weekly publishes died on it — one after writing for 4h55m,
one on page 1 having committed nothing.

## Decision

**The store splits the payload, and `batch_size` keeps its meaning.**
`LanceDBStore.upsert` slices the Arrow payload into pieces of at most
`MERGE_PAYLOAD_LIMIT_BYTES` and issues one `merge_insert` per slice.

**The slice size is measured, not averaged.** A row count derived from the
mean row size can still overshoot, which is the same assumption that caused
the failure. `Table.slice` is zero-copy and its `nbytes` reports the slice
rather than the parent's buffers, so each candidate slice is measured and
shrunk until it fits.

**The ceiling is well under the pool, not just beneath it** — 64 MB against
100 MB. The FTS and BTree index builds allocate from the same pool, and a page
that fails costs the entire publish.

**A single row above the ceiling is sent alone rather than refused.** The pool
is larger than our ceiling, so such a row may well succeed.

## Alternatives considered

### Make `batch_size` a byte budget

The issue's first suggestion, and wrong for a reason that only shows up from
the call site. `search.batch_size` is passed to four different things: the
upstream `table_snapshot`, the `BoundedReconciler`, `_activate_generation`,
and the store write. Three of those are BigQuery state paging, where rows are
the correct unit and bytes would be meaningless. Redefining the field would
change all four to fix one.

It also puts a LanceDB-specific constant in user-facing configuration. The
pool is a property of one store at one version; an operator on DuckDB-backed
retrieval would be tuning against a limit that does not exist for them.

### Keep one merge per page and bisect on failure

Catch `Resources exhausted`, halve, retry. Attractive because it self-heals
without knowing the pool size, and the issue suggests it.

Rejected as the primary mechanism because it pays the failure before it
learns: the 2026-09-13 run wrote for 4h55m and then hit this, and a retry
loop that starts at the failing size still has to fail once per page. It also
depends on matching native error text, which `_operation_failed` deliberately
discards — so the store would need the very detail #490 removed, re-obtained
from a string match.

Worth adding *later* as a backstop if a payload under 64 MB is ever refused;
it is not what should be carrying the common case.

### A configurable cap

Rejected for now. The failure was a knob calibrated in the wrong unit; the fix
should not be another knob calibrated in a unit the operator cannot observe.
Nothing about a deployment changes the pool, because the pool is not
configurable. If a future LanceDB makes it settable, the cap should be derived
from that value rather than typed in by hand.

## Consequences

**LanceDB stops claiming `ATOMIC_BATCH_MUTATION` and claims
`EXACT_MUTATION_RECEIPTS` instead.** A split page is several transactions, so
the atomicity claim would be false for exactly the pages that need splitting.

`docs/architecture/semantic-retrieval.md` has always specified two proofs for
a trustworthy receipt — "exact per-ID durable outcomes *or* prove
`ATOMIC_BATCH_MUTATION` and return an all-success atomic receipt" — but the
`RetrievalFeature` enum carried only the second, and the compiler required it
outright. So the spec's "or" existed on paper and not in code. It does now:
`EXACT_MUTATION_RECEIPTS` is a real feature, and preflight accepts either.

`upsert` earns the one it claims. It confirms every id it was handed is
durably present before returning, and raises otherwise, so a returned receipt
is never ahead of the store — which is the only property the publish loop
actually gates state on. `MutationReceipt.atomic` is documented to mean
exactly that: the receipt is complete and trustworthy, not that the backend
ran one transaction.

DuckDB is untouched and still claims atomicity: its batch really is one
transaction. The withdrawal is specific to the store that has to split.

A page is no longer one Lance transaction. That is safe under a contract the
publish loop already relied on: the store write happens, and only then does
`upsert_state` advance state for that page. A slice that fails leaves the
page's state unadvanced, so the next run republishes the whole page, and
`merge_insert` keyed on the id absorbs the rows that already landed. This is
the same reasoning that already made a whole-page retry correct; splitting
changes how much is replayed, not whether replay is safe.

More `merge_insert` calls per page against object storage. The per-page cost
the 25,000 tuning was buying back is the BigQuery round trip, which is
unchanged — pages, and therefore ledger reads and state MERGEs, are exactly as
before.

`append` is untouched. It uses `table.add`, which has no join build side and
does not draw on this pool; the 2026-09-13 run appended 3.6M rows through it
without trouble. If the index build turns out to exhaust the same pool, that
is a separate path and a separate decision.

The constant carries the measurement it came from. A number like 64 MB is
otherwise indistinguishable from a guess, and the next person to raise it
should have to argue with the table above.

## Evidence

Read at `b73a214`: `LanceDBStore.upsert` built one `pa.Table` per page and
called `merge_insert(...).execute(payload)` once; `search.batch_size` reached
`table_snapshot`, `BoundedReconciler`, `_activate_generation` and the store
write; `execution/search.py` advances `upsert_state` only after the receipt is
checked. Measurements are from #592, taken against the live collection
(stel 0.18.0, lancedb 0.34.0, GCS-backed, 3,644,778 rows).
