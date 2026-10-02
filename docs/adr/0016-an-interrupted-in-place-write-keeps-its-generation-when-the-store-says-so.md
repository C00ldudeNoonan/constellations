# ADR-0016: A failed in-place publish keeps its generation when the store promises interrupted writes leave it sound

- **Status:** accepted
- **Amends:** [ADR-0001](0001-degraded-serving-and-fail-closed-recovery.md) — the
  in-place half of its asymmetry becomes conditional on a store capability;
  the private-build half and fail-closed recovery stand. Also
  [ADR-0003](0003-reader-safe-online-publication.md)'s "a failed in-place write
  cannot safely do so", for the same stores. In-place exclusivity is unchanged.
- **Date:** 2026-10-02
- **Prompted by:** #617 (the outage), #614/#615 (why no publish could end it)

## Context

ADR-0001 fixed #449 by retaining the live generation when a *private* build
fails, and deliberately left the *in-place* path fail-closed: an in-place
publish writes into the collection the activation pointer names, "so a failure
there may have corrupted what was live". The claim clears `active_generation`
up front so that `recover` fails closed on a crash too.

The common weekly path is in-place. An incremental publish with an unchanged
configuration writes into the live generation, and it is the publish that runs
most often and for longest on a large corpus. On 2026-09-13 one of those
failed at its index build (#598). The claim had cleared the pointer and the
failure left it cleared, so the ledger read `failed` with no generation, and
the reader gate refused every query. Four more publishes failed for unrelated
reasons (#592, a network drop, a lost lease, the six-hour read session of
#614), each re-clearing it. `sec_chunk_search` served nothing from 2026-09-27,
when a reader first asked, until this change — while the generation that had
served it on 2026-09-10 sat intact in the store at 3,644,778 rows, and `stel
serving status` printed `active_generation: -`.

The precaution guarded against damage that cannot happen on this store. Every
write `LanceDBStore` makes — each `merge_insert` slice, `add`, `delete`,
`create_index` — is one Lance transaction committing a new table version over
the previous, immutable one, and nothing in the store module compacts or
cleans up prior versions while publishing. A write interrupted anywhere leaves
the prior version readable and every row whole. The worst state an interrupted
in-place publish can leave is a collection in which some pages carry the new
`code_version` and the rest the old: *stale*, row-consistent, and described
exactly by the receipt-gated publication state the next run reconciles from.
That is the state every successful incremental publish passes through between
its pages.

## Decision

A store declares `RetrievalFeature.INTERRUPTION_SAFE_MUTATION` when an
in-place mutation interrupted at any point — error, or process death mid-write
or mid-index-build — leaves the collection readable with every row either
wholly as it was or wholly as the write intended. LanceDB and DuckDB declare
it.

On such a store, an in-place publish's claim keeps `active_generation` and its
configuration fingerprint; a clean failure hands both pointers back and the
scope becomes `degraded`, still serving; `recover` after a crash carries them
forward and serves on. A failed republish is a staleness event, not an outage.
On a store without the declaration, the in-place path behaves exactly as
ADR-0001 specified.

In-place publication still excludes readers for its duration (ADR-0003). With
the pointer no longer cleared, the row records that another way: an in-place
claim writes the status `publishing_in_place`, which is never servable, so
readers are refused with the same retryable "reconciling" error as before, and
a pinned reader still blocks the claim. `publishing` is now only the
private-build claim. `acquire_publish` therefore takes the two facts
separately — `preserves_active_generation` and `excludes_readers` — with the
old coupling as the default so a caller that says nothing still gets the
fail-closed in-place claim.

## Alternatives considered

### Keep clearing the pointer at claim time and restore it only on clean failure

Fixes the five clean failures in the incident without touching the claim.
Rejected because it leaves the crash path exactly where ADR-0001 put it: a
publisher killed mid-write never reaches `mark_failed`, and `recover` can only
carry forward what the claim left on the row. The 2026-09-27 failure was a
lease lost to a host crash. The invariant the issue asks for — a failed
publish never un-serves a ready generation — has to hold at claim time or it
does not hold.

### Preserve the pointer and let readers run alongside an in-place publish

The simplest row: no new status, and `publishing` with a pointer is already
servable. On this store it would even be safe — a reader opens one table
version and a concurrent commit creates another; no cleanup runs. Rejected
here because it reverses ADR-0003's "in-place writes still require exclusive
access" as a side effect of an availability fix, which is precisely the kind
of half-decided change ADR-0001's context warns about. It is a separate
decision with its own measurement (how long readers are refused during a
weekly in-place publish) and is left for one.

### Infer soundness from the receipt contract

`EXACT_MUTATION_RECEIPTS` and `ATOMIC_BATCH_MUTATION` already say a receipt
is never ahead of the store. Rejected because neither says what an interrupted
write leaves behind: a store can give exact receipts over writes that tear
rows or leave a collection unopenable mid-operation. The ledger asks one
question — "is what this publisher was writing into still servable if it
stops?" — and it should read one flag whose docstring answers it, declared by
the store that made the promise, rather than a conjunction a reader has to
reason about.

### Record the publish mode in a new ledger column

Would let the claim keep the pointer and still tell `recover` the mode
explicitly. Rejected because the status column already carries exactly one
value per claim kind once `publishing_in_place` exists, and a schema change to
a frozen table (`stel migrate` plans its rename) costs every deployment a
migration for information the row can already hold.

## Consequences

- **`publishing_in_place` is a fourth reader-visible status.** `publishing`
  narrows to private builds. Anything that pattern-matched on `publishing` to
  mean "a publisher holds the scope" now has two values to match.
- **A crashed in-place publish on an interruption-safe store recovers to
  `degraded` and serves.** `stel serving recover` output changes accordingly,
  and a new `serving:` line on both commands says in words what a reader gets.
- **The store's promise is load-bearing and unverified at runtime.** Nothing
  checks that a store declaring the feature behaves as declared; the
  declaration's comment in each store is the evidence, and adding
  `cleanup_old_versions` or a compaction step to the LanceDB publish path
  would silently falsify it. That is the sharp edge: a change to how the store
  writes has to re-read its capability declaration.
- **`degraded` now also follows an in-place failure**, so its `safe_error_code`
  is the only visible record that a weekly incremental publish is broken. The
  consumer-side alert on it is astrolabe's (astrolabe#802).
- **The row's `rows_*` counts after a degraded in-place failure describe the
  failed publish, not the served generation**, as they already did for a
  degraded private build.

## Evidence

- `src/stel/retrieval/lancedb.py` at this commit contains no call to
  `cleanup_old_versions`, `optimize` or `compact_files`; `upsert` issues one
  `merge_insert` per byte-bounded slice and `ensure_indexes` one
  `create_index` per index. Read 2026-10-02 against lancedb 0.34.0 (the
  pinned `>=0.34,<0.35`).
- The incident ledger row (#617): `fencing_token 18, status failed,
  active_generation NULL`, five failed publishes after `fencing_token 13,
  status ready, generation …g12fbc89e3823`, with the generation still in the
  store at 3,644,778 rows. First `capability_unavailable` 2026-09-27 17:57
  UTC; 61 refused queries by 2026-10-02.
- Both halves are mutation-checked in `tests/test_serving_coordination.py`:
  reverting the failure-path retention, reverting the claim's pointer
  preservation, removing LanceDB's declaration, admitting readers during an
  in-place claim, and letting a pinned reader no longer block the claim each
  fail exactly one named test (PR for #617).
