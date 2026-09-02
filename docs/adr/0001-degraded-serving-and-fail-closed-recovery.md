# ADR-0001: A failed publish keeps serving its live generation; recovery fails closed

- **Status:** accepted
- **Date:** 2026-09-02
- **Prompted by:** #449 (the outage), #450 (the change)

## Context

Publishing a search index writes a new generation and activates it atomically
(#355). A publication that failed cleared the ledger's `active_generation`
unconditionally, and query admission requires a named generation — so **any**
failed publish, and any `stel serving recover` after a crashed one, took a
previously-healthy index out of service until the next successful publish. On
a large corpus that is hours, and it lands exactly when a publish is most
likely to be interrupted: a long one.

The reasoning for the *right* behaviour was already half-written. #355 gave
`active_collection` an explicit asymmetry, with a docstring explaining it: an
in-place publish writes into the collection the pointer names, so a failure
there may have corrupted what was live and the pointer must go; a private
generation build writes where nothing is reading, so its failure leaves the
previous generation correct and the pointer is kept.

`active_generation` was never given the same treatment. Nothing recorded why
it was different, because nothing had decided it was — and the field that
governs whether queries are admitted at all was left clearing on every
failure. That is the specific way this decision resurfaced, and the reason
this ADR exists: half a symmetric decision was written down, and the unwritten
half drifted into an availability bug.

## Decision

Both activation pointers carry the same asymmetry, and the publication claim
records which kind of publish is running so a crash can be told apart from a
clean failure.

A private generation build's failure retains `active_generation`,
`active_collection` and the configuration fingerprint that generation was
published under; the scope becomes `degraded` and keeps answering queries from
it. An in-place publish clears all of them and becomes `failed`, admitting
nothing. `acquire_publish` clears `active_generation` at claim time for an
in-place publish, so `stel serving recover` — which cannot ask a dead process
what it was doing — fails closed on anything it cannot prove was a generation
build.

`degraded` always retains its `safe_error_code`, and heals on the next
successful publish with no operator action.

## Alternatives considered

### Leave the scope `failed` and let the operator republish

The status quo, and it is safe. It is also the outage: the data is intact and
the ledger still names it, so refusing every read is a choice to be
unavailable rather than a consequence of damage. "Yesterday's index" beats
"no index" for a search endpoint, and the operator can already see the failure
in `stel serving status`.

### Serve from any scope that still names a generation

Simpler — one rule, no publish-mode tracking. Rejected because it cannot
distinguish the two failures. A crashed *in-place* publisher leaves the prior
generation named in its row while having possibly half-rewritten the
collection that pointer resolves to, so this rule would serve a corrupted
index. Recovery has no way to ask a process that is gone.

### Have recovery infer the publish mode from ledger state

Attractive because it needs no change to the claim. Rejected: nothing in the
row distinguishes an in-place publisher from a rebuild after the fact, and
inferring it from what *happens* to be present is exactly the guess that
serves corrupt data. The claim knows; a crash destroys that knowledge unless
the claim writes it down first.

### Timeout-based lease stealing, so recovery is not needed

Out of scope here and still rejected on its own terms: an operator asserting
that the old owner is terminated is a stronger guarantee than a clock. This
ADR does not weaken it — serving reads from a generation nobody is writing is
not the thing lease fencing exists to prevent, and recovery still grants
nobody publication authority.

## Consequences

- **`degraded` is a third reader-visible state.** Anything that pattern-matched
  on `ready` versus `failed` now has a case it has not seen. `SERVABLE_STATUSES`
  is the single place that decides, and all three reader gates consult it.
- **A retained generation carries its own configuration fingerprint**, restored
  over the claiming publish's. Retaining the pointers without it would
  advertise the old index as answering under a configuration it was never
  built for; `mark_failed` refuses that combination outright rather than
  writing an incoherent row.
- **A rebuild forced by a configuration change still stops serving** — correctly.
  The retained generation cannot answer queries embedded under the new
  configuration, so admission refuses it early with an accurate message. The
  availability win is real only for `--full-refresh` and crash cases.
- **Clearing the pointer at claim time costs nothing** and is easy to read as a
  bug. Query admission already refuses a scope with a publisher on it, so the
  generation is unservable from the moment the claim lands either way.
- **Recovery reads pointers from the highest-fence row only.** Searching for
  any row still carrying a generation would let the duplicate-row repair path
  resurrect one that a later in-place publish has since rewritten.

## Evidence

- The reader gate requires both conditions — `retrieval/coordination.py`,
  `acquire_query`: `if row[1] not in SERVABLE_STATUSES or row[5] is None`.
  `row[5]` is `active_generation`, which is why clearing it alone was
  sufficient to take the index offline.
- The pre-fix recovery INSERT named `active_collection` but not
  `active_generation`, so the latter landed NULL by omission rather than by a
  decision.
- Both halves are mutation-checked: reverting the `mark_failed` carry, and
  reverting the recovery carry, each fail their tests with a `failed` versus
  `degraded` diff (PR #450).
