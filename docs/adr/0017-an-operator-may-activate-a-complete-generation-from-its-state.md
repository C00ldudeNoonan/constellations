# ADR-0017: An operator may activate a physically complete generation from its recorded state

- **Status:** accepted
- **Amends:** [ADR-0005](0005-re-entry-unit-is-the-existing-checkpoint.md) — its
  "no `stel serving activate` command" alternative is reversed for the case
  automatic resume cannot reach. Its decision on the unit of re-entry stands.
- **Date:** 2026-10-02
- **Prompted by:** #615, with #614 (why resume could not finish) and #617 (why
  nothing was serving meanwhile)

## Context

ADR-0005 declined an operator activate command because #498's automatic
adoption covered the incident that asked for it: the next run under the same
configuration fingerprint finds the orphaned generation, resumes it, and
activates it once it validates. "An operator command lets a human activate a
generation that never validated; the automatic path cannot."

That argument assumes the next run can finish. On the corpus in #615 it could
not. Resume re-reads every page from the warehouse until the page loop has
completed and stamped the generation (#510), and at the observed page rate the
read outlasts BigQuery's six-hour Storage Read session (#614). Each leg
re-paged 146 pages, wrote what it could, and died at the wall; convergence
needed four such legs, about 24 hours of mostly re-reading, each failure also
taking the index offline until #617. Meanwhile a generation holding all
3,644,778 rows — every row `stel plan` reports — sat in the store with its
indices half built, and no command could make it the served one.

The rows it needed to be activated were already described: every write into
it had advanced publication state, in the generation's own scope for the
resumed legs and in the serving scope for the publishes before the pointer was
lost. What stood between the generation and readers was bookkeeping — state at
two `code_version` hashes after a hash-only change (#607), split across two
scopes, and a pointer that `recover` could no longer carry — plus an index
refresh. None of it required reading the corpus again.

## Decision

`stel serving activate <model> --generation <collection> --rows-verified`
makes a named physical collection the served generation of a search model
from the publication state stel recorded for its rows, re-stamped at the
current `code_version`, without reading the corpus from the warehouse.

It refuses, before claiming the scope, unless the collection exists, carries
this model's configuration fingerprint, and holds exactly as many rows as the
upstream relation. Under its claim it assembles the generation's state — its
own scope's records win; the serving scope's fill in keys it never recorded —
re-stamps every record, and refuses unless that state describes exactly the
collection's row count and a sample of its keys drawn across the whole scope
is present in the collection. It then builds missing indices, validates the
collection the way a publish does, swaps the state into the serving scope, and
activates. A refusal after the claim is recorded as a failed publish that
retains whatever pointers were there, so it never un-serves a generation
(#617).

The command requires an explicit target, as `recover` does, and
`--rows-verified`: the operator's assertion that the rows are what the current
code would publish, which is what makes the re-stamp correct and which nothing
in the command can establish.

## Alternatives considered

### Keep resume as the only path and fix the wall

#614's fixes — a read session reopened per page against a pinned snapshot, or
page-level completeness so a resumed leg skips reads it already did — make the
publish finish. They do not make it short: the second leg alone was 73 write
pages at seven minutes each. They are the right fix for the publish, and are
filed; they are not a recovery tool for a generation that is already complete.
Rejected as the *only* path because the operator's need here is to stop
re-reading, not to re-read successfully.

### Re-stamp `code_version` in place and let the next run adopt the generation

#615's second option: an operator re-stamps the state at the new hash and the
ordinary resume finds nothing changed. Rejected because adoption still re-pages
until the generation is stamped complete (#510), and this generation never
was: its source-generation stamp was cleared when a resumed leg began
rewriting it. The re-stamp is half the work and is folded into the command; on
its own it would have left the operator exactly where they started.

### Derive the state from the store's rows instead of trusting the state tables

The store holds the projected row values, and `input_fingerprint` is a
function of them, so the state could be recomputed by scanning the collection.
Strongest in principle: the state would be exactly what the store holds.
Rejected for this change because it reads the whole collection — the vector
column alone is ~11 GB on this corpus (#461) — and depends on float values
round-tripping through the store byte-identically, which is a second claim to
prove. The state tables are stel's own receipts, written only after a store
acknowledgement, and the resume path already trusts them for the same rows
(#492: "the rows are not trusted — the generation's own publication state is").
The membership sample is what covers the one way those receipts can mislead:
describing a different collection than the one named.

### Let the command activate without `--rows-verified`

The refusals cover everything the command can check. What they cannot check is
whether a row the state vouches for is what the *current* code would produce
for it: after a hash-only change it is; after a change to embedding or
chunking it is not, and the next incremental run would have rewritten it. An
operator knows which release they are recovering from; the command does not.
The flag makes that the operator's statement rather than the command's guess,
in the shape `recover --owner-terminated` already uses for the same reason.

## Consequences

- **ADR-0005's footgun is now a flag.** A human can activate a generation
  whose rows the current code would not have produced, by asserting otherwise.
  The next incremental run reconciles against the upstream regardless, so the
  cost of a wrong assertion is one cycle of stale rows, not a divergence; the
  reference says so, and says when the assertion is true.
- **`count_present` and `iter_record_ids` join the store contract.** Every
  store must answer a bounded membership probe and stream its ids in bounded
  pages. Both stores do the first with the predicate their upsert already
  acknowledges on, and the second with the projected scan seeding uses.
- **The command re-stamps state in the generation's scope before it knows the
  activation will succeed.** An interrupted activation leaves that scope at the
  current hash with the serving scope's records copied in, which is exactly the
  state a second attempt (or an ordinary resume) wants. It is idempotent by
  construction.
- **Re-activating the served collection excludes readers for the index
  build**, as an in-place publish does (ADR-0003); activating any other
  collection leaves them on theirs.
- **The row-count check compares against the upstream as it is now**, not as
  it was when the generation was written. A corpus that has grown since
  refuses activation and sends the operator to resume — correctly, since the
  generation is no longer complete for anything that exists.

## Evidence

- #615, 2026-10-02: generation `…__g12fbc89e3823` at 3,644,778 rows, eight
  indices each `indexed=1,805,803 / unindexed=1,838,975`, serving ledger
  `status=failed, active_generation=NULL`; the resumed leg's state at the new
  `code_version` for 1,825,000 rows, the serving scope's at the old for the
  rest.
- #614, 2026-10-01: last successful page at 6h00m36s, failure at 6h07m41s;
  per-page cost 3.72 → 7.48 min across 73 pages.
- `tests/test_serving_activate.py` reproduces the shape — complete private
  generation, lost pointer, state split across two scopes at a pre-release
  hash — and shows the next incremental run writing zero rows afterwards. Each
  refusal has its own test; a late refusal is shown to leave a served
  generation `degraded` and answering.

## Addendum, 2026-10-02: a shortfall is tolerated

Recorded as an addendum rather than an edit, because it changes the decision
above, not only its wording. "Holds exactly as many rows as the upstream" and
"describes exactly the collection's row count" were written as the
completeness checks a publish makes, and they would have refused the very
activation this record exists for. The upstream relation grows with every
embedding run, so a generation is "exactly complete" only for the hours
between its last write and the next filing; and a page whose slices committed
before its state advanced (the 2026-09-20 failure died inside one) leaves rows
the state does not describe. Both are staleness the next incremental run
reconciles — it publishes the rows the collection lacks and re-upserts, by
idempotent merge, the rows its state does not know — and neither is damage.

Both checks now refuse only the direction that is: a collection holding rows
the upstream does not, and state naming rows the collection does not hold.
A shortfall in either is activated and reported on a `pending:` line, so the
operator knows by how much the served index is behind. The consequence above
about a grown corpus refusing activation no longer holds; it is reported
instead.

The state shortfall needed one more step, found in review of #630 after the
first version of this addendum merged. "The next run re-upserts the rows its
state does not know" is true only while their upstream keys exist: stale
discovery enumerates *state* keys absent upstream, so a row with no state
whose key is deleted before the next run is invisible to the sweep and would
be served through every later run. Refusing the shortfall was one answer and
would have sent the #615 operator back to the corpus read; the other, taken
here, is for activation to walk the collection's ids and record every row the
state does not describe under a marker fingerprint no upstream row can match.
The next run then re-upserts the row if its key still exists and deletes it as
stale if not, so the marker lives for exactly one run. `iter_record_ids` joins
the store contract for it; the walk is ids only, in bounded pages, so it costs
a few megabytes a page against the 3.6M-row collection rather than the
vectors beside them. The upstream figure on the `pending:` line is a net
difference — a deletion and an insertion cancel in it — and is now worded as
one.
