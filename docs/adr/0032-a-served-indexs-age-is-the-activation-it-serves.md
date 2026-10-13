# ADR-0032: A served index's age is the activation it serves, read from the ledger

- **Status:** accepted
- **Date:** 2026-10-13
- **Prompted by:** #682

## Context

`list_context_models` is how an agent decides whether to trust what it is about
to read. It answered the "how current is this?" question twice, and both
answers were wrong against a live index.

`ContextResource.summary()` hard-coded `freshness="unknown"`. And
`last_successful_materialization` came from `_successful_materializations`,
which read the single `generated_at` out of the latest `target/run_results.json`
and attributed it to every model that run reported success for.

On 2026-10-12, against astrolabe prod, `sec_chunk_search` reported
`"freshness": "unknown"` and `last_successful_materialization: null` while
serving chunks recorded through 2026-10-09. The index was active and answering.

The artifact answer goes null in three ordinary situations: a later run whose
selection leaves the search index out, a run where the index was not rebuilt,
and a container start that rehydrates the store from object storage without a
run at all. The third is the normal startup path for the deployment that
reported this.

The consequence that matters is not the missing field. It is that an index
which has *stopped* publishing — queries still served, from a generation
getting older, because every publish since has failed (#617) — looked exactly
like one that had just finished. An agent had no way to tell them apart.

## Decision

An index's age is the activation of the generation it is serving, read at
request time from the serving ledger.

1. The ledger gains `activated_epoch`, written **only** by `mark_ready`. A
   failed publish that retains its predecessor leaves it alone, and
   administrative recovery carries it forward with the pointers it retains.
2. `list_context_models` reads it per listing through a new
   `ContextRepository.read_owned_rows`, as the operator, keyed on the
   `(model_name, retrieval_publish, scope_fingerprint)` scope the manifest
   already records.
3. `freshness` is derived from that stamp against an optional model-level
   `search: freshness: stale_after:`, reusing the existing `FreshnessStatus`
   vocabulary: `pipeline_stale` past the policy, `fresh` within it or with no
   policy declared, and `unknown` only when there is no activation on record.
4. The catalog stops reading `run_results.json` entirely.

`activated_epoch` is deliberately not `completed_at`, which already existed on
the row. `completed_at` moves on every finished publication attempt, failures
included, so it dates the last time a publisher ran rather than the rows being
served — which is the conflation this ADR exists to undo.

**A date is about a generation, so an index with no active generation has no
age.** A fail-closed in-place failure clears both pointers, and `mark_failed`
clears the stamp with them; recovery with nothing servable writes none. The
catalog enforces the same invariant on read rather than trusting the writers,
because one missed path reports a successful materialization — possibly
`fresh` — for a scope that answers nothing, which is a worse answer than
`unknown`.

The publish claim is the one place the row may hold a stamp whose generation is
null, and it does so deliberately. `acquire_publish` nulls `active_generation`
unless the claim preserves it, but an in-place failure on an interruption-safe
store hands both pointers back and goes `degraded` (#449, #617) — so the
generation it nulled can be live again minutes later, and clearing its date
would mean a retained generation serving with no age at all. The read-side
invariant is what makes leaving it safe.

The policy is reporting only. `freshness` joins `NON_SEMANTIC_FIELDS`, so
tightening a threshold does not re-key or republish a row; on the 3.6M-row
index this was measured against, the alternative would make a one-word config
edit cost a full republish.

## Consequences

How current an index is becomes live warehouse state rather than artifact
state, which is why it survives a selection that skips the index and a store
rehydrated without a run.

It also costs a warehouse read per listing. One read covers the whole page, the
ledger holds one row per scope, and the read is on a discovery call rather than
a query path — but `list_context_models` is no longer answerable from local
files alone, and a deployment whose MCP process cannot read stel's own schema
now reports `unknown` where it previously reported a number. It reports it
quietly: a ledger that is merely older than the column is an expected state
during an upgrade, not a fault, so the column-presence check distinguishes it
from a read that genuinely failed.

`read_owned_rows` is a second read path on the repository, which is a seam
worth naming rather than widening: `read_rows` filters through
`adapter.list_tables()`, which hides every internal table by design, so it can
only ever report the ledger as absent. The new method takes no `identity`,
because there is no choice to make — a caller holds no grant on stel's
infrastructure.

`freshness` now has two meanings in the same response, which is a real cost in
clarity. Per-result `freshness.status` still describes the row's own declared
contract fields; the model-level one describes stel's publication. Reusing
`pipeline_stale` for the latter is what keeps them reconcilable: both mean "the
refresh is overdue", at their own grain.

Nothing refuses a query on this basis. A stale index is still served, and
saying so is left to the caller — making staleness a refusal would take a
working index offline on a reporting threshold, which is the #449/#617 failure
mode in reverse.

## Alternatives considered

### Keep reading `run_results.json`, and fix the attribution

Record per-model completion times rather than one `generated_at`, and merge
across runs so a later run that skips the index does not erase it. Rejected: it
cannot survive the case that prompted the issue. A container that rehydrates
the store from GCS and serves queries has no run results at all, and the thing
being dated is a generation in a warehouse, not a local file. Merging history
into an artifact to approximate state the warehouse already holds exactly is
the wrong direction.

### Derive it from `completed_at`, already on the row

No migration, no new column. Rejected on correctness: `completed_at` moves when
a publish *fails*, so a scope that has been failing for days would report
itself freshly materialized on every failure. That is precisely the signal #682
was filed to recover, and this alternative is the bug wearing the fix's clothes.

### Read the ledger once at server start

Cheaper — no per-listing read. Rejected: a long-lived `stel mcp serve` process
is exactly where this matters, and a value frozen at boot would report the age
the index had when the server started, forever. The failure mode it has to
catch unfolds over days of a running server.

### Declare the policy as `stale_after` on the row contract

`agent_context/v1` already carries a per-row `stale_after`, and #682 floated
reusing it. Rejected: that field describes one document's own validity, set by
whatever produced the row, while this describes the cadence of stel's
publication of the whole index. Overloading it would mean a model-wide
publication policy inherited from, or conflicting with, per-row document
claims — and the row-level field is projected by user transforms, so stel
cannot assume it is populated at all.

### A `warn_after`/`error_after` pair, matching source freshness

`FreshnessConfig` for sources already has that shape, so the vocabulary would
be familiar. Rejected: the model summary carries a single `freshness` string,
and two thresholds need a third status to be expressible in it. One knob that
maps cleanly onto the existing vocabulary beat two that would have forced a new
one.
