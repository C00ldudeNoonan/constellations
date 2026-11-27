# ADR-0068: Variant selection is an offline library over supplied evidence, and adopting a live result is deferred

- **Status:** accepted
- **Date:** 2026-11-27
- **Prompted by:** #795 (ALE-286), milestone "Research applications"

## Context

ALE-286 asks for the cheapest configuration that meets an explicit quality
floor, chosen from measured quality, provider cost and latency, then validated
on a holdout. `retrieval_compare.py` already lines retrieval variants up with
paired-bootstrap deltas, the run budget (`budget.py`) already enforces spend
caps and now settles billed failures, and `execution/cost.py` already prices
usage. What did not exist was a way to combine quality, three cost phases and
latency into a decision. The reviewed benchmark and holdout that ALE-281
promises, and the section-extraction strategy of ALE-282, are not on master, so
there is nothing reviewed to run a live experiment against.

## Decision

`stel.variant_selection` is a pure library. It consumes per-document counts and
per-phase measurements the caller produced, and returns the observed frontier,
quality-floor feasibility on bootstrap **lower bounds**, the cheapest feasible
selection, a one-shot holdout check of that frozen selection, and an
adopt/defer recommendation with a config diff for review. It makes no provider
call and writes only its own artifact. **Adopt/defer for the live question
(which two models x which two strategies) is deferred**: no reviewed benchmark
exists, so no frontier is reported here, and the library will not produce an
`adopt` below a caller-set document count.

## Alternatives considered

### A new `stel select` command that runs the candidates

Looked right because it makes the experiment one step. It was ruled out for
now: it would need the ALE-281 judgments and the ALE-282 strategy to exist,
would put live provider spend behind a command, and AGENTS.md wants any
external spend to be an explicit opt-in. The library is the part that is
true regardless; a command can wrap it once a benchmark exists.

### A second reservation ledger for experiment spend

The earlier prototype carried a SQLite ledger. Ruled out: `BudgetGuard`
already reserves calls and caps spend, and a second ledger would be a second
contract. The one gap it exposed, a billed failure not charging the budget, is
fixed in `llm_backend.py` instead.

### Ranking by point estimates, or picking the "best" on the frontier

A point-estimate frontier always names a winner, including between two
candidates whose intervals overlap. Ruled out: the result reports ties and
selects only the cheapest candidate whose lower bound clears the floor, so a
tied configuration is never presented as proven better.

### Autonomous plan search and cross-provider routing

Out of scope for ALE-286; the candidate set is given, not searched.

## Consequences

The library cannot verify that judgments were reviewed, documents are
independent, or the holdout was left untouched; those remain the caller's
discipline and are stated in the docs. Pricing is an assumption string plus a
measured/estimated flag, so a stale rate table gives a wrong estimate rather
than an error. Revisit when ALE-281's benchmark lands: run the live experiment,
record the frontier, and supersede this ADR's deferral.
