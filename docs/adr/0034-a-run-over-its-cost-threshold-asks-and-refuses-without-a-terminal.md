# ADR-0034: a run over its cost threshold asks at a terminal, and refuses without one

- **Status:** accepted
- **Date:** 2026-10-15
- **Prompted by:** #586

## Context

The reprocess guard ([ADR-0008](0008-reprocess-guard-defaults-to-fail.md))
stops an unannounced reprocess, and it counts rows. A row is not a price: a
thousand llm rows can cost more than a million embed rows. The guard is also
out of the way for exactly the runs that cost most. `on_code_change:
reprocess` turns it off for a model, and `--accept-reprocess` and
`--full-refresh` release it for a run. Once an operator has said yes to a
reprocess, nothing says what it will cost. ADR-0008 left cost as the third
shape of threshold because the plan had no dollars. #665
([ADR-0033](0033-an-embedding-is-cached-locally-by-text-and-a-plan-is-priced-from-history.md))
gave the plan dollars, priced from run-log history.

## Decision

**A target may set `confirm_cost_above_usd`.** Before the first model runs,
`run` and `build` then plan the selection, price it the way `stel plan`
does, and sum it. If the total is over the threshold, a run with a terminal
on stdin asks `Proceed? [y/N]`. Any other run (piped, scheduled, `--json`,
or called as a library) refuses. `--accept-cost` is yes in advance. The
threshold requires an enabled run log and is rejected at profile resolution
without one. `--full-refresh` and `--accept-reprocess` do not release this
guard: a full refresh is priced at every published row of each paid model.
A model with rows to reprocess and no priced history is named in a warning
and left out of the total.

## Alternatives considered

### A per-model threshold beside `reprocess_limit`

This is where the reprocess guard's knob lives, and it would let one costly
model be held to a lower bar. It lost because the question #586 asks is per
job. Five models at $9 each against a $20 bar is a $45 run that a per-model
threshold never questions. What is worth a question also differs by target,
not by model: $5 is a surprise on dev and routine on prod. A profile setting
is also operator-owned, like the run budget, rather than project YAML that
travels between targets.

### Warn only

Log the projection and carry on. This is the lightest change, and it never
stops a scheduled job. It lost for ADR-0008's reason: a warning nobody reads
does not protect anyone, and the runs this guards are the unattended ones.
The threshold is opt-in, so a target that sets one has asked to be stopped.

### Prompt everywhere, and wait

One behaviour instead of two. It lost because a scheduled or piped run has
nobody to answer the prompt. It would either hang until a timeout or read EOF
as an answer, and EOF must never mean yes. Refusing, with the projection and
the flag in the error, is the same question asked in the only form a
non-interactive run can show.

### Refuse when anything is unpriced

This fails closed: a model the guard cannot price might be the expensive one.
It lost because a target that never set `pricing:` would then refuse every
reprocess forever. That forces `pricing:` as a side effect of a threshold,
and the threshold has no way to say so. The warning names the unpriced
models, and the run budget's `max_cost_usd` is the cap that covers them.
A run log that cannot be read is not the same case and does refuse. Pricing
nothing would project $0 and start the run the threshold exists to stop
(Codex review on #695). `stel plan` still reports in that case, since it
spends nothing.

### Bound the run instead: lower `max_cost_usd`

The budget already stops a run that overspends. It lost because it stops
partway, after paying for the part that ran. The guard asks before the first
provider call. The two compose: the threshold asks whether to start, and the
budget caps what a started run can spend.

## Consequences

- **The projection covers only reprocessing.** It cannot see new input, a
  first build, or a `full` model's per-run rebuild, because a plan reads no
  source. A large new ingest passes under the threshold. The budget is the
  cap there, and the reference says so.
- **It is an estimate in both directions.** It is history's cost per
  provider row times rows, as in ADR-0033. A configured embedding cache under
  `--full-refresh` makes the run cheaper than projected, and a pricier model
  makes it dearer. An embed model under `--reprocess-all` is priced only at
  its stale rows, plus every row when an upstream model's code changed. Rows
  the flag alone reprocesses read their vectors back for free (#655), and
  this also changes `stel plan --reprocess-all`, which priced them at full
  rate before. Without this the guard would question the one recovery run
  that costs nothing.
- **A guarded run costs one run-log read** on top of the reprocess guard's
  plan, which is reused when that guard made one. A target with no threshold
  pays nothing extra.
- **A run inside the threshold logs its projection at `-v`.** It is not
  part of the default ledger, so "projected cost" is visible in `stel plan`
  always, in a run when it matters, and at `-v` otherwise.
- **Plan artifact schema moves to version 3**
  (`metadata.confirm_cost_above_usd`), so `stel plan --json` can tell an
  orchestrator whether a run would stop.
