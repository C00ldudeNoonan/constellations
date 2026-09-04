# ADR-0005: The unit of re-entry is each step's existing checkpoint, not a phase ledger

- **Status:** accepted
- **Date:** 2026-09-04
- **Prompted by:** #493

## Context

#492 stated the problem: a 4.2-hour search publish wrote every row correctly
and failed six seconds into its index step, and nothing could re-enter it at
the step that failed, so the next run re-read the corpus. #493 named the
invariant that incident violated — every step idempotent and re-enterable —
and left four design questions open, the first two of which were about
*mechanism*: what is the unit of re-entry, and where does re-entry state live?
It floated the timing phases from #486 as the candidate unit and an extension
of the serving ledger as the candidate home.

By the time the audit ran, #490, #491, #492 (as #498), #495 and #502 had all
landed. The audit's job became establishing what was true, and the answer
decided the mechanism question by making it moot.

## Decision

The unit of re-entry is the checkpoint each step already keeps, and re-entry
state lives where that checkpoint already lives. No phase-level ledger is
added, no cross-kind checkpoint framework, and no `stel serving activate`
command.

Concretely: the flush window for stages that publish as they go, whose rules
live once in `execution/checkpoint.py`; the page for an in-place search
publish, through the same incremental state; the generation for a private
search build, through the serving ledger plus adoption by configuration
fingerprint (#498). The timing phases are the *vocabulary* the audit classifies
— each is a place a failure can land — and are pinned as such by
`tests/test_reentry_contract.py`, but they are not made into a second source
of truth.

## Alternatives considered

### A first-class phase ledger

The issue's own candidate: make `read` / `store_write` / `state` /
`index_reconcile` first-class, record which one a publish reached, and resume
from it. It lost because every failure point the audit could name was already
covered by an existing checkpoint at the right granularity — page state
covers a failure in `store_write` or `state`, generation adoption covers a
failure in `index_reconcile` or activation — and a second ledger recording the
same facts is a second thing that can disagree. The phases were built to
attribute wall clock; borrowing them as a resume protocol would have
coupled two unrelated concerns through one string.

### A single checkpoint framework across all kinds

Tempting as uniformity. It lost on the audit's shape: the four windowed kinds
already share one (`FlushPublisher`), and the three whole-step kinds — SQL
transforms, ML, eval — have nothing to checkpoint, because the warehouse
replaces their table in one atomic operation and there is no expensive partial
work to lose. A framework for them would be scaffolding around an empty room.
They are recorded as whole-step exceptions with that reason.

### `stel serving activate <generation>` as an operator command

#492 asked for it, for the incident it describes. It lost to #498's automatic
adoption: the next run under the same configuration fingerprint finds the
orphaned generation, resumes it, and activates it only once it validates. An
operator command lets a human activate a generation that never validated; the
automatic path cannot. The incident's actual need — do not throw away 4.2
hours of correct rows — is met without giving anyone a footgun.

## Consequences

- **A new kind or a new phase must be classified before it ships.** The gate
  scans `run_*_model` entry points and `timings.phase(...)` literals; an
  unclassified one fails the suite. This is the same teeth #414 gave bounded
  memory, and it is what stops the next #492 from being discovered in
  production.
- **The gap list is pinned empty.** The audit found no step that breaks the
  contract. Growing the list needs an issue and a row saying why.
- **"Correct redo" and "re-entry" are different, and the tests must know
  which they are exercising.** A run killed between a write and its state
  redoes that window, correctly, under rule 2. The first draft of the in-place
  re-entry test injected its failure inside the first upsert, saw both pages
  rewritten, and briefly read that as a product gap. It was the test. The
  injection belongs *after* a window's write and state have both landed.
- **The one refusal to resume is deliberate.** A scope left `failed` by a
  stranded in-place publisher rebuilds from the warehouse rather than
  adopting the collection it was mutating (ADR-0001, fail closed). Anyone
  tempted to "optimise" that path should read the reason first.

## Evidence

- The audit table in `tests/test_reentry_contract.py` and
  `docs/architecture/idempotent-reentry.md`, 2026-09-04: nine entry points and
  eight phases classified, every verdict pinned to a named test, gap list empty.
- `test_an_interrupted_in_place_publish_resumes_from_state`, added for the
  audit: an in-place publish killed on its second page retries with
  `rows_written == 1`. With the injection moved inside the first upsert the
  retry writes 2, which is rule 2 working, not re-entry failing.
- #492's measurements: 3,613,979 rows, 145 pages, 4.2h, index step failed at
  six seconds; all seven indices later built by hand on the same generation in
  173s. #498 made that generation resumable; this ADR records why nothing
  further was built on top.
