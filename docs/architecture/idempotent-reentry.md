# Idempotent and re-enterable: a step that fails is resumed, not redone

## The contract

> For every step stel runs: **re-running it converges on the same result and
> never corrupts what is there** (idempotent), and **a step that failed resumes
> from where it stopped rather than from the beginning, while a step that
> succeeded is not redone because a later one failed** (re-enterable).

The unit of re-entry is the step's own checkpoint — the flush window for a
stage that publishes as it goes, the page or the generation for a search
publish, the whole step where the warehouse replaces a table in one atomic
operation and there is nothing partial to resume. It is not a new phase ledger;
[ADR-0005](../adr/0005-re-entry-unit-is-the-existing-checkpoint.md) records why.

This is stated as an invariant rather than left to each stage for the same
reason [bounded memory](bounded-memory.md) was: one root cause reached the
same corpus repeatedly, and each fix was correct and local.

| incident | step | what was lost |
|---|---|---|
| #492 | search publish | 3,613,979 rows written correctly over 4.2h, discarded after a six-second index failure; ~4.7h and a full warehouse read to recover |
| #491 | search index build | one transient error at the end of a publish that had been resilient for four hours |
| #495 | search publish | a full corpus rewrite to build an index over vectors that had not changed |
| #502 | search publish | one dead state scope per failed rebuild, ~2.1M rows after #492's incident |

Each was fixed. This page is what makes them add up to something.

## Definitions

**Idempotent** is the property a rerun needs: unchanged rows are skipped, a
republished row converges rather than duplicates, an index rebuilt over the
same rows is the same index. It is largely true by construction — incremental
state, keyed upserts, `create_index(replace=True)` — and until this audit it
was nowhere written down or checked as a whole.

**Re-enterable** is the property a *failure* needs, and it is the one #492
found missing. It has two halves. Forward: an interrupted step resumes from its
last checkpoint. Backward: work a step already finished is not redone because a
step after it failed. The second half is what "the step that failed was not
separable from the steps that had not" means, and it is the more expensive one
to lack — it is what turned a six-second failure into a 4.7-hour recovery.

**The checkpoint rules** that make both true for every stage that publishes in
windows live once, in `execution/checkpoint.py`, and are not restated per stage:

1. A full rebuild clears state before it writes anything. State that outlives
   its rows is the dangerous direction; a failure may leave state *behind* the
   target, never ahead of it.
2. State advances only after the write lands. An interrupted run never records
   a row it did not write.
3. A publication failure is reported without the warehouse's own words.

Rule 2 has a consequence worth stating because it looks like a violation and is
not: a run killed *between a write and its state* redoes that window. That is
correct — the window was never recorded — and it is the difference between a
safe redo and re-entry. Re-entry is measured across windows: what was recorded
stays done.

## The audit

`tests/test_reentry_contract.py` holds this table as data and is the gate on
it. Two things are scanned from the source rather than listed by hand, so a
new one has to be classified before it ships: every top-level `run_*_model`
entry point in `src/stel/execution/`, and every `timings.phase("...")` literal
— the phase vocabulary built for attributing wall clock (#486) is also the
vocabulary of where a publish can be interrupted. Every gate a row cites must
exist as a test, so a deleted gate fails there rather than silently unpinning
its step. The `GAP` list is pinned empty.

### Model kinds

| step | idempotent | re-enterable | unit | why |
|---|---|---|---|---|
| extraction | holds | holds | flush window | publish-then-state per flush; a full refresh replaces state after the write, so an interruption leaves state behind the target, never ahead |
| chunk | holds | holds | flush window | `FlushPublisher` (rules 1–3); deterministic chunk ids, so a rerun reproduces rather than duplicates |
| embed | holds | holds | flush window; the vector for an unchanged text | `FlushPublisher`, plus vector reuse for unchanged text and a resume that reads only what it needs; a budget stop is an ordinary interruption |
| llm | holds | holds | flush window; the provider batch | `FlushPublisher`; batch ids are persisted before the batch is awaited, so an interrupted batch is polled, not resubmitted |
| transform (SQL) | holds | **exception** | the whole step | one atomic replace, or a keyed merge; there is no partial state to re-enter and the warehouse does the work |
| transform (Python) | holds | holds | parent batch under the incremental contract | changed parents publish in batches, each batch's state advancing after its publish; without a contract, one atomic replace |
| ml | holds | **exception** | the whole step | full-only (#53); retrains each run, replaces relations atomically, publishes the artifact from a staged copy or discards it; no per-row checkpoint to resume, no provider spend to lose |
| eval | holds | **exception** | the whole step | pure warehouse arithmetic; full replaces, incremental deletes stale metric rows then merges; cheap enough that the step is the unit |
| search | holds | holds | the page (in place); the generation (private build) | keyed upsert with a durable receipt, state after the receipt; a private build that fails after its rows land is adopted by the next run (#492), its index step retried (#491), its rows seeded rather than re-read for an index-only change (#495), and its state scope swept with it when abandoned (#502); a resume whose upstream generation is unchanged skips the read and goes to the index build (#508) |

### Phases inside a publish

| phase | in | holds | why |
|---|---|---|---|
| `read` | search | both | a cursor-paged snapshot read; a retried cursor returns the identical page |
| `store_write` | search | both | `merge_insert` on the id; append only for a generation that is fresh, unseeded and unresumed |
| `state` | search | both | advances after the page's receipt, per page |
| `index_reconcile` | search | both | `create_index(replace=True)` is idempotent, `num_unindexed_rows` drives it, each index kind has its own bounded retry; a failure keeps the generation for the next run |
| `read` | embed | both | bounded upstream reads; a resume reads only the rows it still needs |
| `reuse` | embed | both | unchanged text reuses its stored vector |
| `provider` | embed | both | calls are per window and never for rows whose state advanced |
| `publish` | embed | both | `FlushPublisher` |

Activation is not a timing phase and is covered by the search row: it is
fenced, and a failure after the state swap clears that state and leaves the
previous generation serving ([ADR-0001](../adr/0001-degraded-serving-and-fail-closed-recovery.md)).

### The exceptions, and why they are the right unit

Three kinds hold re-entry only at whole-step granularity. None of them is a
gap, because in each the property the contract protects — expensive work is
not discarded — holds trivially: there is no expensive partial work.

- **SQL transforms** hand the whole computation to the warehouse. A full
  materialization is a single atomic statement (`CREATE OR REPLACE TABLE … AS
  SELECT` on DuckDB; a `WRITE_TRUNCATE` load job on BigQuery), so an
  interrupted run leaves the target exactly as it was.
- **ML models** are full-only and retrain every run. The primary and each
  secondary relation are replaced atomically and the artifact is staged, then
  published or discarded. A failure between two replaces leaves a mismatched
  pair until the rerun heals it — recorded here, not fixed, because training
  has no per-row checkpoint and costs no provider spend.
- **Eval models** are warehouse arithmetic over two relations already
  materialized. Cheap enough to run on every change.

### One recorded refusal to resume

A scope left `failed` by a stranded **in-place** publisher — the claim clears
the pointer, the process dies — is rebuilt from the warehouse on the next run
that changes configuration, retaining nothing. The rows it wrote are probably
fine (keyed upserts, per-page state), but the ledger has no proof the live
collection is whole, and copying it forward would launder any damage into a
generation that then activates as sound. This is ADR-0001's fail-closed
recovery applied to re-entry, and it is deliberate. An in-place publish
interrupted under an **unchanged** configuration resumes from state as usual.

## The open questions, answered

#493 left four. The audit answers them.

1. **The unit of re-entry** is the step's own checkpoint, and it already
   exists for every kind that has expensive partial work: the flush window
   (`FlushPublisher`), the page (search state), the generation (the serving
   ledger plus adoption by configuration fingerprint). The timing phases are
   the right *vocabulary* for where a failure lands — which is why each is
   classified — but they are not a second ledger.
2. **Where re-entry state lives** is where each kind already keeps progress:
   incremental state scopes for windowed kinds; the serving ledger and the
   generation's own stamp and state scope for search. Nothing new was added.
3. **"Activate an existing generation"** is neither a step nor an operator
   command. #498 made it automatic: the next run under the same configuration
   adopts the orphaned generation, resumes it, and activates it once it
   validates. A human choosing a generation to activate could choose one that
   never validated; the machine cannot.
4. **How much was already true**: all of it, once #490–#492 and #495/#502 had
   landed. The issue shrank to writing it down and gating it, as it said it
   might. The one cell nobody had tested — an ordinary in-place publish killed
   between pages — was tested for this page and holds.

## Adding a step

A new `run_*_model` or a new `timings.phase(...)` fails
`tests/test_reentry_contract.py` until it has a row. The row states both
verdicts, the unit of re-entry, the reason, and the tests that pin it. A step
that does not hold the contract is a `GAP` with its tracking issue in the
reason, and the pinned gap list grows only with one.
