# 0024. A forced reprocess ignores incremental state rather than clearing it

Status: accepted
Date: 2026-10-07
Prompted by: issue #655 (ask 3 of #653); amends [0022](0022-a-schema-probe-is-never-cached-and-a-missed-column-fails-the-run.md)

## Context

ADR-0022 made a dropped upstream column fail the run instead of publishing
without it. It left the other half of #653 open: getting out of a target that
was already poisoned, which at the time cost either `--full-refresh` — 3.6M
rows and about 28k Vertex requests on astrolabe's corpus — or editing
`stel_state` and `stel_sync_watermark` by hand.

What is wanted is narrow: reprocess every published row, and reuse the vectors
already paid for. Two tiers of skipping stand in the way. The inner one is
per-record state, `StateValue(input_fingerprint, code_version)`, consulted by
all five state-reading stages. The outer one is the sync watermark of #611 and
ADR-0015, which can skip a stage's whole parent scan before any record is
considered.

`--full-refresh` clears both, and four other things with them: in
`run_embed_model` a single flag drives `is_incremental`, `rebuild_target`,
`processed_state`, `reuse_reader` and `use_full` at once. Dropping
`reuse_reader` is what makes it expensive, and it is not separable from the
rest by a caller.

## Decision

A new `--reprocess-all` on `run`, `build` and `plan`. It **reads incremental
state as usual and refuses to skip on it**, rather than clearing it.

One helper, `state_for_skipping(state, reprocess_all=...)`, returns the view a
stage may skip against — empty under the flag. Each stage keeps the fetched
state under its own name for removal reconciliation, so the two questions
state answers ("may I skip this record?" and "what did the last run publish?")
are separate at every call site. The outer tier is declined by one clause in
`_can_skip_unchanged_scan`, so no watermark is cleared or forged.

For the reprocess guard of #530, the flag **releases an embed model and keeps
an llm one**. The guard exists to stop unannounced provider *spend*: an embed
model reads its vectors back out of its own target by input hash, so an
announced reprocess whose inputs have not moved costs nothing, while an llm
model has no warehouse-side reuse at all — only an optional local
`llm_cache.duckdb` that a clean or another machine does not have.
`--accept-reprocess` remains the way to say yes to that one.

`--reprocess-all` with `--full-refresh` is refused, and so is `--watch`.

## Alternatives considered

### Clear the model's state (`clear_state(scope)`)

What #655 proposed, and the obvious reading of "force a reprocess". It is the
one alternative that *looks* right, because it delivers the headline behaviour:
mutation-checked, a `clear_state` implementation passes every assertion about
reuse — same rows reprocessed, zero provider calls, vectors and `embedded_at`
identical.

It breaks three other things, none of them loudly:

- **Removal reconciliation reads the published state.** A stage finds what
  vanished upstream by anti-joining `stel_state` against the upstream in the
  warehouse (issue #428), or by a Python set difference where the id column's
  cast does not round-trip. Clearing the state empties the left side of both,
  so a row deleted upstream stays in the target — and an empty removal set is
  also what a run with no removals produces, so nothing reports it.
- **A transform stops being incremental.** `_run_incremental_transform` reads
  an empty state baseline over an existing target as "rebuild with a full
  replace", deliberately: a child-keyed upsert onto rows no per-parent state
  owns would leave orphan children. Clearing state trips that branch and turns
  an announced reprocess into a silent full rebuild — a different operation at
  a different cost.
- **A failed run loses the baseline.** Clear, then crash, and the next
  ordinary run has nothing to resume from. Ignoring state destroys nothing, so
  an interrupted reprocess is resumable.

The same reasoning rules out the smaller variant — a `stel state clear`
command that leaves the next ordinary run to do the work. Non-destructive
behaviour is only reachable from inside the run.

### A flag that only affects embed models

The spend is all in embed, so scoping it there is tempting. But a flag whose
name says "all" and whose effect is one kind is a trap, and the
non-destructive shape makes the other four cheap rather than dangerous: chunk
and transform are CPU only, extraction re-parses (and re-fetches for remote
sources), and llm is the one that costs money — which the guard already
refuses rather than this flag excluding it. A SQL transform is the one genuine
no-op: it holds no per-record state, its skip is the `is_incremental()` branch
in its own SQL, and rendering that False is `--full-refresh`.

### Let `--reprocess-all` satisfy the guard for every kind

Simpler, and wrong for llm: the flag would then be a single word that re-pays
for a whole corpus of provider calls with nothing in front of it. The
asymmetry is the point — the flag is an announcement that the *reprocess* is
intended, not that spending for it is.

### Require `--accept-reprocess` alongside it for every kind

Safe, and it puts two flags in front of the one recovery this exists for,
after an incident where the expensive path was already the only one available.
Rejected as friction in the wrong place; the embed case is the case.

### Widen `--full-refresh` with a `--keep-vectors` modifier

Keeps one flag for "reprocess everything", but `--full-refresh` also drops the
target and replaces it (`use_full`), which is incompatible with reusing what
the target holds. The modifier would have to switch off most of what the flag
means.

## Consequences

- One documented way to reprocess an embed model's whole corpus at no provider
  cost while its inputs are unchanged. `stel plan --reprocess-all` prices it
  first.
- `stel plan` reports `changed` with every published row under the flag, so
  the guard — which reads the same plans — sees the real number. A model whose
  `code_version` has not moved no longer plans as `unchanged` under it.
- **Five stage entry points gained a required parameter.** Deliberate: a
  default would make the flag a silent no-op for any kind the runner forgot to
  pass it to. It surfaced a caller nobody had in mind, the dbt Python-model
  path in `dbt_embed/api.py`.
- The flag does nothing for a SQL transform, and that is documented rather
  than warned about. If this bites, a preflight warning for a selection the
  flag cannot affect is the next step, not a change in meaning.
- **This does not make a poisoned state detectable**, only cheap to leave.
  Finding out that a column is missing is still the operator's job, and
  ADR-0022's refusal is what keeps a *new* one from happening.

## Evidence

- **The cost that motivated it**, from #653 and #655: astrolabe's corpus is
  3.6M rows and about 28k Vertex requests, and recovering one dropped string
  column by `--full-refresh` pays all of them again. Not reproduced here.
- **Reuse holds under the flag**: 2 rows reprocessed, 0 skipped, 0 provider
  calls, 2 cache hits, and `chunk_id`/`embedding`/`embedded_at` byte-identical
  before and after
  (`test_reprocess_all_puts_every_row_through_again_and_pays_for_none`).
- **The rejected design is a plausible wrong answer, not an obviously broken
  one**: implementing the flag as `clear_state(scope)` passes the reuse test
  above and fails `test_reprocess_all_still_reconciles_a_removal`, where a row
  deleted upstream stays published. That asymmetry is why this ADR exists.
- **Both skip tiers are pinned separately.** Without the
  `_can_skip_unchanged_scan` clause, a second untouched run under the flag
  still reports `unchanged` — the stage is never entered, and the flag is a
  no-op in exactly the case it is for
  (`test_reprocess_all_declines_the_unchanged_parent_skip`).
- **The guard asymmetry is pinned in one test**, both halves: `chunk_embeddings`
  runs under the flag alone, `chunk_facts` still refuses and names
  `--accept-reprocess`. Both are on the default `on_code_change: fail` with
  `reprocess_limit: 0`.
- 7 of 8 mutations caught on the first pass. The miss was the mutation itself
  being unreachable: it swapped the mapping passed to the removal path, which
  the warehouse anti-join never reads. Re-run against the real counterfactual
  (`clear_state`), the test fails as it should — and the docstring that had
  described only the Python fallback was corrected.
