# ADR-0008: Paid models refuse an unannounced reprocess by default, and the guard reads the plan

- **Status:** accepted
- **Date:** 2026-09-07
- **Prompted by:** #530

## Context

Every incremental decision compares a stored `(input_fingerprint,
code_version)` pair against a recomputed one, and `code_version` is model-wide:
a configuration change moves it for every published row at once. For the two
kinds that pay a provider per row — `embed:` and `llm:` — that makes the next
run after a config change a corpus-wide re-spend, and nothing announced it.
The run budget (`max_api_calls`) was the only backstop, and the astrolabe
profile sizes it at 150k calls as a tripwire for runaway loops; a deliberate
re-embed of 3.6M chunks (~28k requests) passes under it without comment.

The cascade is the larger version of the same problem, and the one #529's
audit found written by hand in a YAML comment: a chunk model whose
`chunk_size` changes re-keys every chunk id, so the embed model below it —
its own configuration untouched — sees every row as new and pays for all of
them. No check inside the embed stage can see that coming, because by the
time the embed stage runs, the chunk model above it has already re-run and its
state carries the current `code_version`. The signal is gone.

Search already had the answer for its own kind: `on_index_change: fail`
refuses a rebuild-required change and names the field. Embed and llm needed
the equivalent, and #530 left two questions open: whether the default should
refuse or warn, and where the check should live.

## Decision

`embed.on_code_change` and `llm.on_code_change` default to `fail`, with
`reprocess_limit` (rows, default 0) as the tolerance. Before the first model
runs, `run` and `build` plan the whole selection with the same code `stel
plan` uses (#529) and refuse the run if any guarded model's plan says more
than its limit of published rows would reprocess, whether because its own
configuration changed or because a model above it did. `--accept-reprocess`
and `--full-refresh` are the operator saying so; `new` and `full` models never
trip it. The policy fields are excluded from `code_version`.

## Alternatives considered

### Default `reprocess` with a warning

The backward-compatible choice: nothing an existing project does today would
start failing. It lost because a warning is exactly what the incident record
says does not work. The 28-hour, 48GB embed run that ADR-0005's theme (ALE-55)
opens with logged its way to a MemoryError; `turbulence_index_job` in the
downstream project logged twenty successes over a table that had not gained a
row in four months. Output that a human has to read to be protected is not a
guard. The theme this shipped under states the position directly: "refusing to
spend should be the default for the kinds that spend." The upgrade cost is one
`--accept-reprocess` per deliberate change, and the changelog says so.

### A check inside each execution path

The obvious place: `_run_embed_model` and `run_llm_model` already load
`processed_state` before the first provider call, so counting rows whose
`code_version` differs is free there. It lost on the cascade. That count is
zero for an embed model whose chunk parent re-keyed, because the embed model's
own `code_version` did not move; the re-keyed rows arrive as *new* keys, and a
first build also arrives as new keys, so "many new rows with existing state" is
ambiguous between a re-key and a large ingest without reading which state keys
went missing upstream. Both stages deliberately gave up holding the id domain
(#428) to keep memory bounded, and a warehouse anti-join count is a new
adapter method per warehouse. The plan already classifies the cascade from
the DAG and each model's `code_version` alone, one aggregate query per model,
before anything runs. Reading it is both cheaper and sees more.

### A fraction of existing state as the threshold

Proposed in #530 as one of three shapes. It lost because `code_version` is
model-wide: the stale count is either every row or the remainder of an
interrupted reprocess, never a meaningful fraction. The one case a threshold
serves is tolerance — a small model, or a resume after an accepted run was
killed — and an absolute row count is the legible knob for that. Cost was the
third shape and needs tokens the plan cannot know; #529's follow-up notes carry
it.

### Guarding `backend: llm` extraction and `uses_llm` transforms too

They spend money as well. Deferred rather than rejected: extraction state is
document-keyed and its batch submission bills differently, and a `uses_llm`
transform fans out per parent by its own code, so neither has a row count the
plan can put a number on yet. When they do, the same field applies.

## Consequences

- **A run over a guarded model costs one aggregate query per selected model
  before it starts** — the plan's queries, over the whole selection, because
  the cascade an embed pays for starts above it. A selection with no guarded
  model skips them. On BigQuery that is about a second per model; a run that
  takes hours does not notice, a tight development loop over many models
  might.
- **`--source-filter` and `--read-filter` runs are guarded against the whole
  published state**, not the slice. A partition run over a changed model
  refuses until told otherwise, which is conservative in the right direction.
- **Resuming an accepted reprocess that was interrupted trips the guard
  again** for the remainder, because acceptance is a flag, not state. The
  operator passes the flag again or sets `reprocess_limit`. Recording
  acceptance in the warehouse was considered and is not worth a second thing
  that can disagree (ADR-0005's argument).
- **Tests and examples that reconfigure a paid model and re-run must now say
  so** with `accept_reprocess=True`. That is the point.

## Evidence

- #529's audit, 2026-09-06: `depends_on` is part of `code_version` for chunk,
  embed, llm, and search models (`src/stel/versioning.py`); the astrolabe
  project's `sec_chunk_embeddings.yml` reasons out its own re-embed by hand.
- `tests/test_reprocess_guard.py::test_chunk_change_is_refused_and_every_way_forward_works`:
  a `chunk_size` change on the rag example is refused for all three paid
  models downstream, none of which changed, and passes with
  `accept_reprocess=True`.
- The 150k-call budget in `astrolabe/constellations/profiles.yml`, described
  there as "a TRIPWIRE, not a planned stop".
