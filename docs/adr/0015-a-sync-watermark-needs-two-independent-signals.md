# ADR-0015: A sync watermark needs two independent signals, not one

- **Status:** accepted
- **Date:** 2026-10-01
- **Prompted by:** #573, #611

## Context

A no-op incremental run still cost 5m40s across 6 models on the reddit
example, 137s of it a search index publishing nothing, because every model
reopens and rescans its full parent to rediscover that nothing changed, even
when the run's own upstream already proved it.

A first attempt used a same-invocation signal: skip a model's scan when its
immediate parent wrote 0 rows *this run* and the model's own `code_version`
matched every published row. It was built, tested against a new suite, and
reverted after the repo's own existing tests caught two distinct failures:

- `test_metadata_only_update_rewrites_chunks_with_stable_ids`: a raw `UPDATE`
  on a parent's output table, with no model run involved at all. The parent
  then honestly reports 0 rows written next run.
- `test_failed_publication_leaves_no_stale_state_and_preserves_others`: a
  publish failure rolls back atomically. The parent already committed its new
  content *before* the failure, so on retry it truthfully reports 0 new rows,
  while the child's own state can't see that its row is stale.

Both reduce to one root cause: "the parent wrote 0 rows this invocation" does
not imply "the parent's current content matches what the child already
consumed." A persisted, cross-invocation watermark was proposed instead
(issue #611) — but a second design pass found the same shape of problem one
layer down.

## Decision

**Two signals, read fresh every time, both must match.**

**`state_generation`** — cheap: one aggregate query over `stel_state`, a row
count and the most recent `last_run_at`. `last_run_at` is stamped only on a
row actually written, so it never advances on a scan that found nothing. By
itself it has its own blind spot, found by this feature's own regression
suite: removing a row touches no surviving row's own timestamp, so a
timestamp-only signal cannot tell a deletion happened. The row count closes
that — a same-count replace (one row removed, a different one added) is still
caught because the replacement is a write, which does move `last_run_at`.

**`table_content_fingerprint`** — the authoritative confirmation, paid only
once the cheap signal already looks unchanged: a warehouse-side aggregate hash
over the parent's *actual current rows* (`bit_xor(hash(t))` on DuckDB,
`BIT_XOR(FARM_FINGERPRINT(TO_JSON_STRING(t)))` on BigQuery, both verified
empirically to change under `UPDATE`, `DELETE`, and `ALTER TABLE ADD COLUMN`).
This is what closes the gap the first attempt could not: a write that never
touched `stel_state` at all. Running this repo's own test suite against the
wired-up skip surfaced that this is not a rare case — at least 19 existing
tests across `test_chunking.py`, `test_embedding.py`, and `test_embed_flush.py`
edit a model's output table directly with raw SQL as a cheap way to simulate a
scenario, which is exactly the channel `state_generation` cannot see.

**Neither signal alone is sufficient, so both are required.** A plain content
hash alone would be sufficient on its own and would not need
`state_generation` at all — but paying a full aggregate scan of the parent on
*every* eligible run defeats a meaningful share of the point, so the cheap
signal exists specifically to avoid paying the expensive one except when it
is actually needed to confirm a true no-op.

**A watermark is invalidated by any state-table migration.** A v1 state row's
fingerprint was keyed on a document, not necessarily the grain v2's
`record_key` expects (a chunk model needs one per chunk, not one per source
document) — carried over unchanged by `_migrate_v1_state`, verified only by
row count. A child trusting a stale watermark across that shape change could
skip the one real scan that would notice its migrated fingerprints are wrong
for their new grain, so the migration drops every sync watermark in the same
pass, not just the model it happens to run for.

## Alternatives considered

### Trust the same-invocation signal, narrowed further

Restrict the original design to cases provably safe from the two known
failures (e.g. only when no model in the DAG has ever failed). Rejected: there
is no cheap way to know "has this model's last attempt ever failed" without
persisting exactly the kind of cross-invocation state this ADR ends up
building anyway, so the narrowing collapses into the chosen design with extra
steps.

### Content fingerprint only, no cheap pre-filter

Simpler code, one signal, always authoritative. Rejected on cost: the
mechanism exists because a full scan of a multi-million-row corpus is
expensive, and a content fingerprint is still a full scan, just a cheaper and
entirely server-side one. Computing it on *every* eligible run — including
every run where something genuinely changed, which `state_generation` catches
far more cheaply — would spend real warehouse compute (and, on BigQuery,
real billed bytes scanned) on confirmations that were never going to matter.

### Detect raw-SQL mutation some other way (a trigger, a checksum column)

A database trigger or a maintained checksum column updated on every write
would need to exist on tables stel does not fully own the DDL lifecycle of,
and would not fire for a write that bypasses stel's own write path by
definition — the same gap, moved rather than closed. Re-deriving the
fingerprint from the table's actual current content, each time, is the only
approach that does not depend on the mutation having gone through code that
could have been bypassed.

### Let the content-hash gap stand as a documented limitation

Ship `state_generation` alone and document "an externally-mutated table is
not detected" as a known boundary, the way other invariants in this codebase
assume stel is the sole writer to its own tables. Rejected once the blast
radius was measured: this repo's own tests rely on exactly that channel
routinely (not a rare hypothetical), so the same shortcut that makes those
tests convenient to write is a channel a real deployment could hit too.

## Consequences

**The skip's net benefit is smaller than "nothing at all" for a model whose
parent is large.** A no-op run now costs one cheap aggregate query plus,
whenever that looks unchanged, one full server-side scan of the parent —
still far less than today's read-into-Python-and-classify pipeline, but not
free. This is the deliberate trade of the two-tier design: free in the
common case the cheap signal already resolves (a real change happened
somewhere), one scan in the case that matters most (truly nothing changed).

**`search:`, `ml:`, and `eval:` models are excluded.** A search publish also
sweeps stale retrieval generations inline, a side effect this must not
suppress; `ml:` and `eval:` do not fit the same state-scoped-by-model-name
contract the checks read.

**A handful of existing tests needed updating, not weakening.** Three tests
asserted that a specific reconciliation mechanism (an anti-join page, a
resume-reuse metric, a bounded-streaming read) *ran and reported zero* on a
no-op corpus. The skip makes the same guarantee true a cheaper way — the
mechanism does not run at all — so each assertion now branches on
`result.status == "unchanged"` and asserts the right invariant for whichever
path actually executed, rather than assuming only the old one could.

## Evidence

Read at `be26cd1`. DuckDB's `hash(t)` and BigQuery's
`TO_JSON_STRING(t)`/`FARM_FINGERPRINT`/`BIT_XOR` were verified directly
against a live BigQuery project and a local DuckDB instance to change under
`UPDATE`, `DELETE`, and `ALTER TABLE ADD COLUMN` before being relied on. The
19-test figure for raw-SQL-mutation testing patterns and the full regression
suite run (`tests/test_chunking.py`, `test_embedding.py`, `test_embed_flush.py`,
`test_incremental_transforms.py`, `test_bounded_memory.py`, plus the fast and
e2e tiers in full) are from this change's own validation.
