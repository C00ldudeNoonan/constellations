# Bounded memory: peak follows the flush window, not the corpus

## The contract

> For every model kind, peak memory is O(flush window) + O(per-parent unit) +
> O(distinct keys) — never O(corpus **bytes**).

Corpus size should be bounded by warehouse capacity, not by process memory. A
run whose input does not fit in RAM should still finish.

### The key term is real, and deliberate

The third term is not a rounding error and should not be read as one. Stages
that reconcile deletions hold every id at once: `_stream_upstream_ids` and
`_stream_document_ids` return a `set[str]` of the upstream keys, and a resuming
embed additionally holds `_EmbeddingReuseReader._target_keys` for the existing
target. Measured, a Python string set costs ~108 bytes per id:

| corpus | one key set | a resuming embed (two) |
|---|---|---|
| 200k rows | 21 MB | 42 MB |
| 3.6M rows | ~370 MB | ~740 MB |

What the fixes bought is that this scales with row *count* and not row *width*
— for the 3.6M-chunk corpus, ~370MB of ids instead of ~7GB of text and vectors.
That is the difference between a run that finishes in a 4GB container and one
that cannot finish at all. But it is not constant, and an operator sizing a
container needs the number rather than the word "bounded".

Eliminating it means pushing removal detection into the warehouse as an
anti-join rather than a Python set difference — tracked as issue #428.

**The residency tests do not cover this term.** They measure the largest single
frame or batch an adapter hands a stage, which is exactly the O(corpus bytes)
failure the incidents were; a cumulative container is invisible to them. That
limitation is why the term is written down here.

This is stated as an invariant rather than left as a property of each stage
because the same root cause reached production five times, through five
different stages, and each fix was correct and local:

| incident | stage | what was corpus-sized |
|---|---|---|
| #383 → #385 | transform | whole-table dependency reads, per-row dict fingerprints |
| #401 → #402 | embed output | every vector accumulated for one end-of-run publish |
| #407 | embed resume | the whole existing target, vectors included |
| #410 → #411 | embed input | `read_table(upstream)` before the first provider call |
| #418 → #420 | BigQuery snapshot | an unpartitioned `OVER()` made the warehouse buffer the projection |

The sequence is the argument. Five local fixes did not converge on a bounded
system, because nothing said what bounded meant.

## What the invariant costs a stage

A stage holds the contract when all of these are true:

- **Schema comes from a zero-row read.** `read_table(t, limit=0)` returns
  column names and dtypes without the rows.
- **Rows arrive in batches.** `table_snapshot()` streams; `read_table()` does
  not. `STREAMING_TABULAR_READS` has been required of adapters since #385.
- **Whole-column checks are projected.** Validating a key domain means reading
  that column, not the payload beside it — on both sides of the wire (#418).
- **Output publishes per window.** `FlushPublisher` writes and advances state
  every `flush_every` rows, so a failure re-pays one window rather than the run.
- **Validation that can fail the run happens first.** A NULL or duplicate key
  should fail before the first provider call, not after the corpus is paid for.

## What enforces it

**`tests/test_bounded_memory.py`.** Two mechanisms, deliberately different:

*The shape.* Every `adapter.read_table()` call site in `src/stel` is classified
in a table — `bounded`, `exception` with a reason, or `gap` with a tracking
issue — and an AST scan fails when a new one appears unclassified. `read_table`
is `SELECT *` into one frame, so an unreviewed call on a corpus-scale relation
is an O(corpus) peak. This makes a whole-table read an argued decision rather
than the default that five incidents made it.

*The property.* A stage is run against a real warehouse with the adapter
instrumented to record how many rows it ever materializes at once. A regression
to `read_table(upstream)` fails with the offending call site named.

**Why residency and not memory.** The obvious gate — run a stage under a memory
ceiling — measures peak working set, which is an allocator high-water mark
rather than what is held. Measured on the DuckDB read path, a streaming loop
grows +203MB across a 4× corpus purely from allocating and freeing per-batch
frames, against +528MB for a whole-table read. Real separation, but a factor of
1.7 on a number that also moves with the platform allocator — not a margin to
fail a build on. Counting rows is exact and cross-platform.

**The limits of both.** The scan sees only what the source says: #418 was
corpus-sized buffering *inside BigQuery*, from SQL that streamed correctly on
stel's side, and no call-site audit would have found it. The property test
needs a warehouse and a fixture, so it covers a stage at a time rather than
everything. Neither replaces measuring a real corpus.

## Where it does not hold yet

Tracked, and the list may only shrink — `test_the_unbounded_list_only_shrinks`
fails if a stage is added to it:

- **#423 — chunk** reads its whole upstream registry before chunking anything.
  The worst-placed of the two: chunk feeds embed, so a corpus can be
  OOM-killed here before any of embed's bounded-memory work is reached.
- **#424 — llm** reads its whole upstream before the first provider call, and
  its whole existing target to collect id values. The #410 and #407 holes,
  unfixed for the stage where a re-run costs the most per row.

Stages that read whole tables *by contract* are recorded as exceptions with
their reasons in the same table — classic ML training fits one matrix, a
transform full refresh needs every parent, the concept-cloud export has no
flush window to be bounded by. Those are decisions, not debt.

## Changing a fingerprint while bounding a read

Every one of these fixes had the same hazard, and it is worth stating once.
Incremental state compares an `input_fingerprint` computed from the upstream
record. If reading in batches produces even subtly different Python values than
reading whole — a widened dtype, a shifted datetime unit — every existing
corpus silently reprocesses at full provider cost, and nothing raises.

The end-to-end tests cannot catch it: they exercise one read path, and both
sides of the comparison move together. Pin the equivalence directly, the way
`test_streaming_the_input_preserves_input_fingerprints` does.
