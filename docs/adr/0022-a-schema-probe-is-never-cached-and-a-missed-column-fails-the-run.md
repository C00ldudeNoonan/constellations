# ADR-0022: a schema probe may not be answered from a cache, and a column it missed fails the run

- **Status:** accepted
- **Date:** 2026-10-06
- **Prompted by:** #653

## Context

Five places read a relation's shape with `read_table(table, limit=0)` — chunk,
embed (twice), and llm (twice). They want column names and dtypes, nothing
else, and #410/#423/#424 narrowed them to a zero-row read precisely so a
contract check would stop pulling a corpus into memory.

That made them *metadata* queries wearing a payload query's clothes, and on
BigQuery they inherited the client default `use_query_cache=True`. BigQuery
does not evict a cached `SELECT * ... LIMIT 0` when the table gains a column:
the cached probe reported 51 columns while `get_table()` reported 52 at the
same moment, and the entry outlived the table's modification by at least seven
minutes. A zero-row, zero-byte result evidently does not carry the table
dependency that invalidates a cached result over data.

An embed model is where that answer does the most damage, because its output
schema *is* the probe's schema plus its generated fields, fixed for the whole
run. Every flush frame is built with it, and `pl.DataFrame(rows, schema=...)`
keeps the keys the schema names and discards the rest without a word. So the
upstream's new column was dropped from every row, state advanced for all of
them, and the sync watermark recorded the model as caught up. Nothing
recovered: both skip tiers matched on every rerun, the search model downstream
failed on the missing column every time, and the documented escape
(`--full-refresh`) also turns off vector reuse — 3.6M rows and about 28k Vertex
requests to restore one string column.

Two call sites inside the BigQuery adapter had the same hazard with a worse
outcome. The Iceberg paths probe a *staging* table whose name is deterministic,
and feed the answer to an explicit `CREATE TABLE`, so a cached answer from a
previous run's staging table builds the target with the wrong schema rather
than merely writing it short.

## Decision

A freshness-critical schema probe is a named adapter operation,
`read_table_schema`, separate from `read_table(table, limit=0)`. SQL adapters
inherit it; an adapter whose reads may be answered from a cache of any kind
must override it with one that cannot be, as BigQuery now does by passing
`use_query_cache=False` on the same `LIMIT 0` query. The two Iceberg staging
probes go through the same helper.

Independently, an embed model refuses to publish rows carrying a column the
probe did not see. The check runs at snapshot open — before the first provider
call — and names the columns. A column whose name collides with a generated
field is refused as a collision, by the same check the run already makes
against the probe, run a second time against the columns actually read.

## Alternatives considered

### Read the schema from `client.get_table()` instead of running a query

The issue's own first suggestion, and `_state_columns` already does exactly
this: free, authoritative, no job. It loses because these dtypes *become* an
embed model's output column types. Answering from `get_table()` means stel
owning a BigQuery-to-polars type mapping that would have to agree with the
Arrow path on every type forever; one disagreement silently changes what every
embed model writes, in a release whose stated change was a cache flag. The
uncached query is the same query through the same Arrow path, and `LIMIT 0`
bills no bytes, so what the cache was saving here is a job's latency rather
than scan cost.

### Turn the query cache off for every stel query

It would have fixed this and anything like it, and the diff is one line. It
loses on cost: the cache is what makes a repeated payload read cheap, and the
unchanged-parent skip (#611, ADR-0015) is built on the premise that a no-op run
is cheap. Paying for every re-read to fix a metadata staleness bug is a large
regression for a narrow cause.

### Fix the probe and trust it

The probe and the read are two queries at two moments, so even an uncached
probe can be overtaken by a writer between them. Without the refusal that gap
stays silent and self-confirming, which is the actual defect in #653 — the
staleness was recoverable, the silence was not.

### Widen the output schema mid-run instead of refusing

Tempting, and it would self-heal: `on_schema_change: append_new_columns` is
already what a non-first publication uses, so the MERGE would add the column.
It loses to the reason the schema is fixed in the first place (#401 review): a
passthrough column that happens to be all-NULL in the first flush infers as
Null, the target column is created from that, and a later flush carrying real
values fails on conversion *after* its provider calls are paid. Widening
mid-run puts that failure mode back.

### Compare the read's columns against the output schema

What the first implementation did, and it has a hole, found by Codex reviewing
#654. The output schema is the probed columns *plus* the generated fields, so a
late column carrying a generated field's name is already in it: the subtraction
reports nothing dropped, `_embedding_row` writes the generated value over the
upstream one, and the row publishes with state advanced — precisely the silent
loss the check exists to break. The comparison is against the probed columns
instead, which reports that column, and the collision check runs again against
what was read so the message matches the fault: the two refusals give opposite
advice, and "re-run to pick them up" would send a collision round a loop.

### Refuse at publish instead of at snapshot open

The drop happens at publish, so that is where the check reads most naturally.
It loses by one window of provider spend: publish is after the embedding calls
for those rows. Snapshot open is the earliest moment the read's own schema is
known, and it is before the first call.

## Consequences

- **One more BigQuery job per probe per run**, zero bytes but a round trip. A
  project with many chunk/embed/llm models pays that several times per run.
- **A writer touching an upstream mid-run now fails the run.** That is the
  intent, but it converts a previously "successful" (wrongly partial) run into
  a hard error, and stel does not coordinate overlapping runs against one
  target — that remains a project responsibility.
- **Nothing here recovers a state already poisoned** by the old behaviour.
  #653's third ask was unmet when this was written and is now closed by
  `--reprocess-all` (issue #655,
  [0024](0024-a-forced-reprocess-ignores-incremental-state-rather-than-clearing-it.md)),
  which reprocesses every row with vector reuse intact. Note that this ADR
  guessed the mechanism wrong: clearing a model's state does leave
  `reuse_reader` live, but it also stops removals reconciling and turns a
  transform's reprocess into a silent full rebuild, so the flag reads state
  and declines to skip on it instead of clearing it.
- **The content fingerprint that gates the unchanged-parent skip is still
  cached.** Examined and left: it is an aggregate over row data, where
  BigQuery's documented invalidation applies, and making it uncached would put
  a full scan back on exactly the no-op path ADR-0015 exists to keep cheap. If
  a stale fingerprint is ever observed, this is the place to look first.
- The llm and chunk stages get the fresh probe but need no refusal: neither
  passes upstream columns through to its output, so neither can drop one.

## Evidence

- **The cache does not evict on a column addition.** From #653, against a
  sandbox BigQuery dataset on stel 0.20.0: `use_query_cache=True` gave
  `cache_hit=True`, 51 columns, `section_topic` absent; `use_query_cache=False`
  gave `cache_hit=False`, 52 columns, `section_topic` present; `get_table()`
  reported 52 at the same moment. The stale entry outlived the table's
  modification by at least seven minutes and survived an uncached run of the
  same query. Reported, not reproduced here — this session had no live
  BigQuery. `test_integration_schema_probe_sees_a_column_added_since_the_last_probe`
  pins it for the next run that has credentials.
- **Uncaching is sufficient for the forward path.** Also from #653: with every
  stel query forced uncached from a clean sandbox, the same chain ran clean —
  `section_topic` published on all 11,897 rows with 0 provider calls and 11,897
  cache hits, every vector, `embedding_input_hash` and `embedded_at` unchanged,
  and the search model republished in place.
- **The generated-field collision was reachable late.** With the output schema
  as the comparison, an upstream that gained an `embedding_model` column
  between the probe and the read published 2 rows, reported success and
  advanced state, with the upstream's values replaced by the generated ones:
  `test_a_late_generated_column_collision_is_refused_not_overwritten` fails
  that way against the pre-review code. Its sibling,
  `test_embed_rejects_generated_columns_case_insensitively`, turned out to
  pass even with a case-sensitive collision check — via a duplicate-column
  error from DuckDB at publish, after the id scan and the provider calls the
  early refusal exists to avoid — so both now pin the message rather than the
  column name.
- **polars drops silently.** `pl.DataFrame([{"a": 1, "b": 2}], {"a": pl.Int64})`
  returns `columns == ["a"]` with `height == 2` — no error, no warning.
  Checked 2026-10-06 against polars 1.40.1, google-cloud-bigquery 3.42.1.
