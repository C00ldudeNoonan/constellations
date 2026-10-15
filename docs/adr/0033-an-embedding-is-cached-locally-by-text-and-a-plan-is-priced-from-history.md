# ADR-0033: an embedding is cached locally by text, and a plan is priced from history

- **Status:** accepted
- **Date:** 2026-10-15
- **Prompted by:** #665 (asks 2 and 3; ask 1 is [ADR-0026](0026-embedding-reuse-is-keyed-by-content-and-fetched-by-id.md))

## Context

astrolabe's SEC corpus (3.67M chunks, `text-embedding-005`) was embedded
about 1.9 times. ADR-0026 fixed the id-space half: reuse is keyed by text, so
a re-keyed row keeps its vector. It deliberately left three cases open,
because the vector in each is somewhere target reuse cannot look:

1. **A crash before a flush.** #401 made embed publish at flush boundaries,
   but a vector embedded after the last flush lives only in memory. The
   08-27 backfill was relaunched through repeated ~1.3h failures.
2. **Duplicate text within a run.** The target index is read once, before the
   first window, so it cannot know a vector the run produced a window ago.
3. **Duplicate text across runs once the target row is gone** — deleted, or
   rebuilt by `--full-refresh`, which turns target reuse off.

Separately, `stel plan` reported rows and requests but no dollars, so a
launch comment for a large reprocess carried no number. A plan reads no
source and no model table, and what a row costs depends on its text.

## Decision

**Every vector a provider returns is written, as soon as its batch returns,
to a local DuckDB file at `embedding.cache_path`, keyed on
`(embedding_input_hash, embedding_config_hash)`, and every window consults it
before the target.** Within a window, identical text is sent once whether or
not a cache is configured. **`embedding.pricing` prices embeds** (per million
tokens and/or characters, characters counted by stel), and **`stel plan`
prices each reprocessing model at its recent cost per provider row, read from
the run log**: pooled `estimated_cost_usd / (rows_processed - cache_hits)`
over its last five priced runs, times the rows to reprocess, shown as a
ceiling.

## Alternatives considered

### Keep the cache in the warehouse, beside the target

A stel-owned table would follow the project between hosts, which a local
file does not. Lost on the write pattern: the cache must be written per
provider batch to protect the crash window, and on BigQuery a single-row DML
INSERT was measured at 2.2–2.6s (#541) — slower than the provider call it
would follow, once per batch, ~14,700 times for this corpus at a batch of
250. The warehouse already holds the durable copy (the target); the cache's
job is the gap between a provider response and the next publish, and that
gap is local.

### Write the cache per window instead of per provider batch

One write per flush instead of one per batch, and simpler. Lost because it
reopens exactly the window the cache exists to close: a crash between the
first batch of a window and that window's write loses the batch, which is
case 1 at a smaller scale.

### Store vectors as float32

Halves the file (~11 GB instead of ~22 GB for this corpus). Lost because the
target stores `Float64` and a vector restored from the cache would then
differ from the one the provider returned, so a `--full-refresh` rebuild
would change every vector in the index while reporting nothing changed.

### Dedupe across the whole run in memory, with no file

Would cover case 2 without a cache file. Lost on #401's residency contract:
a map from every distinct text hash to its vector is the whole corpus's
vectors held at once — ~22 GB at this size — on the path that exists because
the corpus did not fit. Deduping *within a window* is bounded by
`flush_every`, so it is kept, and it is why ADR-0026's "wrong layer" note
about in-run dedupe no longer quite applies: the window-sized half needs no
layer at all.

### Fail the run when a cache write fails

The usual rule — a durability feature that cannot persist should say so.
Lost because of what is in hand at that moment: vectors already paid for,
one publish away from the warehouse. Raising discards them, which is the
spend the cache exists to prevent. The run logs one warning and continues
without the cache. Opening the cache is the opposite case and does fail the
run, because it happens before any provider call.

### Price a plan from the upstream's text lengths

Exact for character-billed embeds. Lost because it reads the upstream model
table, which a plan promises not to do, and on BigQuery a scan of the text
column is itself a billed query — a plan that costs money to tell you what a
run costs.

### Price a plan from history's tokens times current rates

Would track a price change immediately rather than one run later. Lost
because the run log has no characters column, and Vertex text embeddings
bill characters; adding one is a schema change to an existing append-only
table (`RUN_LOG_SCHEMA` is a contract, and `append_rows` refuses a new
column on an older table). Cost per row already folds the right unit in.

### Price a plan per provider request

The plan already counts requests. Lost because a request's cost is the
texts it carries, which is the provider's batch split, not anything the run
log records.

## Consequences

- `--full-refresh` no longer re-embeds when a cache is configured. Forcing a
  re-embed means deleting the cache file; the reference says so.
- The cache grows without bound: ~8 bytes per dimension per distinct text.
  There is no eviction; deleting it is always safe.
- A plan's cost lags a price change by one run, and is `-` until a model has
  a priced run in the run log. It is a ceiling: reuse, caches and duplicates
  only lower it, and the plan can see none of them.
- `rows_processed - cache_hits` is the per-row denominator for every paid
  kind, so a kind whose `cache_hits` is not in row units is priced only
  approximately. Embed counts hits per row and an `llm:` model per row (one
  call each); `backend: llm` extraction counts them per provider call against
  documents processed.
- Plan artifact schema moves to version 2 (`estimated_cost_usd` per model,
  `metadata.cost_source`).

## Evidence

- BigQuery DML latency (2.2–2.6s per INSERT, 2.7–3.2s per DELETE against
  astrolabe's prod dataset) is #541's measurement.
- Vertex `text-embedding-005` pricing ($0.025 per 1M input characters) and
  the 7.78B characters billed are from #665's Cloud Monitoring figures.
