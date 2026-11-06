# ADR-0052: embed reads its upstream in the target's layout order

- **Status:** accepted
- **Date:** 2026-11-05
- **Prompted by:** #749 (ALE-556), following #664

## Context

#664 made the BigQuery publish prune: the MERGE and the embed reuse read-back
carry a range on the partition column and a value list on each clustering
column, built from the batch's own rows. They skip storage only as far as the
batch is *local* in those columns.

Embed cuts its windows from the upstream stream in whatever order the warehouse
returns it, which spans the whole corpus. A 128-row window against astrolabe's
`sec_chunk_embeddings` (3.67M rows, partitioned by `filing_date_dt`, clustered
on `symbol, form_type`) covered nearly every month and dozens of symbols, so
the predicates were true of almost every block. Measured on the 2026-10-22
build: about 46 GiB of BigQuery per window before Vertex was called.

## Decision

Ask the adapter for the upstream **in the target's layout order**: the
partition column, then each clustering column, then the id. A window is then a
few symbols in a few months, and the existing pruning does its job unchanged.

- `table_snapshot` takes `order_by`, gated by a new capability,
  `ORDERED_TABULAR_READS`. An adapter without it refuses the read; it is never
  answered unordered, because the caller would believe its windows local.
- The order is ascending with NULLs first, stated in SQL on both adapters
  (DuckDB defaults to NULLs last).
- The id is last. That makes the order total, so ties break the same way on
  every read and a resumed run, reading the rows still to do, cuts the windows
  the failed run had left.
- On BigQuery an ordered read always goes through the query job, because the
  direct Storage Read and the key-segmented read cannot sort. The session
  reads the sorted destination table through one stream.
- Embed asks only when the target declares a layout and the upstream carries at
  least one of its columns. Otherwise the read is unchanged, and the call carries
  no `order_by` keyword at all, so an adapter that predates it is never handed it.
- **An ordered open that fails falls back to the unordered read**, with a warning
  that says the publish will prune less. Ordering buys pruning and nothing else, so
  a warehouse refusing to sort a wide, large result (BigQuery's "resources
  exceeded") must not end a run that has embedded nothing yet. Only the open falls
  back: a read that fails while streaming has already put rows in a window and ends
  the run. A changed table is not retried.

## Alternatives rejected

- **Sort a bounded buffer inside embed.** Adapter-agnostic and no seam change,
  but the locality is the buffer's rather than the corpus's, and windows would
  depend on arrival order, so a resume would cut different windows. Both defeat
  the point.
- **Cut windows from the work set, not the stream.** Collect the rows that need
  embedding, sort them, then window. Exact, but it holds the work set in
  memory, which the bounded-memory contract (#401) rules out for a full
  re-embed.
- **Add layout columns to the existing unordered read and trust the warehouse.**
  Neither BigQuery's query path nor the Storage Read API promises an order for a
  clustered table.

## Consequences

- A DuckDB snapshot validates itself by re-reading and comparing a digest in
  row order. The re-read now carries the same `ORDER BY`; without it every
  ordered read would have reported its own table as changed.
- An ordered BigQuery read is billed as a query over the projected columns,
  plus the sort. Embed's upstream read was already a `SELECT *` query job, so
  this adds the sort and nothing else.
- Resume determinism depends on a BigQuery single-stream session returning a
  sorted destination table in the order it was written. That is how it
  behaves, not something the Storage Read API documents. A credential-gated test
  (`STEL_BQ_TEST_PROJECT`) reads a shuffled 3,000-row table back in both shapes
  embed uses and fails without the `ORDER BY`; it shows the mechanism, not that it
  holds at corpus scale. If it ever stopped holding, windows would lose locality
  and cost would return to the old level; results would stay correct.
- The sort of a wide `SELECT *` at corpus scale is not exercised by any test. The
  fallback above is the answer to it failing, and a run that took it is visible in
  the warning. Sorting only the layout columns and id, and fetching text by id,
  would avoid sorting wide rows; it was not built because it needs the text fetched
  per window, which costs a scan of the text column each time.
- **Not changed: the strays guard.** #749 asked for the prune guard to be
  scoped to the batch's range. That cannot be done: the guard exists to find
  target rows whose key is in the batch and whose layout values are *outside*
  it, so a range on those columns would exclude exactly what it looks for. It
  reads the key and layout columns of the whole target, which is narrow. The
  21 GiB attributed to it in the measurement most likely includes the child
  MERGE, which a script reports under its parent job. Measure the guard's own
  bytes before changing it.
- `transform:` merges on partitioned targets are not covered. They share the
  pruned MERGE but not this read.
