# ADR-0024: a pruned incremental MERGE first proves no matched row lies outside the batch's layout values

- **Status:** accepted
- **Date:** 2026-10-09
- **Prompted by:** #664

## Context

An incremental publish on BigQuery is one MERGE of a staging table into the
target, joined on the key. BigQuery reads every partition and block of the
target that might hold one of the batch's keys. When the target is partitioned
and clustered on columns the key says nothing about -- astrolabe's embeddings
table is partitioned by filing month and clustered by `symbol, form_type`,
keyed on a content hash -- that is most of the table: 4.45 GiB billed per
flush against 28.5 GiB, 4 TiB over one backfill. The same shape costs the embed
resume's keyed read-back 21 GiB per lookup.

BigQuery prunes on a constant predicate over the partition or clustering
column. The batch's values on those columns are known: the staging table holds
them. Adding `AND target.filing_date BETWEEN lo AND hi AND target.symbol IN
UNNEST(symbols)` to the join prunes the read to the batch's partitions and
blocks. The hazard is a target row whose key is in the batch and whose layout
values are not -- a filing re-dated into another month, a row whose symbol was
corrected. The pruned join does not see it, the MERGE's `WHEN NOT MATCHED`
inserts the key a second time, and the next `unique` test is the first thing to
notice. The adapter's incremental contract promises one row per key; a saving
that can break it is not one stel can turn on.

## Decision

The publish is one script. It reads the batch's layout values from the staging
table into script variables, then asks, with one join of the target to the
staging table on the key projected to the key and layout columns only, whether
any matched target row falls outside the pruned predicate (a NULL layout value
counts as outside). If none does, the pruned MERGE runs; if one does, the
unpruned MERGE runs. Either way every key ends with one row, and the operator
configures nothing: the pruning is on whenever the model declares a layout and
the batch carries the columns without NULLs, with a value list capped at 1,000
distinct values so a per-row id clustered beside the key is not sent as an
array the size of the batch. The embed reuse read carries the same predicates
without a guard, because for it a missed row is a paid re-embed and never wrong
output.

## Alternatives considered

### Add the predicates unguarded, as dbt's `incremental_predicates` does

dbt-bigquery lets the operator write the pruning predicate into the merge
condition and documents that a row whose predicate columns change is
duplicated. It is the right shape for a tool where the operator owns the
statement. stel's incremental publish is a contract the executors rely on --
state advances on the belief that the key is unique in the target -- and the
operator did not write the MERGE, so the operator cannot be the one to know
when the hazard applies. Declined: a saving that silently breaks the key
contract is a defect this repository has already paid for in another form
(a green materialization over wrong data produces no alert at all).

### Make the pruning opt-in configuration

A `merge_prune: true` under `warehouse_options` would keep the default safe
and let astrolabe opt in. Declined: it moves the same unanswerable question
("will any of my rows ever change partition?") to the operator, and the
default stays the 4 TiB behaviour for every project that does not read the
option's caveats. The guard makes the question moot, so the pruning can be on
by default.

### Run the guard as a separate query, then a plain MERGE with query parameters

Two round trips instead of a script; the MERGE stays a plain DML statement
whose `num_dml_affected_rows` is unambiguous, and the batch's values travel as
query parameters. Declined, narrowly: it is two jobs per flush where one did
the work, for a publish that already issues thousands of flushes per backfill,
and it would introduce query parameters as a second pruning mechanism beside
the script variables this adapter already relies on for partition pruning
(`insert_overwrite`'s `IN UNNEST(stel_partitions)`). One documented mechanism
is exercised rather than two. Neither form closes the window between the guard
and the MERGE against a concurrent writer; overlapping runs against one target
were never coordinated by the publish and still are not.

### Prune by the partition range alone and skip the clustering columns

The partition range is the simpler predicate and needs no cardinality cap.
Declined as the whole answer: #664 measured that an embed batch drawn from
across the corpus spans most months, so BigQuery's own dynamic pruning from the
join already left little for a range to add -- the clustering columns are
where the saving is for that table.

## Consequences

- The guard is a read of the key and layout columns across the whole target on
  every pruned publish. On a table whose MERGE was already cheap it is the
  visible cost, and on the SEC table a few hundred MiB against 4.45 GiB.
- The publish job is a script, so its parent job's `num_dml_affected_rows` is
  whatever BigQuery reports for a multi-statement job; the `rows_affected`
  field of the publication telemetry is less informative than it was for the
  plain MERGE. `rows_written` is counted from the batch and is unchanged.
- The pruning is on the *declared* layout. A target whose physical layout has
  not been rebuilt to match its declaration prunes nothing, correctly.
- A row whose layout values change still publishes correctly, at the unpruned
  price for that batch. A model whose rows do that routinely gets no saving
  and pays the guard; the telemetry is how to see it.
- The embed reuse read's miss on a re-laid-out row is a cost the docs state,
  not a guarded case.

## Evidence

- `region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT` for astrolabe's project,
  2026-08-08..10-08, as reported in #664: 917 embed MERGEs at 4.45 GiB each
  against a 28.5 GiB table; 103 reuse read-backs at 21.3 GiB each; per-symbol
  registry batches read 1.7 GiB of 4.7, which is what dynamic pruning from a
  narrow batch already achieves.
- BigQuery's documented pruning with script variables is what
  `_insert_overwrite_script` has relied on since issue #91; the same form is
  used here. The saving itself is not measured in this repository, which has no
  BigQuery credentials: `test_integration_layout_pruned_merge_keeps_one_row_per_key`
  pins the one-row-per-key property against a live project when
  `STEL_BQ_TEST_PROJECT` is set, and astrolabe's next full-corpus publish is
  where the bytes-billed number comes from.
