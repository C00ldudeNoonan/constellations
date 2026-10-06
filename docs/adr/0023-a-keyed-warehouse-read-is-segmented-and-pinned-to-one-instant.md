# ADR-0023: a keyed warehouse read is segmented and pinned to one instant, and resumes from its last completed segment

- **Status:** accepted
- **Date:** 2026-10-06
- **Prompted by:** #614

## Context

A search publish reads its source table through one BigQuery Storage Read
session. BigQuery ends that session at six hours, and a full republish of a
3.64M-row corpus runs longer than that, so it died at roughly the halfway point
every time. Resume could not help: the next leg re-paged the whole table from
the start, so each leg was spent re-reading the rows it had already published,
and the legs got slower as the table grew.

The read has to survive the session limit without losing the property that
makes the publish sound: the rows it reads are one consistent relation, and a
row removed upstream after the read began is still reconciled correctly.

## Decision

A keyed, unfiltered read of an `INTEGER` or `STRING` key column is cut into
segments of 250,000 rows by key. Each segment is a read session restricted to
its key range, and every session reads the table `AS OF` one instant that BigQuery's
clock supplied when the read was planned. The adapter records, on the
collection, the instant, the segment boundaries and how many leading segments
have been fully yielded. A later attempt that adopts the collection re-plans
its read onto that record, skipping the completed segments, provided the
instant is within 48 hours. Past that, or for any other key type, the read
starts over, as it always did.

The completed count advances only once every row of a segment has been
yielded, and the caller has written each page before asking for the next one.
So a recorded position always describes rows that are already published.

## Alternatives considered

### Reopen one session per N pages, reading from the start each time

The obvious fix and the one the incident's first suggestion named. A session
cannot seek, so reopening at page N still reads pages 1 through N-1, and the
resume cost is quadratic in the number of sessions. Declined: it moves the
wall without removing the re-reading.

### Hash-partition the key with a fingerprint function in the row restriction

Would give segments of even size without a boundary query. Declined: the row
restriction's accepted functions are documented only by example, and a
restriction the service rejects fails the publish at its first session, not
at planning. A key-range restriction uses only comparisons the documentation
states.

### Approximate quantiles for the boundaries

`APPROX_QUANTILES` is one scan with no single-worker sort. Declined because its
accepted input types do not clearly include strings, and because exact
boundaries keep each segment at the stated row count, which the six-hour
arithmetic depends on. The exact boundary query is a window over the key column
alone, which is the narrow part of the table; this is the cost to measure on a
live run.

### Re-plan at open rather than after the stamp is read

Simpler: read the stamp first, then open. Not possible here. The snapshot's
schema feeds the config fingerprint, and that fingerprint is what selects the
resumable generation, so the snapshot has to exist before the stamp can be
read. The snapshot therefore opens on a fresh plan, and is re-planned onto the
recorded one before any row is read.

### Persist progress on the warehouse, not in the collection stamp

Would keep the store's schema unchanged. Declined: the progress has to be
written by the same process that holds the publish claim and read back by the
next one, and the collection stamp is already where the claim's other
generation facts (`source_generation`, `source_rows`) live.

### Keep the snapshot unpinned and fail a read whose table changed

What the single-session read did: the generation check failed the publish if
the table moved during the read. That is what made a long publish impossible
while the table was being written to. Declined for a pinned read, which cannot
change under itself.

## Consequences

- **A keyed BigQuery read is no longer one session.** It is one session per
  segment, so it issues more Storage Read sessions and a few small query jobs
  to plan, and the boundary query is a window over the key column. The cost is
  to be measured on the real 3.64M-row table; it is not measured here.
- **The time travel window bounds resume.** A publish that stops for longer
  than 48 hours starts over. The table's time travel setting may be as short as
  2 days, so the 48-hour limit sits below it.
- **Resume re-plans at the current time as well.** The key-domain check runs as
  of the current instant, so a NULL key that arrived after the recorded instant
  fails the resume at open, even though the recorded snapshot is clean. The
  publish would also fail on a fresh run, so this is honest, but it is not
  the snapshot the resume reads.
- **A predicated or keyless read keeps the single-session path**, and so still
  ends at six hours. The search publish reads neither, but a future caller
  that does inherits the old limit.
- **DuckDB and other adapters report no progress**, so they keep the
  full-read-on-retry behaviour, which the store's contract already allowed.

## Evidence

Tests in `tests/test_segmented_snapshot.py` pin the segments' row restrictions,
the single shared instant, progress moving only once rows are yielded, a
resume that skips completed segments, the refusal of a point older than 48
hours, and the refusal of a re-plan after reading has started. The two
progress and pinning checks were each broken on purpose and caught.

Not yet measured: a live run against BigQuery. The row-restriction and
`table_modifiers` fields were checked against the Storage Read API reference;
the boundary query's cost and the single-worker sort on a 3.64M-row key have
not been run against the real table.
