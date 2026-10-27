# ADR-0038: Embedding publication overlap is opt-in

- **Status:** accepted
- **Date:** 2026-10-27
- **Prompted by:** #432 (ALE-300)

## Context

An embed run computes a flush window and then publishes it before computing
the next. After DuckDB's per-row state write was fixed, a 400-document run
fell from 48.93 to 6.76 seconds. A later BigQuery dev full refresh published
80,000 rows before a provider failure; its successful writes accounted for
303.252 seconds during 1,270.328 seconds of model wall time. That partial run
does not measure a complete corpus, but publication is still material there.

## Decision

Offer `embed.overlap_publication: true` as an execution-only option. Prepare
the next window, including warehouse reuse reads, before launching a single
worker to publish the prior window. Compute the next window while that worker
publishes. Wait for the publish result before advancing progress or submitting
another publish. Keep `FlushPublisher` responsible for clear, write, then
state ordering. Leave the default serial and leave chunk publication serial.

## Alternatives considered

### Always overlap

BigQuery can save part of its publication time, but DuckDB's measured write
cost became small after the state fix. Overlap holds two windows and can pay
for one later window before a preceding publication fails. Making that the
default would impose a memory and spend tradeoff on runs with no measured
benefit.

### Publish while reading or querying reuse for the next window

This would overlap more work, but it calls one adapter from two threads at
once. The adapter's session and publication contracts do not promise that.
Preparing the next window first keeps adapter operations serial while still
overlapping the provider with the write.

### Overlap chunk publication too

The measured expensive stage was embed, and its provider wait offers an
independent activity to overlap. There is no corresponding chunk timing to
justify a second path or its extra memory. Chunk stays serial until measured.

## Consequences

Peak embed work can reach two `flush_every` windows. If publication fails,
the next window may have completed provider calls but has no state advancement;
the response cache can retain its vectors for retry when configured. A retry
without that cache may pay again for at most that one window. Recent-window
reuse supplies duplicate text from the unpublished window before the target
reuse query, preserving deduplication across the overlap boundary.

## Evidence

- [DuckDB before/after measurement](https://github.com/C00ldudeNoonan/constellations/issues/432#issuecomment-5577445242), 2026-09-09.
- [BigQuery partial-run attribution](https://github.com/C00ldudeNoonan/constellations/issues/432#issuecomment-6391193698),
  October 2026. The run failed after 80,000 published rows, so the 303.252
  seconds of publication are evidence of successful writes, not a full-run
  benchmark.
