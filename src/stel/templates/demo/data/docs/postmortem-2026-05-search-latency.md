---
title: "Postmortem: search latency, 2026-05-20"
kind: postmortem
team: discovery
published_at: "2026-05-22"
---

# Postmortem: search latency, 2026-05-20

Product search stayed up, but for three hours the p99 latency of the Search
Service rose from 180 ms to 4.2 seconds. Storefront pages that render search
results timed out for customers on slow connections.

## Impact

From 09:15 to 12:20 UTC, about 12% of search requests took longer than two
seconds. Conversion on search-driven sessions fell by a fifth during the
window. No data was lost and no request returned a wrong result; requests
were only slow.

## Root cause

A scheduled reindex rebuilt the catalog index on the same nodes that were
serving queries. The rebuild normally runs at 02:00, but a timezone change
in the scheduler moved it into European business hours. While the rebuild
ran, query threads competed with it for memory, and the Redis cache in front
of the Search Service was evicting entries faster than it could refill them,
so more queries reached the index cold.

## Detection

Latency alerts fired at 09:22. The on-call engineer first suspected the
cache cluster and spent 40 minutes scaling Redis, which helped a little but
did not address the rebuild.

## Resolution

We paused the reindex at 11:55 and latency recovered within 25 minutes.
Reindexing now runs on a separate set of nodes and swaps the finished index
in atomically, so a rebuild can no longer compete with live queries.

## Follow-ups

- Pin scheduler jobs to UTC and alert when a job's start time moves.
- Add the reindex status to the Search Service dashboard, so it is the first
  thing on-call sees.
- Document the cache-versus-index triage order in the search runbook.
