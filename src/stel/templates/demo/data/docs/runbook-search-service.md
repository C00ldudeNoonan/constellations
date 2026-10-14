---
title: "Runbook: Search Service"
kind: runbook
team: discovery
published_at: "2026-05-28"
---

# Runbook: Search Service

The Search Service answers product search and autocomplete. Queries hit a
Redis cache first, then the catalog index. The index is rebuilt nightly
from the catalog database.

## Search is slow

Triage in this order, because it is the order that cost us time in May:

1. Is a reindex running? Check the reindex status panel. A rebuild on
   serving nodes is the most common cause of slow search.
2. Is the cache hit rate falling? A hit rate under 60% means queries are
   reaching the index cold. Scaling the cache cluster helps only once the
   cause of the evictions is gone.
3. Is one query pattern dominating? Autocomplete storms from a broken client
   look like general slowness.

## Search returns stale results

The nightly reindex failed or has not swapped in. Check the last successful
swap time; anything older than 30 hours is stale. Rerun the reindex on the
rebuild nodes, never on the serving nodes.

## Who owns it

The discovery team owns the service and the reindex job.
