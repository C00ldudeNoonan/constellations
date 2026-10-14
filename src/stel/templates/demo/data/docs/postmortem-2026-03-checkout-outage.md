---
title: "Postmortem: checkout outage, 2026-03-11"
kind: postmortem
team: payments
published_at: "2026-03-14"
---

# Postmortem: checkout outage, 2026-03-11

For 47 minutes, roughly one in three checkout requests failed with a 503.
The Payments API could not get a connection to Postgres, so every call that
needed to write an order timed out after ten seconds and was retried by the
storefront, which made the backlog worse.

## Impact

Between 14:02 and 14:49 UTC the Payments API returned 503 for 31% of
checkout requests. Around 2,400 customers saw a failed payment; most retried
successfully within the hour. No payment was charged twice. The Webhook
Dispatcher kept delivering order events for the requests that did succeed,
so downstream systems were never inconsistent, only behind.

## Root cause

A deploy at 13:58 raised the Payments API's worker count from 24 to 64 per
pod without changing the connection pool size. Each worker opened its own
pool of ten connections, so the fleet tried to hold 3,840 connections
against a primary database configured for 2,000. Postgres began refusing
new connections, and the workers that were refused held their requests
open while they waited.

## Detection

The first page fired at 14:06 on the checkout error-rate alert. The
connection-saturation alert on the primary database had been silenced during
a maintenance window two weeks earlier and never re-enabled, which cost us
about four minutes.

## Resolution

We rolled the deploy back at 14:31. Error rates recovered by 14:49, once
the retried requests had drained. A connection pooler now sits between the
Payments API and Postgres, so worker count and connection count are no
longer the same number.

## Follow-ups

- Put PgBouncer in front of the primary database for every service, not
  only the Payments API.
- Alert on silenced alerts older than 48 hours.
- Make the deploy pipeline reject a worker-count change that would exceed the
  database's connection limit.
