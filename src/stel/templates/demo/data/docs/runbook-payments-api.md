---
title: "Runbook: Payments API"
kind: runbook
team: payments
published_at: "2026-03-20"
---

# Runbook: Payments API

The Payments API takes a cart, authorizes the card with the processor, and
writes the order. It is the only service that writes to the orders tables
in Postgres, and it publishes an order event to Kafka once the write
commits.

## Checkout errors are rising

1. Check the connection-pool panel first. If PgBouncer reports waiting
   clients, the primary database is saturated; see the next section.
2. Check the processor's status page. A processor outage shows up as
   authorization timeouts, not 503s.
3. If neither, look at the most recent deploy. Most checkout incidents
   follow a deploy within the hour.

## The database is saturated

Do not add Payments API pods: every pod adds connections. Lower the worker
count instead, or roll back the deploy that raised it. The March outage was
exactly this, and scaling up would have made it worse.

## Rolling back

Roll back with the deploy tool rather than by editing the deployment, so
the pipeline records the rollback. A rollback takes about four minutes to
reach every pod.

## Who owns it

The payments team owns the service. Page them through the payments rotation;
for anything touching the orders schema, also tell the data platform team.
