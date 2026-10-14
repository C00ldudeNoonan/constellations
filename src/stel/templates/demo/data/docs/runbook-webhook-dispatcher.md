---
title: "Runbook: Webhook Dispatcher"
kind: runbook
team: integrations
published_at: "2026-07-10"
---

# Runbook: Webhook Dispatcher

The Webhook Dispatcher reads order, refund, and shipment events from Kafka
and delivers them to customer endpoints, retrying failures with backoff.

## Webhooks are late

1. Check consumer lag on the order-events and refund-events topics. Lag over
   15 minutes is customer-visible.
2. Find the slowest endpoints. One slow endpoint should now be moved to the
   retry queue automatically; if it is not, move it by hand.
3. Check that the dispatcher is consuming at all. A stuck consumer group
   shows lag rising with zero throughput.

## A customer says they never got an event

Look the event up by order id in the delivery log before replaying anything.
Most "missing" events were delivered and rejected by the customer's
endpoint. Replay only events with no successful delivery, so the customer
does not receive duplicates.

## Who owns it

The integrations team owns the dispatcher. Kafka itself belongs to the
data platform team; page them if the brokers, not the consumers, are
unhealthy.
