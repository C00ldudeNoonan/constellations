---
title: "Postmortem: delayed webhooks, 2026-07-02"
kind: postmortem
team: integrations
published_at: "2026-07-06"
---

# Postmortem: delayed webhooks, 2026-07-02

Order and refund webhooks reached customers up to six hours late. Nothing
was dropped, but several customers' fulfilment systems shipped orders that
had already been refunded, because the refund event arrived after the
warehouse picked the order.

## Impact

Between 03:40 and 09:50 UTC, the Webhook Dispatcher delivered events with a
median delay of 2.5 hours and a maximum of 6 hours. 41 customer accounts
received at least one event more than an hour late. Eleven reported a
shipped-then-refunded order.

## Root cause

One customer's endpoint began responding in 29 seconds instead of failing
fast. The dispatcher consumes from a single Kafka partition per event type,
and it retries a slow endpoint in line rather than moving it aside, so one
slow consumer held up every event behind it. Consumer lag on the
order-events topic grew to 1.9 million messages.

## Detection

Nothing paged. The Kafka consumer-lag alert existed but was routed to an
email list nobody on the integrations team reads. A customer opened a
support ticket at 08:12, and the on-call engineer was paged from that.

## Resolution

We disabled the slow endpoint at 08:40 and the backlog drained by 09:50.
The dispatcher now moves an endpoint that exceeds its latency budget to a
separate retry queue, so one customer can no longer delay another.

## Follow-ups

- Route consumer-lag alerts to the integrations pager.
- Partition the order-events topic by customer, not by event type.
- Publish a status-page entry automatically when lag passes 15 minutes.
