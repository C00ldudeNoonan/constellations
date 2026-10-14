---
title: "Release notes: August 2026"
kind: release_note
team: product
published_at: "2026-08-29"
---

# Release notes: August 2026

## Webhook delivery you can see

Every webhook now has a delivery history in the dashboard: when it was
sent, how long the endpoint took, and what it answered. The Webhook
Dispatcher moves a slow endpoint to its own retry queue, so one slow
integration can no longer delay anyone else's events.

## Refunds in one step

Partial refunds no longer need a support ticket. The Payments API accepts a
refund for any subset of an order's lines, and the refund event is
delivered to your webhook within a minute.

## Session limits

The Auth Gateway signs out sessions idle for more than 30 days. Customers
using passkeys sign back in with one tap.
