---
title: "Release notes: June 2026"
kind: release_note
team: product
published_at: "2026-06-30"
---

# Release notes: June 2026

## Faster search

Search results now render in under 200 ms for 99% of queries. The Search
Service rebuilds its index on dedicated nodes and swaps it in atomically, so
the nightly refresh no longer slows anything down.

## Saved carts

Signed-in customers can now save a cart and come back to it from any
device. Saved carts are held for 30 days.

## Sign-in with passkeys

The Auth Gateway now accepts passkeys alongside passwords. Customers can add
a passkey from their account settings; existing passwords keep working.
