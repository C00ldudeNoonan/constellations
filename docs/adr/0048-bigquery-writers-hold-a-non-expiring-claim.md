# ADR-0048: BigQuery writers hold a non-expiring claim

- **Status** Accepted. Amends 0047. Recovery amended by [0058](0058-bigquery-writer-recovery-requires-quiescence.md).
- **Date:** 2026-11-03
- **Prompted by:** #741, PR #742 (ALE-515)

## Context

ADR-0047's snapshot-bound source deletion approval is retained. Its proposed
BigQuery transaction cannot, by itself, preserve the reviewed baseline against
concurrent insert-only MERGEs and append loads. BigQuery explicitly permits
those operations alongside a mutating transaction. See Google's
[transaction concurrency documentation](https://docs.cloud.google.com/bigquery/docs/transactions#transaction_concurrency).
The guard needs exclusion from before discovery through the final publication,
including first creation, unguarded writers, new keys and state writes.

## Decision

Every BigQuery run/build holds one cooperative claim for its physical data
project and dataset, acquired before discovery. Independently called model
publication and state mutation APIs participate too. The table-creation API
creates `stel_writer_claim` with a random owner label and `exists_ok=False`;
only the creator proceeds. The dataset's default expiration is removed before
any work is authorized. Profile names, billing projects and credentials are
not lock identities. Read-only source discovery does not initialize state.

Claims never expire, get stolen, or get cleared by a staging sweep. While a
claim is held, every query/load submission rechecks ownership, records its job
for completion checks, and carries the owner label. Transparent query-job
retries are disabled. Confirmed completions are forgotten to bound tracking
memory. An ambiguous submission permanently prevents further submissions by
that holder. Release requires unchanged ownership and terminal status for all
outstanding jobs. A failure to prove either strands the claim and poisons that
adapter, even if a warehouse job eventually succeeds.
Table metadata mutations use the same ambiguity counter: a timed-out delete
could otherwise land after a replacement writer published. Recovery must use
audit records to account for these non-job API requests too, or remain blocked.

No participating actor can revoke a holder, so authority cannot move while a
delayed job is running. This avoids depending on a per-statement SQL fence
which append loads cannot enforce. Manual recovery is outside normal execution:
stop the old process, account for submissions, confirm all jobs terminal in all
execution projects, inspect target/state consistency, and only then delete the
claim. There is no force-unlock or timer-based recovery API.

This is cooperative exclusion, not warehouse authorization. All writers must
upgrade together. IAM and operator controls must exclude older stel versions,
direct clients and independently executed trusted raw SQL from model targets
and their state. Arbitrary SQL APIs are not a sandbox or a SQL write parser;
the caller must enter `writer_claim()` before independent raw SQL mutations.
Serving-ledger reader coordination is not a model-target publication and does
not acquire the writer claim. Read-only consumers remain available.

## Alternatives considered

- **Rely on mutating transactions:** rejected because insert-only and append
  writes do not contend on those locks. Snapshot assertions still catch
  pre-existing disagreement but cannot see later appends.
- **Expiring leases with per-write fences:** rejected for this change because
  native load jobs cannot atomically enforce a SQL fence. A job may outlive its
  process and lease; giving authority to another writer then is unsafe.
- **A per-target claim:** narrower concurrency, but state and schema setup are
  shared and full runs can publish multiple related targets. A dataset claim
  gives a single canonical boundary across all selected models and aliases.
- **A local lock:** rejected because separate hosts and containers would not
  share it. The warehouse must arbitrate first acquisition.
- **An operator assertion that no other writer exists:** documents a
  precondition without enforcing it against current stel runs.

## Consequences

BigQuery model writers in one dataset run serially, including projects without
the deletion guard. Separate datasets remain independent. An interrupted or
ambiguous run can require manual intervention; convenience must not silently
revoke a process that may still submit work. The target/state comparison and
atomic deletion from ADR-0047 remain necessary: writer exclusion does not
repair inconsistent history or prove source completeness.

Deterministic tests cover competing creation/upsert/append/state/delete paths,
profile routing aliases, lost acquisition/submission responses, pending jobs,
owner changes, recovery and runner ordering. A credential-gated live test
exercises actual table-creation arbitration, removal of default expiration,
competing first publication and append, and a lost real load-job response
followed by terminal-job confirmation and synthetic manual recovery.
