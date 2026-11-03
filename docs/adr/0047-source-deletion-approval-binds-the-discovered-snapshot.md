# ADR-0047: source deletion approval binds the discovered snapshot

- **Status:** proposed
- **Date:** 2026-11-02
- **Prompted by:** #741

## Context

An incremental extraction removes documents missing from source discovery. An
incomplete listing is indistinguishable from intentional deletion. A separate
preflight comparison races with the listing used by the actual run. Filtering
the run suppresses removals, including downstream reconciliation, so it cannot
replace a complete unfiltered build when obsolete content must disappear.

## Decision

An incremental extraction can opt into a guard. When its own discovery finds
previously published documents missing, it refuses before deleting or
extracting. An operator independently reviews the removals and approves the
reported fingerprint on a subsequent unfiltered run. The fingerprint covers
the project, target, model, entire prior state, and discovered documents and
their content hashes. A count threshold or a generic yes flag would authorize a
different deletion set; a snapshot fingerprint fails closed when it changes.

The adapter verifies the complete state and target baseline and deletes both
within one transaction. A mismatch or an adapter without the capability fails
closed. Full refresh is refused for guarded models. Filtered runs remain
additive; they do not imply that removals reached downstream models.

## Consequences

The operator needs access to independently compare the listing and published
state; the error does not expose document identifiers. Approval is an explicit
operator action, not authentication. This is opt-in to preserve existing
projects' behavior. The guard cannot establish that the listing is complete,
and later pipeline failures still require operational recovery. Baseline
comparison and transactional deletion add warehouse reads and mutation cost.

## Open concurrency blocker

The proposed BigQuery transaction alone does not enforce the reviewed baseline
against concurrent writers: insert-only MERGEs and append loads can run while
it mutates the target and state, and its snapshot does not see those writes.
This must be fixed before the guard is released. A viable cooperative design
needs one canonical claim for each physical extraction target, acquired before
discovery and held across deletion and every publication. Each target and
state mutation must enforce the same fence atomically; first creation and
append paths cannot bypass it. Recovery must account for warehouse jobs that
outlive the process rather than stealing a claim on a timer.

Such a claim controls participating writers only. Older stel versions, direct
warehouse writes, and other clients can bypass it unless warehouse permissions
enforce the writer boundary. The operator-facing contract must state this
limit, and concurrency tests must cover new keys, existing keys, first creation,
recovery and delayed jobs before the issue can be closed.
