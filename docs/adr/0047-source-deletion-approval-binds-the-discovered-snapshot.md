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
