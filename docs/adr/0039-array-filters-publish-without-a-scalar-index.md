# Array filters publish without a scalar index

- **Status:** accepted
- **Date:** 2026-10-28
- **Prompted by:** constellations#722, ALE-432

## Context

Filterable attributes are selected for BTree creation during search publication.
That includes `array[string]` user and policy attributes, but LanceDB rejects
a BTree on a list column. A valid project therefore fails after writing its
rows, then retries the same deterministic failure twice. Array membership
predicates already work without an index.

## Decision

Exclude array attributes from scalar-index selection. Keep their membership
predicates and publication shape unchanged. Reject `sortable: true` for an
array attribute during configuration validation. Native index argument and
schema failures expressed as `ValueError` or `TypeError` fail on their first
attempt and pass through the existing sanitized error boundary.

## Alternatives considered

### Build a LabelList index

LanceDB 0.34.0 exposes `LabelList` for list containment. Adopting it would
require index-kind selection, inspection and incremental lifecycle coverage
alongside BTree, FTS and vector indexes. This incident establishes a
correctness failure, not a measured need for that optimization. Leave arrays
unindexed until measurements justify that additional lifecycle contract.

### Refuse array filters during compilation

This would avoid the late failure, but remove working CLI and policy
membership filtering even though the predicate itself needs no index.

## Consequences

Array filtering scans values rather than using a list index, so large
collections may need a later performance improvement. Scalar attributes and
the merge key retain their BTree indexes. No artifact, collection descriptor
or fingerprint change is needed; existing rows remain reusable. The separate
MCP operator-contract mismatch is tracked in constellations#723 / ALE-433.

## Evidence

On 2026-10-28, the SQL-backed regression in `tests/test_search.py` reproduced
`lancedb_index_failed` on the array's BTree after three `ValueError` attempts
with LanceDB 0.34.0. It passes publication, membership filtering and a rerun
when arrays are excluded. The installed SDK's `lancedb/index.py` declares
`LabelList` separately from `BTree`; `tests/test_array_policy_filters.py`
also exercises unindexed array overlap in the real engine.
