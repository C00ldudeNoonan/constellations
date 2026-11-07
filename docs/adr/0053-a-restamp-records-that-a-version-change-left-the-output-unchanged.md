# ADR-0053: a restamp records that a code_version change left the output unchanged

- **Status:** accepted
- **Date:** 2026-11-06
- **Prompted by:** #750 (ALE-577), following #572 and ADR-0050

## Context

A release can move a model's `code_version` without changing a row it
publishes. ADR-0050 narrows how often that happens, but not to never: any
release that edits code a backend reaches, or a transform's versioned
implementation, still re-keys every published row on first run. The only ways
through were `--accept-reprocess` and `--reprocess-all`, which authorize the
rewrite rather than avoid it (about 12 hours for a 20,000-row registry), and
`stel plan` said only "code_version differs for N of N published rows". An
operator could not tell a false positive from a real change without recomputing
the hash payload by hand.

`stel_state` keeps one digest per row. The inputs behind it are not stored
anywhere, so no later release can say what moved.

## Decision

**Record the inputs.** After a clean run, stel writes the identity fields behind
the `code_version` it published under to `stel_code_identity`, keyed by scope and
version. A field is a top-level entry of the hashed payload, one level deep
(`extraction.backend_implementation`, `extraction.options`, `transform_code_hash`),
each reduced to a 64-bit digest. Digests, not values: prompts and options can be
sensitive and `stel plan` writes `plan.json`. `stel plan` diffs the current
fields against the record for the stale version most rows carry and names what
moved. A scope published before the record existed gets "not recorded" rather
than an empty list, because an empty list would read as "nothing moved".

**Restamp with `stel adopt --select`.** It rewrites `code_version` in
`stel_state` for the selected models' scopes to the current value and appends
what it did to `stel_adoption_log`: who, when, the old and new versions, and the
row count. It reads no source, calls no provider and touches no model table.
`--select` is required and `--dry-run` reports without writing.

**Gate releases.** `tests/fixtures/release_identity.json` holds the identity
fields of each built-in backend and transform as the previous release computed
them. The suite fails when the candidate's differ without an `intended_changes`
entry, and when an entry names a change that did not happen. A seeded change to
the JSON backend's source proves it can fail.

## Alternatives

**A per-row components column in `stel_state`.** Exact per row, but it widens the
largest table in the warehouse by a JSON blob per row and migrates both
adapters' state schema. The side table holds one row per scope and version.

**Recompute the old payload from the previous release.** Impossible: the old
source is not installed, and the point is that the operator has already upgraded.

**Make `--accept-reprocess` cheap by reusing outputs.** Reuse exists for embed
vectors (#665) but not for an extraction registry, where the rewrite is the
cost. It also leaves the false positive in place.

**Compare against a published wheel in CI.** Truer to "the previous release",
but it needs network and PyPI in the suite, and breaks offline. A committed
baseline, re-taken at each release, tests the same thing deterministically.

## Consequences

- Records are written from the first run after upgrading to a release that has
  this; the upgrade that introduces it still cannot name what moved, and says so.
- `adopt` leaves the model table alone, so an output column that mirrors
  `code_version` (a registry watched by `update_when_changed`) keeps the old
  value until that row is next rewritten for another reason.
- Adopting is a claim by the operator that output is unchanged. stel does not
  check it. The log is how it is audited.
- Search-scope models record and adopt against the serving descriptor scope,
  like `stel plan`.
- The baseline is a file a release must re-take (`docs/release.md`).
