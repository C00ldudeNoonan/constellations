# Architecture decision records

Numbered, append-only records of decisions that had a real alternative.
`docs/architecture/` holds the larger accepted designs; these hold the
narrower "why not the other thing" calls that otherwise live only in commit
history (issue #311).

## Index

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-degraded-serving-and-fail-closed-recovery.md) | A failed publish keeps serving its live generation; recovery fails closed | accepted; amended by 0003, 0016 |
| [0002](0002-vector-search-mode-is-an-index-build.md) | Switching `exact` <-> `approximate` is an index build, not a whole-index invalidation | accepted; amended by 0003 |
| [0003](0003-reader-safe-online-publication.md) | Online changes use private generations, append writes, and reader-aware retirement | accepted; amended by 0004, 0016 |
| [0004](0004-seed-private-generation-from-store.md) | An index-only change fills its private generation from the store, not the warehouse | accepted |
| [0005](0005-re-entry-unit-is-the-existing-checkpoint.md) | Re-entry resumes from each step's existing checkpoint; no phase ledger, no activate command | accepted; amended by 0017 |
| [0006](0006-saas-context-is-landed-then-rendered.md) | SaaS context is landed by an EL tool and rendered by stel; no first-party connectors | accepted |
| [0007](0007-native-drive-files-carry-a-change-token.md) | Native Drive files carry a change token named as such, never a fake content hash | accepted |
| [0008](0008-reprocess-guard-defaults-to-fail.md) | Paid models refuse an unannounced reprocess by default; the guard reads the plan, not per-stage state | accepted |
| [0008](0008-mcp-hits-carry-declared-attributes.md) | A returnable attribute is additive within `mcp_context/v1`, not a v2 | accepted |
| [0009](0009-serving-holds-the-warehouse-when-the-adapter-allows.md) | The serving session holds its warehouse connection only when the adapter says one may outlive a request | accepted |
| [0010](0010-warehouse-identity-is-a-granted-attribute.md) | The warehouse identity a governed read runs as is a granted attribute, resolved per subject; missing or ambiguous is a refusal | accepted |
| [0011](0011-an-entitlement-interval-is-one-row-and-one-attribute.md) | An entitlement interval is one row, one attribute, and never a one-sided bound | accepted |
| [0012](0012-native-failure-detail-goes-to-an-operator-named-file.md) | The native detail behind a sanitized failure goes to a file the operator named, never to a log level | accepted; amended by [0014](0014-a-debug-switch-owns-its-destination.md), [0020](0020-a-native-panics-stderr-write-is-documented-not-intercepted.md) |
| [0013](0013-a-merge-page-is-bounded-by-bytes-in-the-store.md) | A merge page is bounded by bytes, in the store, before it is sent — `batch_size` keeps counting rows | accepted |
| [0014](0014-a-debug-switch-owns-its-destination.md) | A debug switch owns its own destination, and marks the records it is for | accepted |
| [0015](0015-a-sync-watermark-needs-two-independent-signals.md) | A sync watermark needs two independent signals — cheap state, and an authoritative content fingerprint — not one | accepted |
| [0016](0016-an-interrupted-in-place-write-keeps-its-generation-when-the-store-says-so.md) | A failed in-place publish keeps its generation when the store promises interrupted writes leave it sound; readers stay excluded by status | accepted |
| [0017](0017-an-operator-may-activate-a-complete-generation-from-its-state.md) | An operator may activate a physically complete generation from its recorded state, re-stamped, without re-reading the corpus | accepted |
| [0018](0018-a-resume-drops-its-indices-and-rebuilds-once.md) | A resumed generation drops its indices and rebuilds them once, as a fresh build already does | accepted |
| [0019](0019-recovery-skips-confirmation-only-for-a-provably-dead-local-owner.md) | The publish claim records its holder and a heartbeat; recovery skips the confirmation only for a provably dead owner on this host, and never acts on heartbeat age | accepted |
| [0020](0020-a-native-panics-stderr-write-is-documented-not-intercepted.md) | A native panic's stderr write is outside the sanitizer and is documented, not intercepted; the store still sanitizes what Python sees | accepted; amends 0012 |
| [0021](0021-an-index-behind-on-rows-is-extended-not-retrained.md) | An index behind on rows is extended over the new rows; a rebuild is for an index that has to change shape | accepted |
| [0022](0022-a-schema-probe-is-never-cached-and-a-missed-column-fails-the-run.md) | A schema probe is never answered from a cache, and a column it missed fails the run instead of being dropped | accepted |
| [0023](0023-a-keyed-warehouse-read-is-segmented-and-pinned-to-one-instant.md) | A keyed warehouse read is segmented and pinned to one instant, and resumes from its last completed segment | accepted |
| [0024](0024-a-forced-reprocess-ignores-incremental-state-rather-than-clearing-it.md) | A forced reprocess reads incremental state and declines to skip on it, rather than clearing it | accepted; amends 0022 |
| [0025](0025-a-declared-entity-scope-filters-after-retrieval-and-reports-what-it-excluded.md) | A declared entity scope filters after retrieval and reports what it excluded | accepted |
| [0026](0026-embedding-reuse-is-keyed-by-content-and-fetched-by-id.md) | Embedding reuse is keyed by content and fetched by id | accepted |
| [0028](0028-the-serving-catalog-reads-the-declaration-the-artifact-was-compiled-with.md) | The serving catalog reads the declaration the artifact was compiled with, not the live project file; the manifest carries it and moves to v3 | accepted; resolves 0025's known limitation |

## When to write one

The test: **would a competent contributor plausibly try the alternative we
rejected?** If yes, the reasoning needs to outlive the pull request that
established it. If the choice was obvious, or had no live alternative, it does
not need an ADR.

An ADR should take fifteen minutes. It is not a design document and not a
substitute for one.

## Conventions

- **Numbered and immutable.** Superseding an ADR means writing the next one
  and marking the old one superseded — never editing a decision in place. How
  the thinking changed is the record's whole value.
- **Record negative results, with evidence.** "We measured X and it ruled out
  Y" is the highest-value content and the first thing lost. Cite the
  measurement, the dependency source read, or the live reproduction, and say
  when — so nobody re-derives it, and nobody assumes it still holds after the
  underlying thing moves.
- **Link from the issue that prompted it.** Issues stay the work tracker; the
  ADR is the durable record. Add the ADR path to the issue or PR that made the
  call.
- **Update the index above** when adding one.

Start from [`0000-template.md`](0000-template.md).

## What does not belong here

- Decisions with no rejected alternative — those are just the code.
- Large accepted designs, which stay in `docs/architecture/`.
- A running backfill of everything already shipped. #311 scoped that out
  deliberately: write them going forward, and backfill only when a decision
  resurfaces and its reasoning turns out to be unwritten.
