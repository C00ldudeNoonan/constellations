# Architecture decision records

Numbered, append-only records of decisions that had a real alternative.
`docs/architecture/` holds the larger accepted designs; these hold the
narrower "why not the other thing" calls that otherwise live only in commit
history (issue #311).

## Index

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-degraded-serving-and-fail-closed-recovery.md) | A failed publish keeps serving its live generation; recovery fails closed | accepted; amended by 0003 |
| [0002](0002-vector-search-mode-is-an-index-build.md) | Switching `exact` <-> `approximate` is an index build, not a whole-index invalidation | accepted; amended by 0003 |
| [0003](0003-reader-safe-online-publication.md) | Online changes use private generations, append writes, and reader-aware retirement | accepted; amended by 0004 |
| [0004](0004-seed-private-generation-from-store.md) | An index-only change fills its private generation from the store, not the warehouse | accepted |
| [0005](0005-re-entry-unit-is-the-existing-checkpoint.md) | Re-entry resumes from each step's existing checkpoint; no phase ledger, no activate command | accepted |
| [0006](0006-saas-context-is-landed-then-rendered.md) | SaaS context is landed by an EL tool and rendered by stel; no first-party connectors | accepted |
| [0007](0007-native-drive-files-carry-a-change-token.md) | Native Drive files carry a change token named as such, never a fake content hash | accepted |

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
