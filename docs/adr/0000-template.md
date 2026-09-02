# ADR-0000: <the decision, as a short noun phrase>

- **Status:** proposed | accepted | superseded by [ADR-nnnn](nnnn-slug.md)
- **Date:** YYYY-MM-DD
- **Prompted by:** #nnn (issue or PR that forced the decision)

## Context

The constraint that forced a choice. What was true, what broke, or what could
not be built without deciding. Enough that someone who was not here can tell
whether the constraint still holds — if it stops holding, this ADR is the
thing to revisit.

## Decision

One paragraph, imperative. What we do now.

## Alternatives considered

The section that pays for the practice. One sub-heading per option that was
genuinely on the table, each ending in the specific reason it lost. "We did
not think of it" is not an alternative; "we tried it and it cost X" is.

### <alternative>

Why it looked right, and the specific thing that ruled it out.

## Consequences

What this makes harder, not only what it makes possible. Name the sharp edges
a future contributor will hit, and anything now load-bearing that does not
look it.

## Evidence

Only when the decision rests on something checkable, and then always: a
measurement with its number, a dependency's source read at a version, a live
reproduction. Say where it came from and when, so nobody re-derives it — and
so nobody assumes it still holds when the underlying thing may have moved.
