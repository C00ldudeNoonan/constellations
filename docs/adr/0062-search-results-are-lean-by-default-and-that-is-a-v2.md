# ADR-0062: Lean search results are the default, and that is `mcp_context/v2`

- **Status:** accepted
- **Date:** 2026-11-23
- **Prompted by:** #812 (ALE-108)

## Context

Every `search_context` result inlined `interval`, `freshness` and a
`lineage` block. `lineage.resources` is the same list of upstream resources on
every chunk of a response, and most other fields in the block are null or
identical across results. A read that wants five passages paid for five copies
of provenance it did not ask for. `get_context_lineage` already answers the
per-chunk question.

## Decision

`search_context` takes `detail`, `lean` or `full`, defaulting to `lean`. A lean
result leaves `interval`, `freshness` and `lineage` null; `full` fills them as
before. The response echoes `detail`. Because a default response now omits
fields it used to carry, the contract moves to `mcp_context/v2`.

## Alternatives considered

### Keep `full` as the default and make `lean` opt-in, staying on v1

Additive within v1 by [ADR-0008](0008-mcp-hits-carry-declared-attributes.md)'s
rule, and no client breaks. Rejected because the issue is about not paying for
provenance by default: an opt-in nobody passes saves nothing. ADR-0008 reserves
a bump for a response that stops meaning what it meant, and a default that
drops fields is that.

### Keep v1 and make the three fields nullable

The smallest diff. Rejected because a v1 client reading `lineage` off a result
would crash on a marker that still says v1. The marker is the only place a
client can learn the shape changed.

### Drop the three fields from the result type altogether

Would make `full` impossible, and `lineage` is a governance feature that
provenance-sensitive callers use inline. They get it by asking.

## Consequences

A caller that read those fields passes `detail: "full"`. Transcript capture
keeps recognizing v1 markers, since a transcript records calls already made.
`get_document` and `get_context_lineage` are unchanged apart from the marker.
