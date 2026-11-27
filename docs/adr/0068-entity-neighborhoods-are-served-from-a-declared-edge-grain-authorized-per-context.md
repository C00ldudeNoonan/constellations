# ADR-0068: Entity neighborhoods are served from a declared edge grain, authorized per context

- **Status:** accepted
- **Date:** 2026-11-27
- **Prompted by:** #796 (ALE-424), theme ALE-418; builds on #763 (graph export) and #768 (relation validity intervals)

## Context

An agent could find chunks that mention an entity but could not ask what the
entity is related to without a graph store. The relations exist: `extract_relations`
writes them, `link_entities` resolves their mentions, `supersede_relations`
closes their intervals. They are not served, and serving them has a hazard the
chunk tools do not: a relation names two mentions and the extractor's
`document_id`, while authorization decides on a `context_id` and its registry
row. The two identifiers are the same only when the project configures them to
be (the demo does). A tool that reads the relation table directly would either
serve every edge to every caller or guess which context an edge belongs to.

## Decision

**Edges are a declared, versioned `agent_context/v1` grain, `context_entity_edges`,
and every edge names the context that asserts it.** `get_entity_neighbors` walks
that grain, and serves and traverses an edge only when the caller may read the
registry row and chunk of its context, the test `search_context` applies to a
hit. An edge whose context is missing, ambiguous, mismatched or denied is
neither returned nor walked. The context is an explicit argument of the
projection helper, and `validate_context_entity_edges` fails publication for an
unresolved one.

The walk is bounded by hops (at most 3), an edge cap across all hops with a
`truncated` flag, and the per-read scan limit, which raises instead of returning
a partial graph. As-of reads use the half-open interval `relations_as_of`
uses, pinned against it by a test.

## Alternatives considered

### Read the relation model directly, matching `document_id` to `context_id`

Needs no new model. Ruled out because the match is a convention, not a
contract: a project whose extractor keys on a source document would have its
edges silently attached to nothing, or to the wrong context, and the wrong
caller's grants would decide who reads them. Making the mapping the project's
declared output moves that failure to publication.

### Authorize the endpoints' contexts through `context_entity_links`

Edges would be served when the caller can read a chunk linking either endpoint.
Ruled out because the fact is asserted by one passage, not by the links of its
entities. A caller could read a chunk that mentions Apple and, through it, learn
a relation only a restricted document states.

### Filter only the final response

Walk every edge, then drop the ones the caller may not read. Ruled out because
traversal is itself an answer: a hidden first-hop edge would decide which
second-hop edges appear, revealing it by what is reachable. Only authorized
edges extend the frontier.

### Offer `entity_scope` on the tool

Ruled out for now. A scope would have to say whether it limits the start, the
endpoints or every intermediate entity, and reporting less than that would let
an excluded entity shape the answer invisibly (ADR-0025). `term` and
`relation_types` narrow a walk instead.

## Consequences

- A project publishes one more model to use the tool, and writes the
  `context_id` mapping once. The demo shows it in about forty lines.
- Authorizing per edge reads each asserting context's chunk and registry rows,
  so a wide neighborhood costs reads proportional to its distinct contexts,
  bounded by `max_scan_rows`.
- An unauthorized or unknown start entity returns an empty answer, which a
  caller cannot tell from an entity with no relations. That is deliberate: a
  distinct answer would confirm the entity exists.
- Belief time is not served. `observed_at` and `recorded_*` do not give
  bitemporal belief semantics, and the tool says so.
