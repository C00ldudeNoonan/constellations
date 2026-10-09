# ADR-0025: a declared entity scope filters after retrieval and reports what it excluded

- **Status:** accepted
- **Date:** 2026-10-09
- **Prompted by:** #628

## Context

`search_context` gained `entity_scope`, so an agent can ask for "every
central bank" rather than for the exact entity strings a project happened to
write. The scope names declared things: an entity class, or one vocabulary
term optionally expanded through the declared `broader` hierarchy.

The complication is where the facts live. A declared term reaches a chunk
through `context_entity_links`, a warehouse relation on a different grain from
the search index: one row per context-to-entity relationship, keyed by
`context_id`. The index itself holds the chunk's declared `returned: true`
attributes and nothing about entity links, so the store's prefilter — which is
what `filters:` compiles into — cannot see them.

`limit` is the other constraint. A scope that is applied where a caller cannot
see it turns a thin answer into a lie: "three results" reads as "the corpus
holds three" when it may mean "the scope kept three of forty".

## Decision

The scope is applied **after retrieval**, against the entity links a hit
already carries, and **before the `limit` slice**.

`_search_context` already reads each readable hit's links to populate
`SearchContextResult.entities`. The scope filters that same set, so it adds no
warehouse read and no store round trip. Filtering before the slice means
`limit` counts results that satisfy the scope rather than retrieval hits that
may not.

Matching is `(entity_namespace, entity_key)` against `(vocabulary name, term
label)` — the reading of a link row that `config.vocabulary.declared_terms`
defines and the concept cloud already follows. A row whose namespace names no
declared vocabulary matches nothing, which is what stops a fuzzy or
hand-maintained resolver's rows from satisfying a scope that names a declared
class. The label is compared forward, through `canonical_entity_key`, rather
than by decoding a row's key back into a label: that encoding is ours and its
inverse is not published.

Every scoped response carries `entity_scope_applied`, which states the
requested class or term, what it expanded to, and `results_excluded` — the
readable hits the scope dropped. Each result carries `matched_terms`, naming
the term it actually matched and whether that was the requested term, one
beneath it, or one above it. Expansion is off unless asked for.

The consequence a caller has to be able to see: because the scope filters a
retrieved candidate set, a narrow scope over a broad query can return fewer
results than `limit` while matching documents exist deeper in the corpus.
`results_excluded` is how a caller learns that raising `candidate_limit` may
return more, and the reference documents it in those terms.

## Alternatives considered

### Resolve the scope to `context_id`s in the warehouse, then prefilter the store

The precise answer, and the one that makes `limit` mean what it means on an
unscoped search. Declined: it reintroduces exactly the failure #628's first
half removed. Reading `context_entity_links` for every row matching a class is
an unbounded scan — it refuses past `max_scan_rows`, so a large corpus would
fail the search outright rather than answer it — and the resulting
`context_id IN (...)` prefilter would carry six figures of values into the
store on the SEC corpus. The scan cost is paid on every query, not once.

### Materialize the declared class onto the chunk as a filterable attribute

Then `entity_scope` would compile into the store's prefilter like any other
`filters:` entry, with exact `limit` semantics and no post-filtering. This is
the right long-term shape and is not rejected on the merits — it is rejected
as out of scope here, because it changes what a search model must publish and
therefore forces a re-embed of every corpus that wants class filtering. Worth
revisiting as a declared, opt-in attribute when a project asks for exact
`limit` semantics under a scope.

### Expand the hierarchy by default

Rejected on the issue's own terms: an agent must be able to tell that a result
came from a narrower term, never silently. A default-on expansion makes the
unexpanded search unreachable and gives the reported `matched_terms` the job
of undoing a surprise rather than describing a request.

### Report only that expansion happened, not which term matched

Cheaper, and it satisfies a literal reading of "reported in the response".
Declined: an agent ranking or citing a hit needs to know *which* narrower
term reached it — "this is about the Federal Reserve" and "this is about some
central bank" are different claims about the same result.

### Return an empty result set for an undeclared class or term

Declined: an undeclared class and an empty corpus are different answers, and a
caller that cannot distinguish them will conclude the corpus is empty when it
mistyped a class name. Both are refused with `invalid_request` naming what is
declared. A project with no `vocabularies:` at all is refused with
`capability_unavailable`, because there the capability is absent rather than
the request wrong.
