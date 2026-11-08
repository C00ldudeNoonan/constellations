# ADR-0057: An ontology import is an allow-list resolved into the declaration at load

- **Status:** accepted
- **Date:** 2026-11-08
- **Prompted by:** #767 (ALE-419)

## Context

`vocabularies:`, `classes:` and `relations:` were written by hand, so a
standard ontology such as FIBO had to be retyped, lost its IRIs and drifted
from upstream. `ontologies:` reads a Turtle or RDF/XML file instead. Four
choices had a real alternative.

## Decision

**Resolve at load, into the declaration that already exists.** The loader
parses the file and merges what it finds into `vocabularies:`, `classes:` and
`relations:`, the way a CSV vocabulary seed resolves (#718). Every consumer
(enum fields, relation contracts, the concept cloud, the MCP catalog) sees an
ordinary declaration and needs no ontology awareness. The rejected alternative
was to keep the parsed graph as its own object, read by consumers that want
it: two sources of truth for one class, and a reasoner nobody asked for. The
only shape change is an optional `iri` on a term, a class and a relation
pairing, which a dump omits when it is unset; the serving manifest moves to
v5 because a v4 reader would refuse the new key.

**The allow-list is required, and its order is the order terms take.** FIBO is
thousands of classes; importing all of it is never what a project wants. RDF
has no statement order, so file order cannot be identity, and sorting by label
would make an enum's order something nobody wrote. The allow-list is authored,
ordered and reviewable, so a re-serialized file never moves a `code_version`,
and a reordered allow-list does, exactly as reordering inline `terms:` does.

**Hierarchy is the allow-list's induced subgraph.** A `skos:broader` or
`rdfs:subClassOf` link to an IRI outside the allow-list is not followed. The
alternatives were to pull ancestors in automatically, which makes the import
larger than what was listed, or to refuse, which makes every FIBO subset list
the whole chain to `owl:Thing`. A link inside the allow-list is kept.

**Ambiguity fails rather than picks.** An IRI typed as two kinds (a class and
a concept), a class with two allow-listed parents, a term with two broader
terms or two labels in the configured language all fail the load, naming the
IRIs. stel's classes and terms have one parent and one label; choosing one
silently would publish a hierarchy nobody declared.

A relative IRI in the file resolves against `https://stel.invalid/<path>`,
not the file's location, so this machine's path never reaches an identity.
`owl:imports` is not followed and external XML entities are not resolved: a
parse reads the one validated file.
