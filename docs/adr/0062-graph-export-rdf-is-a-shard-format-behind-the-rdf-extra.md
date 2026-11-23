# ADR-0062: Graph export RDF is a shard format behind the `rdf` extra, with `urn:` identifiers

- **Status:** accepted; amends 0055, 0061
- **Date:** 2026-11-23
- **Prompted by:** #811 (ALE-602)

## Context

ADR-0055 deferred JSON-LD and Turtle: `stel graph-export` wrote JSON Lines
only, and a triple store wants RDF. ADR-0061 then made the export incremental
by per-document shards with a manifest fingerprint, so the formats had to fit
that layout. The `rdf` extra (`rdflib>=7.1`) already exists, added for ontology
import (ADR-0057), and ontology import does not yet give an exported node an
IRI.

## Decision

**A format is how a shard's bytes are written, nothing else.** `--format
jsonl|jsonld|turtle` (default `jsonl`) changes the serializer and the shard
extension (`.jsonl`, `.jsonld`, `.ttl`). The graph is built once, by the same
`build_document`, so the three formats are one graph; the manifest records
`format`, and a manifest without it is a JSON Lines export. Fingerprinting,
staging, stale-shard removal and the bounded read are unchanged: a document is
serialized one at a time, so residency stays one document.

**Every RDF shard is a standalone file.** It carries its own prefixes or
`@context`, and RDF triples are a set, so an entity linked in several documents
is the same triples in each and merges when the directory is loaded. No file
needs another to parse.

**A directory keeps one format.** A manifest of another format is refused,
untouched, with the same "export elsewhere or delete" advice as an unreadable
one.

**What identifies a node is a URN built from what the export already has.** An
entity is `urn:stel:entity:<entity_id>` (the agent-context id the JSON Lines
nodes are keyed by), a relation `urn:stel:relation:<relation_id>`, a document
`urn:stel:document:<document_id>`, each percent-encoded; terms are
`urn:stel:graph:<name>`. A class is a literal, because it has no IRI until an
ontology import supplies one. An edge is a resource with `subject` and `object`
links and its provenance and interval as properties, not a bare predicate.

**rdflib is imported only when a shard is serialized.** `graph_rdf.py` is the
only module that reaches for it, through `import_optional_dependency`. `--format
jsonld` or `turtle` without the extra fails in `check_format`, before the
project, profile or warehouse is read, naming `pip install 'stel[rdf]'`. A
JSON Lines export never imports it.

## Alternatives rejected

- **Mint HTTP IRIs under a stel domain.** stel owns no such domain, and an IRI
  that dereferences to nothing, or to someone else's page, is worse than one
  that says it is an identifier. A `urn:` states what the value is. A
  configurable `--iri-base` is deferred until ontology IRIs exist (ALE-419
  follow-up), because two sources of node IRIs is the thing to avoid.
- **Use `canonical_id` as the IRI.** It is a string in the operator's
  namespace, not an IRI, and two namespaces may reuse one. `entity_id` already
  includes the namespace and `entity_name`.
- **One RDF file per export.** It defeats ADR-0061: one changed document
  rewrites the file, and writing it needs every document's triples resident.
- **Convert an existing directory when the format changes.** It means keeping
  two extensions' shards consistent through a crash. A new directory, or
  deleting this one, is cheap and has one answer.
- **Hand-write the serializations and skip rdflib.** Turtle escaping and
  JSON-LD compaction are what rdflib is for, and the tests parse the files back
  with it. JSON-LD ordering, which rdflib leaves unspecified, is normalized
  after it so shard bytes (and so fingerprints) are stable.
- **N-Triples or N-Quads.** Line-based and concatenable, but the issue asked for
  JSON-LD and Turtle; either can be added as a format without a design change.

## Consequences

- A change of rdflib version that alters its serialization rewrites every
  shard once on the next export. Nothing else is affected.
- A class's ancestors become an unordered set of literals; the nearest-first
  order exists only in the JSON Lines `class_ancestors` array.
- Revisit when ontology import supplies IRIs for entities and classes: node
  identity may then prefer them, and `class` becomes a link.
