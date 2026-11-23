"""RDF serializations of one document's graph shard (ALE-602, ADR-0062).

`graph_export` builds each document's nodes and edges as plain rows. This
module turns the same rows into Turtle or JSON-LD, so the three formats are
the one graph and a loader can pick whichever it reads. It is the only code
that touches rdflib, imported when a shard is serialized and never at import
time, so the core install stays lean: without the `rdf` extra the export fails
before the project or warehouse is read, naming the install command.

What identifies a node is stated, not invented from a domain stel does not
own. Until an imported ontology supplies IRIs (ALE-419), every identifier is a
URN built from a value the export already carries:

- an entity is `urn:stel:entity:<entity_id>`, the agent-context id the JSON
  Lines export keys nodes by, so the three formats join on the same value;
- a relation (one extracted fact) is `urn:stel:relation:<relation_id>`;
- a document is `urn:stel:document:<document_id>`.

Terms are `urn:stel:graph:<name>`. A class is a literal, not an IRI, because a
class has no IRI until an ontology gives it one. An edge is a resource, not a
bare predicate: a fact carries its document, extractor and validity interval,
and a plain triple would drop them.

Every shard is a self-contained document with its own prefixes or `@context`,
and a shard's triples are a set, so loading the directory is loading each file:
an entity linked in several documents is the same triples in each and merges.

Output is deterministic: the same rows give the same bytes, which the shard
fingerprint depends on.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from .optional_dependencies import import_optional_dependency

if TYPE_CHECKING:
    from rdflib import Graph

RDF_FORMATS = ("jsonld", "turtle")

TERMS = "urn:stel:graph:"
ENTITY_IRI = "urn:stel:entity:"
RELATION_IRI = "urn:stel:relation:"
DOCUMENT_IRI = "urn:stel:document:"
XSD = "http://www.w3.org/2001/XMLSchema#"

_STRINGS = {
    "entity_namespace": "entityNamespace",
    "entity_name": "entityName",
    "canonical_id": "canonicalId",
    "class": "class",
    "relation_type": "relationType",
    "method": "method",
    "extractor": "extractor",
    "extractor_version": "extractorVersion",
    "subject_resolver": "subjectResolver",
    "subject_resolver_version": "subjectResolverVersion",
    "object_resolver": "objectResolver",
    "object_resolver_version": "objectResolverVersion",
}
_INSTANTS = {
    "valid_from": "validFrom",
    "valid_to": "validTo",
    "observed_at": "observedAt",
}
# Which IRI scheme an edge column's value is in, for the columns that name
# another resource.
_REFERENCES = {
    "subject_entity_id": ("subject", ENTITY_IRI),
    "object_entity_id": ("object", ENTITY_IRI),
    "document_id": ("document", DOCUMENT_IRI),
    "superseded_by": ("supersededBy", RELATION_IRI),
}


def require_rdflib(feature: str) -> Any:
    """rdflib, or an error naming the install command. Called before any
    project, profile or warehouse is read."""
    return import_optional_dependency("rdflib", extra="rdf", feature=feature)


def _iri(scheme: str, value: str) -> str:
    return scheme + quote(value, safe="")


def _jsonld_context() -> dict[str, Any]:
    context: dict[str, Any] = {"stel": TERMS, "xsd": XSD}
    for name in (*_STRINGS.values(), "classAncestor"):
        context[name] = {"@id": f"stel:{name}"}
    for name in _INSTANTS.values():
        context[name] = {"@id": f"stel:{name}", "@type": "xsd:dateTime"}
    for name, _scheme in _REFERENCES.values():
        context[name] = {"@id": f"stel:{name}", "@type": "@id"}
    context["directed"] = {"@id": "stel:directed", "@type": "xsd:boolean"}
    context["confidence"] = {"@id": "stel:confidence", "@type": "xsd:double"}
    context["sentenceIndex"] = {"@id": "stel:sentenceIndex", "@type": "xsd:integer"}
    return context


def _node_triples(rdflib: Any, graph: Graph, row: Mapping[str, Any]) -> None:
    stel = rdflib.Namespace(TERMS)
    subject = rdflib.URIRef(_iri(ENTITY_IRI, str(row["entity_id"])))
    graph.add((subject, rdflib.RDF.type, stel.Entity))
    for column, term in _STRINGS.items():
        if row.get(column) is not None:
            graph.add((subject, stel[term], rdflib.Literal(str(row[column]))))
    for ancestor in row.get("class_ancestors") or ():
        graph.add((subject, stel.classAncestor, rdflib.Literal(str(ancestor))))


def _edge_triples(rdflib: Any, graph: Graph, row: Mapping[str, Any]) -> None:
    stel = rdflib.Namespace(TERMS)
    xsd = rdflib.XSD
    subject = rdflib.URIRef(_iri(RELATION_IRI, str(row["relation_id"])))
    graph.add((subject, rdflib.RDF.type, stel.Relation))
    for column, (term, scheme) in _REFERENCES.items():
        if row.get(column) is not None:
            graph.add((subject, stel[term], rdflib.URIRef(_iri(scheme, str(row[column])))))
    for column, term in _STRINGS.items():
        if row.get(column) is not None:
            graph.add((subject, stel[term], rdflib.Literal(str(row[column]))))
    # The lexical form is written as the JSON Lines export has it: rdflib would
    # otherwise rewrite `Z` as `+00:00`, so the three formats would disagree.
    typed = [(term, row.get(column), xsd.dateTime) for column, term in _INSTANTS.items()]
    if row.get("confidence") is not None:
        typed.append(("confidence", repr(float(row["confidence"])), xsd.double))
    if row.get("sentence_index") is not None:
        typed.append(("sentenceIndex", str(int(row["sentence_index"])), xsd.integer))
    for term, lexical, datatype in typed:
        if lexical is not None:
            literal = rdflib.Literal(str(lexical), datatype=datatype, normalize=False)
            graph.add((subject, stel[term], literal))
    graph.add((subject, stel.directed, rdflib.Literal(bool(row["directed"]))))


def _canonical_jsonld(text: str) -> bytes:
    """rdflib's JSON-LD with every ordering rdflib leaves to chance fixed.

    A graph is a set, so the serializer is free to emit nodes and a node's
    repeated values in any order, and a string-hash seed changes it between
    processes. The shard fingerprint would then see a change that is not one.
    """
    document = json.loads(text)
    nodes = document["@graph"] if "@graph" in document else [
        {key: value for key, value in document.items() if key != "@context"}
    ]
    for node in nodes:
        for key, value in node.items():
            if isinstance(value, list):
                node[key] = sorted(value, key=lambda item: json.dumps(item, sort_keys=True))
    nodes.sort(key=lambda node: node["@id"])
    result = {"@context": document["@context"], "@graph": nodes}
    return (json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def serialize_rows(rows: Sequence[Mapping[str, Any]], *, kind: str, graph_format: str) -> bytes:
    """One shard file's rows as `graph_format`; empty bytes for no rows.

    `kind` is `nodes` or `edges`, the two trees a shard is split across.
    """
    if not rows:
        return b""
    rdflib = require_rdflib(f"The `{graph_format}` graph export format")
    graph = rdflib.Graph()
    graph.bind("stel", TERMS)
    add = _node_triples if kind == "nodes" else _edge_triples
    for row in rows:
        add(rdflib, graph, row)
    if graph_format == "turtle":
        return graph.serialize(format="turtle").encode("utf-8")
    text = graph.serialize(format="json-ld", context=_jsonld_context(), auto_compact=False)
    return _canonical_jsonld(text)
