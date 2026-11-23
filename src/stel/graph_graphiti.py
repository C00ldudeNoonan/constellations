"""Graphiti records for one document's graph shard (ALE-603, ADR-0065).

Graphiti (graphiti-core) stores a temporal fact graph: entity nodes, entity
edges that carry a fact and the interval it held over, and episodes, the
sources the facts came from. `graph_export` builds each document's nodes and
edges as plain rows; this module turns the same rows into the records Graphiti's
own models take, so `stel graph-load` can save them with `Model.save(driver)`
and no hand transformation. It is the only export code that reaches for
graphiti-core, imported when a shard is written and never at import time: without
the `graphiti` extra the export fails before the project or warehouse is read,
naming the install command.

Every record is validated against the installed graphiti-core model before it
is written, so a model that drifts from this shape fails the export, not a load
three steps later.

**Each document is one episode.** A document's nodes shard holds its entity
nodes, one `episodic_node` and a `mentions` edge from the episode to each
entity; its edges shard holds one `entity_edge` per extracted fact, whose
`episodes` names the document's episode. The episode's content is the facts as
short statements (`<subject canonical id> <relation type> <object canonical
id>`) and the document id. It never holds document text: stel does not read
mention text for the export.

**What fills Graphiti's temporal fields.** Graphiti distinguishes when a fact
was true (valid time) from when the graph learned of it (transaction time).
The file carries only what stel measured, which is valid time, and the loader
stamps the rest, so a file stays byte-identical across exports (ADR-0055):

- `EntityEdge.valid_at`: the relation's `valid_from`, null when unknown.
- `EntityEdge.invalid_at`: the relation's `valid_to`, null when open.
- `EntityEdge.reference_time`: the relation's `observed_at`, null when unknown.
- `EntityEdge.expired_at`: the load time for an edge `supersede_relations`
  closed (`superseded_by` is set), else null.
- `EpisodicNode.valid_at`: the earliest `observed_at` among the document's
  edges, or the load time when none has one.
- `created_at` on any node or edge: the load time.

Embeddings are not written: stel embeds with its own models, and a Graphiti
store needs its own embedder for vector search. Text search over names and
facts works without them.

Identity: every uuid is a `uuid5` of the group id and the `urn:` the RDF format
uses for the same thing (ADR-0063), so an entity linked in several documents is
the same node in each and merges when loaded.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from .graph_rdf import DOCUMENT_IRI, ENTITY_IRI, RELATION_IRI
from .optional_dependencies import import_optional_dependency

SOURCE_DESCRIPTION = "stel graph export"

# What the loader stamps, and what the export validates against; a fixed instant
# keeps the validation independent of the clock.
_VALIDATION_TIME = datetime(1970, 1, 1, tzinfo=UTC)
_GROUP_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")


def require_graphiti(feature: str) -> Any:
    """graphiti-core, or an error naming the install command. Called before any
    project, profile or warehouse is read."""
    return import_optional_dependency(
        "graphiti_core", extra="graphiti", feature=feature, distribution="graphiti-core"
    )


def group_id_for(project: str) -> str:
    """Graphiti accepts letters, digits, dashes and underscores in a group id.

    A name that needed cleaning carries a digest of the original, so `sales/eu`
    and `sales_eu` stay two groups instead of merging their records on load.
    """
    cleaned = _GROUP_UNSAFE.sub("_", project)
    if cleaned == project:
        return cleaned or "stel"
    digest = hashlib.sha256(project.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned}_{digest}"


def _uuid(group_id: str, name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{group_id}|{name}"))


def _entity_uuid(group_id: str, entity_id: str) -> str:
    return _uuid(group_id, ENTITY_IRI + entity_id)


def _present(values: Mapping[str, Any]) -> dict[str, Any]:
    # A graph database cannot store a null property, and "unknown" is the
    # absence of one.
    return {key: value for key, value in values.items() if value is not None}


def _entity_record(node: Mapping[str, Any], group_id: str) -> dict[str, Any]:
    return {
        "record": "entity_node",
        "uuid": _entity_uuid(group_id, node["entity_id"]),
        "name": node["canonical_id"],
        "group_id": group_id,
        "labels": ["Entity"],
        "summary": "",
        "attributes": _present(
            {
                "entity_id": node["entity_id"],
                "entity_namespace": node["entity_namespace"],
                "entity_name": node["entity_name"],
                "canonical_id": node["canonical_id"],
                "entity_class": node["class"],
                "class_ancestors": node["class_ancestors"] or None,
            }
        ),
    }


def _fact(edge: Mapping[str, Any], canonical: Mapping[str, str]) -> str:
    subject = canonical[edge["subject_entity_id"]]
    obj = canonical[edge["object_entity_id"]]
    return f"{subject} {edge['relation_type']} {obj}"


def _edge_record(
    edge: Mapping[str, Any], canonical: Mapping[str, str], episode: str, group_id: str
) -> dict[str, Any]:
    return {
        "record": "entity_edge",
        "uuid": _uuid(group_id, RELATION_IRI + edge["relation_id"]),
        "group_id": group_id,
        "source_node_uuid": _entity_uuid(group_id, edge["subject_entity_id"]),
        "target_node_uuid": _entity_uuid(group_id, edge["object_entity_id"]),
        "name": edge["relation_type"],
        "fact": _fact(edge, canonical),
        "episodes": [episode],
        "valid_at": edge["valid_from"],
        "invalid_at": edge["valid_to"],
        "reference_time": edge["observed_at"],
        "attributes": _present(
            {
                "relation_id": edge["relation_id"],
                "directed": edge["directed"],
                "method": edge["method"],
                "confidence": edge["confidence"],
                "document_id": edge["document_id"],
                "sentence_index": edge["sentence_index"],
                "extractor": edge["extractor"],
                "extractor_version": edge["extractor_version"],
                "subject_resolver": edge["subject_resolver"],
                "subject_resolver_version": edge["subject_resolver_version"],
                "object_resolver": edge["object_resolver"],
                "object_resolver_version": edge["object_resolver_version"],
                "superseded_by": edge["superseded_by"],
            }
        ),
    }


def _parse_instant(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _episode_time(edges: Sequence[Mapping[str, Any]]) -> str | None:
    observed = [_parse_instant(edge["observed_at"]) for edge in edges if edge["observed_at"]]
    if not observed:
        return None
    return min(observed).astimezone(UTC).isoformat().replace("+00:00", "Z")


def build_records(
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    *,
    document_id: str,
    group_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """A document's `(nodes shard records, edges shard records)`, validated.

    The nodes shard is empty when the document has no nodes: an edge needs two
    linked entities, so a document with edges always has nodes.
    """
    if not nodes:
        return [], []
    episode = _uuid(group_id, DOCUMENT_IRI + document_id)
    canonical = {node["entity_id"]: node["canonical_id"] for node in nodes}
    edge_records = [_edge_record(edge, canonical, episode, group_id) for edge in edges]
    facts = [record["fact"] for record in edge_records]
    node_records = [_entity_record(node, group_id) for node in nodes]
    node_records.append(
        {
            "record": "episodic_node",
            "uuid": episode,
            "name": document_id,
            "group_id": group_id,
            "labels": ["Episodic"],
            "source": "json",
            "source_description": SOURCE_DESCRIPTION,
            "content": json.dumps(
                {"document_id": document_id, "facts": facts},
                sort_keys=True, ensure_ascii=False, separators=(",", ":"),
            ),
            "valid_at": _episode_time(edges),
            "entity_edges": [record["uuid"] for record in edge_records],
        }
    )
    entities = [record for record in node_records if record["record"] == "entity_node"]
    node_records.extend(
        {
            "record": "mentions",
            "uuid": _uuid(group_id, f"mentions|{episode}|{entity['uuid']}"),
            "group_id": group_id,
            "source_node_uuid": episode,
            "target_node_uuid": entity["uuid"],
        }
        for entity in entities
    )
    for record in (*node_records, *edge_records):
        to_model(record, _VALIDATION_TIME)
    return node_records, edge_records


def to_model(record: Mapping[str, Any], loaded_at: datetime) -> Any:
    """The graphiti-core model for one record, with what the loader owns filled
    in. The export validates through this same function, so what it writes is
    what the loader will build."""
    require_graphiti("The Graphiti graph format")
    from graphiti_core.edges import EntityEdge, EpisodicEdge
    from graphiti_core.nodes import EntityNode, EpisodicNode

    fields = {key: value for key, value in record.items() if key != "record"}
    kind = record["record"]
    if kind == "entity_node":
        return EntityNode.model_validate({**fields, "created_at": loaded_at})
    if kind == "episodic_node":
        valid_at = fields.pop("valid_at") or loaded_at
        return EpisodicNode.model_validate(
            {**fields, "valid_at": valid_at, "created_at": loaded_at}
        )
    if kind == "mentions":
        return EpisodicEdge.model_validate({**fields, "created_at": loaded_at})
    if kind == "entity_edge":
        superseded = "superseded_by" in fields["attributes"]
        return EntityEdge.model_validate(
            {**fields, "created_at": loaded_at, "expired_at": loaded_at if superseded else None}
        )
    raise ValueError(f"unknown Graphiti record kind {kind!r}")


def _line(record: Mapping[str, Any]) -> str:
    return json.dumps(
        record, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ) + "\n"


def serialize_document(
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    *,
    document_id: str,
    group_id: str,
) -> tuple[bytes, bytes]:
    """A document's two shard files' bytes: `(nodes, edges)`, empty for none."""
    require_graphiti("The `graphiti` graph export format")
    node_records, edge_records = build_records(
        nodes, edges, document_id=document_id, group_id=group_id
    )
    return (
        "".join(_line(record) for record in node_records).encode("utf-8"),
        "".join(_line(record) for record in edge_records).encode("utf-8"),
    )


def read_records(payload: bytes) -> Iterator[dict[str, Any]]:
    for line in payload.decode("utf-8").splitlines():
        if line:
            yield json.loads(line)
