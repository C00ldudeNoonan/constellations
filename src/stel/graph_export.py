"""Graph egress: linked entities and their relations as nodes and edges (#762).

Entity links and relations land as warehouse tables, and the concept cloud
renders them for a person. This is the machine-readable counterpart: a
directory of JSON Lines a downstream graph store can load without a consumer
re-deriving node identity.

Two properties carry the design (ADR-0055):

- **A node is keyed by the agent-context `entity_id`.** The id is
  `make_entity_id(entity_namespace, entity_name, canonical_json(canonical_id))`,
  the same one `project_entity_link` writes, so an exported node joins to a
  governed metric that keys the same entity. The links table does not carry
  `entity_name` (it is the operator's choice), so the export takes it rather
  than guessing: a default would mint ids that join to nothing, silently.
- **An edge is one extracted fact, not an aggregate.** The concept cloud's
  weighted edges collapse documents away; a graph store wants to ask "which
  document said this, and which extractor version asserted it". Every edge
  therefore names its relation row, document, extractor and the resolvers that
  linked its two ends.

`build_graph` is a pure function over frames so it is exercised without a
warehouse; `export_graph` reads the frames through the active adapter and
`write_graph_export` writes the files. Output is deterministic and carries no
timestamp, so two exports of unchanged tables are byte-identical.

An edge also carries the interval its fact holds over (`valid_from`, `valid_to`,
`observed_at`, `superseded_by`), when the relation model has those columns
(ADR-0056, schema v2).

Only intended output tables are read. Mention text is never read: a node is a
canonical id and an edge is a pair of them, so nothing the operator withheld
upstream (`include_mention_text`) can reach the files.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from .agent_context import canonical_entity_key, make_entity_id
from .config import load_project
from .config.class_hierarchy import ClassDeclaration, class_ancestors, class_parents
from .config.vocabulary import Vocabulary, VocabularyTerm, declared_terms

GRAPH_EXPORT_SCHEMA = "stel.graph_export/v2"
NODES_FILENAME = "nodes.jsonl"
EDGES_FILENAME = "edges.jsonl"
GRAPH_MANIFEST_FILENAME = "graph_manifest.json"

_LINK_COLUMNS = (
    "mention_id",
    "canonical_id",
    "entity_namespace",
    "status",
    "resolver",
    "resolver_version",
)
_RELATION_COLUMNS = (
    "relation_id",
    "document_id",
    "subject_mention_id",
    "object_mention_id",
    "relation_type",
    "directed",
    "method",
    "status",
    "confidence",
    "sentence_index",
    "extractor",
    "extractor_version",
)


class GraphExportError(Exception):
    pass


@dataclass(frozen=True)
class _Linked:
    """What an entity link says about one mention."""

    entity_id: str
    entity_namespace: str
    canonical_id: str
    resolver: str | None
    resolver_version: str | None


@dataclass(frozen=True)
class GraphExport:
    project: str
    entity_name: str
    nodes: tuple[dict[str, Any], ...]
    edges: tuple[dict[str, Any], ...]
    # Relation rows the graph left out, by reason, so a count that looks short
    # is explained rather than discovered.
    edges_skipped: dict[str, int]


def _text(value: object) -> str | None:
    # A null stays null; `str(None)` would export the string "None".
    return None if value is None else str(value)


def _number(value: object) -> float | None:
    # NaN is not JSON; a missing score is null, not an error at write time.
    if isinstance(value, float) and value != value:
        return None
    return value if isinstance(value, int | float) else None


def _instant(value: object, column: str) -> str | None:
    """A validity timestamp as UTC ISO-8601 with a `Z`, so the file reads the
    same wherever it is loaded. A naive datetime is read as UTC, as everywhere
    else a relation row carries an instant (ADR-0056)."""
    if value is None:
        return None
    if not isinstance(value, datetime):
        raise GraphExportError(
            f"the relation model's column {column} must hold timestamps; "
            "`extract_relations` and `supersede_relations` write it that way"
        )
    stamped = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return stamped.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _clean_entity_name(entity_name: str) -> str:
    cleaned = entity_name.strip()
    if not cleaned:
        raise GraphExportError("entity_name must not be empty")
    return cleaned


def _require_columns(frame: pl.DataFrame, required: Sequence[str], model: str) -> None:
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise GraphExportError(
            f"the {model} is missing column(s) {', '.join(missing)}; the graph "
            "export reads the output of link_entities and extract_relations"
        )


def _linked_mentions(links: pl.DataFrame, entity_name: str) -> dict[str, _Linked]:
    """Mention id -> its resolved entity, for matched links only.

    An ambiguous or unmatched mention names no single entity, and the governed
    context drops it for the same reason (`context_entity_links`). A mention
    two matched links resolve to different entities is a defect upstream, so it
    fails the export instead of letting row order pick a side.
    """
    mentions: dict[str, _Linked] = {}
    matched = links.filter(
        (pl.col("status") == "matched")
        & pl.col("canonical_id").is_not_null()
        & pl.col("entity_namespace").is_not_null()
    )
    for row in matched.iter_rows(named=True):
        canonical = row["canonical_id"]
        if not str(canonical):
            continue
        namespace = str(row["entity_namespace"])
        linked = _Linked(
            entity_id=make_entity_id(
                namespace, entity_name, canonical_entity_key(str(canonical))
            ),
            entity_namespace=namespace,
            canonical_id=str(canonical),
            resolver=_text(row["resolver"]),
            resolver_version=_text(row["resolver_version"]),
        )
        mention_id = str(row["mention_id"])
        known = mentions.get(mention_id)
        if known is not None and known.entity_id != linked.entity_id:
            raise GraphExportError(
                f"mention '{mention_id}' is matched to two different entities; "
                "the linking model must resolve a mention to one"
            )
        mentions[mention_id] = linked
    return mentions


def _node(
    linked: _Linked,
    entity_name: str,
    terms: Mapping[tuple[str, str], VocabularyTerm],
    parents: Mapping[str, str],
) -> dict[str, Any]:
    term = terms.get((linked.entity_namespace, linked.canonical_id))
    entity_class = term.entity_class if term is not None else None
    return {
        "entity_id": linked.entity_id,
        "entity_namespace": linked.entity_namespace,
        "entity_name": entity_name,
        "canonical_id": linked.canonical_id,
        "class": entity_class,
        "class_ancestors": (
            list(class_ancestors(entity_class, parents)) if entity_class else []
        ),
    }


def _edge(row: dict[str, Any], subject: _Linked, obj: _Linked) -> dict[str, Any]:
    return {
        "relation_id": str(row["relation_id"]),
        "subject_entity_id": subject.entity_id,
        "object_entity_id": obj.entity_id,
        "relation_type": str(row["relation_type"]),
        "directed": bool(row["directed"]),
        "method": str(row["method"]),
        "confidence": _number(row["confidence"]),
        "document_id": str(row["document_id"]),
        "sentence_index": row["sentence_index"],
        "extractor": str(row["extractor"]),
        "extractor_version": str(row["extractor_version"]),
        "subject_resolver": subject.resolver,
        "subject_resolver_version": subject.resolver_version,
        "object_resolver": obj.resolver,
        "object_resolver_version": obj.resolver_version,
        # The interval the fact holds over (ADR-0056). A relation model built
        # before these columns existed exports null: the validity is unknown.
        "valid_from": _instant(row.get("valid_from"), "valid_from"),
        "valid_to": _instant(row.get("valid_to"), "valid_to"),
        "observed_at": _instant(row.get("observed_at"), "observed_at"),
        "superseded_by": _text(row.get("superseded_by")),
    }


def build_graph(
    *,
    project: str,
    links: pl.DataFrame,
    relations: pl.DataFrame,
    entity_name: str,
    vocabularies: Mapping[str, Vocabulary],
    classes: Sequence[ClassDeclaration],
) -> GraphExport:
    """Nodes and edges from entity-link and relation frames.

    A relation becomes an edge when it is `asserted` and both its mentions
    resolve to a matched entity, and the two entities differ. An undirected
    relation (co-occurrence) is written once with its ends in `entity_id`
    order, so the same pair never appears as two edges that differ only in
    which mention came first.

    The skip reasons are exclusive and checked in the order `not_asserted`,
    `unlinked_mention`, `self_relation`: a row is counted under the first that
    applies, so a non-asserted row never also counts as unlinked.
    """
    entity_name = _clean_entity_name(entity_name)
    _require_columns(links, _LINK_COLUMNS, "linking model")
    _require_columns(relations, _RELATION_COLUMNS, "relation model")

    mentions = _linked_mentions(links, entity_name)
    terms = declared_terms(vocabularies)
    parents = class_parents(classes)
    nodes: dict[str, dict[str, Any]] = {}
    for linked in mentions.values():
        if linked.entity_id not in nodes:
            nodes[linked.entity_id] = _node(linked, entity_name, terms, parents)

    skipped = {"not_asserted": 0, "unlinked_mention": 0, "self_relation": 0}
    edges: dict[str, dict[str, Any]] = {}
    for row in relations.iter_rows(named=True):
        if row["status"] != "asserted":
            skipped["not_asserted"] += 1
            continue
        subject = mentions.get(str(row["subject_mention_id"]))
        obj = mentions.get(str(row["object_mention_id"]))
        if subject is None or obj is None:
            skipped["unlinked_mention"] += 1
            continue
        if subject.entity_id == obj.entity_id:
            skipped["self_relation"] += 1
            continue
        if not bool(row["directed"]) and subject.entity_id > obj.entity_id:
            subject, obj = obj, subject
        edge = _edge(row, subject, obj)
        if edge["relation_id"] in edges:
            raise GraphExportError(
                f"relation_id '{edge['relation_id']}' appears twice in the "
                "relation model; it must be unique"
            )
        edges[edge["relation_id"]] = edge

    return GraphExport(
        project=project,
        entity_name=entity_name,
        nodes=tuple(nodes[key] for key in sorted(nodes)),
        edges=tuple(edges[key] for key in sorted(edges)),
        edges_skipped=skipped,
    )


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(
            row, sort_keys=True, ensure_ascii=False, allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    ).encode("utf-8")


def _write_file(path: Path, payload: bytes) -> None:
    # Replaced rather than rewritten so a reader never sees half a file.
    scratch = path.with_name(path.name + ".tmp")
    try:
        scratch.write_bytes(payload)
        scratch.replace(path)
    except BaseException:
        scratch.unlink(missing_ok=True)
        raise


def write_graph_export(
    graph: GraphExport,
    output_dir: Path,
    *,
    linking_model: str,
    relation_model: str,
) -> Path:
    """Write `nodes.jsonl`, `edges.jsonl` and `graph_manifest.json`.

    The manifest is written last, so a directory that has one is complete. It
    carries the schema version, the models read, row counts and a digest of
    each file, and no timestamp, so unchanged inputs give identical bytes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # A manifest left by an earlier export would vouch for files about to be
    # replaced, so it goes first: a directory with a manifest is then complete.
    (output_dir / GRAPH_MANIFEST_FILENAME).unlink(missing_ok=True)
    payloads = {
        NODES_FILENAME: (_jsonl(graph.nodes), len(graph.nodes)),
        EDGES_FILENAME: (_jsonl(graph.edges), len(graph.edges)),
    }
    for name, (payload, _) in payloads.items():
        _write_file(output_dir / name, payload)
    manifest = {
        "schema": GRAPH_EXPORT_SCHEMA,
        "project": graph.project,
        "entity_name": graph.entity_name,
        "linking_model": linking_model,
        "relation_model": relation_model,
        "files": {
            name: {"rows": rows, "sha256": hashlib.sha256(payload).hexdigest()}
            for name, (payload, rows) in sorted(payloads.items())
        },
        "edges_skipped": dict(sorted(graph.edges_skipped.items())),
    }
    _write_file(
        output_dir / GRAPH_MANIFEST_FILENAME,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return output_dir


def export_graph(
    project_dir: str | Path,
    *,
    linking_model: str,
    relation_model: str,
    entity_name: str,
    target: str | None,
    profiles_dir: str | Path | None,
) -> GraphExport:
    """Read the two models through the active adapter and build the graph."""
    from .adapters import create_adapter
    from .profile import resolve_profile

    # Checked before the project, profile or warehouse is touched.
    entity_name = _clean_entity_name(entity_name)
    project_path = Path(project_dir)
    project, _, _ = load_project(project_path)
    resolved = resolve_profile(
        project, project_path, target=target,
        profiles_dir=Path(profiles_dir) if profiles_dir else None,
    )
    with create_adapter(resolved.warehouse, project_dir=project_path) as adapter:
        links = adapter.read_table(linking_model)
        relations = adapter.read_table(relation_model)
    return build_graph(
        project=project.name,
        links=links,
        relations=relations,
        entity_name=entity_name,
        vocabularies=project.vocabularies,
        classes=project.classes,
    )
