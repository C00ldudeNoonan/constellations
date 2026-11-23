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

The export is incremental by document (ADR-0061, schema v3). A document is the
unit: its nodes and edges are one *shard* of two files, named by a hash of the
document id, and `graph_manifest.json` carries a fingerprint per shard. A
re-export streams both tables ordered by `document_id`, joins the two streams
document by document so only one document's rows are resident, builds each
shard and rewrites only the ones whose fingerprint moved. A document that left
the source is stale in the same sense a published key absent from its upstream
is stale in the state reconciler (`state_reconciliation.py`, #153/#764): absence
is judged against exactly the rows this export read, the streams are validated
to be strictly ascending rather than trusted, an empty source makes every
shard stale, and nothing is removed until the whole read has succeeded.

`build_document` is a pure function over one document's rows so it is
exercised without a warehouse; `export_graph` reads the tables through the
active adapter and `stage_graph` / `commit_graph` write the files. Output is
deterministic and carries no timestamp, so two exports of unchanged tables
leave every file, and its modification time, alone.

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
import re
import shutil
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .adapters.base import AdapterError, TableReadSnapshot
from .agent_context import canonical_entity_key, make_entity_id
from .config import load_project
from .config.class_hierarchy import ClassDeclaration, class_ancestors, class_parents
from .config.vocabulary import Vocabulary, VocabularyTerm, declared_terms

GRAPH_EXPORT_SCHEMA = "stel.graph_export/v3"
GRAPH_MANIFEST_FILENAME = "graph_manifest.json"
NODES_DIRNAME = "nodes"
EDGES_DIRNAME = "edges"
# v1 and v2 wrote these two whole files; v3 replaces them with shards.
LEGACY_SCHEMAS = ("stel.graph_export/v1", "stel.graph_export/v2")
LEGACY_FILENAMES = ("nodes.jsonl", "edges.jsonl")

_STAGE_DIRNAME = ".graph_export_stage"
_SHARD_NAME = re.compile(r"[0-9a-f]{32}\.jsonl")
_PREFIX_NAME = re.compile(r"[0-9a-f]{2}")
# Rows pulled per warehouse batch. Residency is this plus one document.
_READ_BATCH_ROWS = 10_000

LINK_COLUMNS = (
    "entity_link_id",
    "document_id",
    "mention_id",
    "canonical_id",
    "entity_namespace",
    "status",
    "resolver",
    "resolver_version",
)
RELATION_COLUMNS = (
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
# Present on a relation model built after ADR-0056; read when there.
_OPTIONAL_RELATION_COLUMNS = ("valid_from", "valid_to", "observed_at", "superseded_by")

DocumentRows = tuple[str, Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]


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
class GraphContext:
    """What turns a document's rows into nodes: the operator's `entity_name`
    and the declarations a node's class is read from."""

    entity_name: str
    terms: Mapping[tuple[str, str], VocabularyTerm]
    parents: Mapping[str, str]

    @classmethod
    def create(
        cls,
        entity_name: str,
        vocabularies: Mapping[str, Vocabulary],
        classes: Sequence[ClassDeclaration],
    ) -> GraphContext:
        return cls(
            entity_name=_clean_entity_name(entity_name),
            terms=declared_terms(vocabularies),
            parents=class_parents(classes),
        )


@dataclass(frozen=True)
class DocumentGraph:
    """One document's shard: its nodes, its edges, and what it left out."""

    nodes: tuple[dict[str, Any], ...]
    edges: tuple[dict[str, Any], ...]
    # Relation rows the document's graph left out, by reason, so a count that
    # looks short is explained rather than discovered.
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


def require_columns(available: Sequence[str], required: Sequence[str], model: str) -> None:
    missing = [name for name in required if name not in available]
    if missing:
        raise GraphExportError(
            f"the {model} is missing column(s) {', '.join(missing)}; the graph "
            "export reads the output of link_entities and extract_relations"
        )


def _linked_mentions(
    links: Iterable[Mapping[str, Any]], entity_name: str
) -> dict[str, _Linked]:
    """Mention id -> its resolved entity, for matched links only.

    An ambiguous or unmatched mention names no single entity, and the governed
    context drops it for the same reason (`context_entity_links`). A mention
    two matched links resolve to different entities is a defect upstream, so it
    fails the export instead of letting row order pick a side.
    """
    mentions: dict[str, _Linked] = {}
    for row in links:
        canonical = row["canonical_id"]
        if (
            row["status"] != "matched"
            or canonical is None
            or row["entity_namespace"] is None
            or not str(canonical)
        ):
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


def _edge(row: Mapping[str, Any], subject: _Linked, obj: _Linked) -> dict[str, Any]:
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


def build_document(
    *,
    links: Sequence[Mapping[str, Any]],
    relations: Sequence[Mapping[str, Any]],
    context: GraphContext,
) -> DocumentGraph:
    """One document's nodes and edges from its link and relation rows.

    A relation becomes an edge when it is `asserted` and both its mentions
    resolve to a matched entity, and the two entities differ. An undirected
    relation (co-occurrence) is written once with its ends in `entity_id`
    order, so the same pair never appears as two edges that differ only in
    which mention came first.

    A relation's mentions are resolved against its own document's links, which
    is where `extract_relations` takes them from.

    The skip reasons are exclusive and checked in the order `not_asserted`,
    `unlinked_mention`, `self_relation`: a row is counted under the first that
    applies, so a non-asserted row never also counts as unlinked.
    """
    mentions = _linked_mentions(links, context.entity_name)
    nodes: dict[str, dict[str, Any]] = {}
    for linked in mentions.values():
        if linked.entity_id not in nodes:
            nodes[linked.entity_id] = _node(
                linked, context.entity_name, context.terms, context.parents
            )

    skipped = {"not_asserted": 0, "unlinked_mention": 0, "self_relation": 0}
    edges: dict[str, dict[str, Any]] = {}
    for row in relations:
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

    return DocumentGraph(
        nodes=tuple(nodes[key] for key in sorted(nodes)),
        edges=tuple(edges[key] for key in sorted(edges)),
        edges_skipped=skipped,
    )


def _grouped_by_document(
    rows: Iterable[Mapping[str, Any]], model: str
) -> Iterator[tuple[str, list[Mapping[str, Any]]]]:
    """`(document_id, its rows)` from a stream ordered by `document_id`.

    The order is checked here, not assumed: a warehouse that collates ids
    differently from Python would otherwise let the two-stream join below pair
    the wrong documents without a word. A stream that moves backwards, or
    returns to a document it left, fails the export instead.
    """
    current: str | None = None
    group: list[Mapping[str, Any]] = []
    for row in rows:
        raw = row["document_id"]
        if raw is None or not str(raw):
            raise GraphExportError(
                f"the {model} has a row with no document_id; every row must name "
                "the document it came from"
            )
        document_id = str(raw)
        if document_id != current:
            if current is not None:
                if document_id < current:
                    raise GraphExportError(
                        f"the {model} is not in ascending document_id order "
                        f"('{document_id}' follows '{current}'); the warehouse's "
                        "ordering must match a plain string comparison"
                    )
                yield current, group
            current, group = document_id, []
        group.append(row)
    if current is not None:
        yield current, group


def merge_documents(
    links: Iterable[Mapping[str, Any]], relations: Iterable[Mapping[str, Any]]
) -> Iterator[DocumentRows]:
    """Join the two `document_id`-ordered streams into one document at a time.

    Holds one document's link rows and one document's relation rows, however
    large the tables are. A document present in only one table is yielded with
    nothing for the other: links with no relations still give nodes, and
    relations with no links give only skipped relations.
    """
    link_groups = _grouped_by_document(links, "linking model")
    relation_groups = _grouped_by_document(relations, "relation model")
    link = next(link_groups, None)
    relation = next(relation_groups, None)
    while link is not None or relation is not None:
        if link is not None and (relation is None or link[0] < relation[0]):
            yield link[0], link[1], []
            link = next(link_groups, None)
        elif relation is not None and (link is None or relation[0] < link[0]):
            yield relation[0], [], relation[1]
            relation = next(relation_groups, None)
        else:
            assert link is not None and relation is not None
            yield link[0], link[1], relation[1]
            link = next(link_groups, None)
            relation = next(relation_groups, None)


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


def _shard_id(document_id: str) -> str:
    # A document id is arbitrary text, so the file is named by its hash.
    return hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:32]


def _shard_path(root: Path, kind: str, shard: str) -> Path:
    # Two hex characters of fan-out keep a directory from holding every shard.
    return root / kind / shard[:2] / f"{shard}.jsonl"


def _fingerprint(nodes: bytes, edges: bytes) -> str:
    digest = hashlib.sha256()
    for part in (nodes, edges):
        digest.update(hashlib.sha256(part).digest())
    return digest.hexdigest()


@dataclass(frozen=True)
class _Prior:
    """What the output directory already holds, per its manifest."""

    present: bool
    documents: dict[str, dict[str, Any]]
    legacy_files: tuple[str, ...]


def _manifest_error(path: Path, why: str) -> GraphExportError:
    return GraphExportError(
        f"{path} {why}; refusing to overwrite an export it cannot vouch for. "
        "Export to another directory, or delete this one if its contents are disposable"
    )


def _read_prior(output_dir: Path) -> _Prior:
    path = output_dir / GRAPH_MANIFEST_FILENAME
    if not path.is_file():
        return _Prior(False, {}, ())
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise _manifest_error(path, "is not readable JSON") from error
    schema = data.get("schema") if isinstance(data, dict) else None
    if schema in LEGACY_SCHEMAS:
        # v1/v2 wrote whole files. They are replaced once the new export has
        # committed, never before: a failed run leaves the old export intact.
        return _Prior(False, {}, LEGACY_FILENAMES)
    if schema != GRAPH_EXPORT_SCHEMA:
        raise _manifest_error(path, f"is schema {schema!r}, not {GRAPH_EXPORT_SCHEMA}")
    documents = data.get("documents")
    if not isinstance(documents, dict):
        raise _manifest_error(path, "has no `documents` object")
    for entry in documents.values():
        if not (
            isinstance(entry, dict)
            and isinstance(entry.get("nodes"), int)
            and isinstance(entry.get("edges"), int)
            and isinstance(entry.get("sha256"), str)
            and isinstance(entry.get("shard"), str)
            and _SHARD_NAME.fullmatch(f"{entry['shard']}.jsonl")
        ):
            raise _manifest_error(path, "has a malformed `documents` entry")
    return _Prior(True, documents, ())


@dataclass(frozen=True)
class GraphExportResult:
    output_dir: Path
    documents: int
    node_rows: int
    edges: int
    edges_skipped: dict[str, int]
    shards_written: int
    shards_unchanged: int
    shards_removed: int


@dataclass
class Staged:
    """An export read in full and validated, not yet visible in the directory."""

    output_dir: Path
    prior: _Prior
    documents: dict[str, dict[str, Any]]
    changed: list[str]
    unchanged: int
    skipped: dict[str, int]
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def stage_dir(self) -> Path:
        return self.output_dir / _STAGE_DIRNAME

    def discard(self) -> None:
        if self.stage_dir.exists():
            shutil.rmtree(self.stage_dir, ignore_errors=True)


def _shard_files(entry: Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    return ((NODES_DIRNAME, entry["nodes"]), (EDGES_DIRNAME, entry["edges"]))


def stage_graph(
    documents: Iterable[DocumentRows],
    output_dir: Path,
    *,
    project: str,
    linking_model: str,
    relation_model: str,
    context: GraphContext,
) -> Staged:
    """Consume the document stream, building every shard and staging the changed
    ones in a scratch directory inside `output_dir`.

    Nothing the previous export wrote is touched: a document that fails
    validation, or a read that fails, leaves the directory as it was. Memory is
    one document plus the manifest's per-document entries, which is the
    distinct-key term of the bounded-memory contract.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _refuse_symlinked_trees(output_dir)
    staged = Staged(
        output_dir=output_dir,
        prior=_read_prior(output_dir),
        documents={},
        changed=[],
        unchanged=0,
        skipped={"not_asserted": 0, "unlinked_mention": 0, "self_relation": 0},
    )
    # A scratch directory a crashed run left behind is ours and is stale.
    staged.discard()
    try:
        for document_id, links, relations in documents:
            graph = build_document(links=links, relations=relations, context=context)
            for reason, count in graph.edges_skipped.items():
                staged.skipped[reason] += count
            if graph.nodes or graph.edges:
                _stage_document(staged, document_id, graph)
    except BaseException:
        staged.discard()
        raise
    staged.manifest = {
        "schema": GRAPH_EXPORT_SCHEMA,
        "project": project,
        "entity_name": context.entity_name,
        "linking_model": linking_model,
        "relation_model": relation_model,
        "totals": {
            "documents": len(staged.documents),
            "node_rows": sum(entry["nodes"] for entry in staged.documents.values()),
            "edges": sum(entry["edges"] for entry in staged.documents.values()),
        },
        "edges_skipped": dict(sorted(staged.skipped.items())),
        "documents": dict(sorted(staged.documents.items())),
    }
    return staged


def _stage_document(staged: Staged, document_id: str, graph: DocumentGraph) -> None:
    payloads = {NODES_DIRNAME: _jsonl(graph.nodes), EDGES_DIRNAME: _jsonl(graph.edges)}
    shard = _shard_id(document_id)
    entry = {
        "shard": shard,
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "sha256": _fingerprint(payloads[NODES_DIRNAME], payloads[EDGES_DIRNAME]),
    }
    staged.documents[document_id] = entry
    # A shard the manifest vouches for but the directory lost is rewritten.
    on_disk = all(
        _shard_path(staged.output_dir, kind, shard).is_file()
        for kind, rows in _shard_files(entry)
        if rows
    )
    if on_disk and staged.prior.documents.get(document_id) == entry:
        staged.unchanged += 1
        return
    staged.changed.append(document_id)
    for kind, rows in _shard_files(entry):
        if rows:
            target = _shard_path(staged.stage_dir, kind, shard)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payloads[kind])


def _refuse_symlinked_trees(output_dir: Path) -> None:
    """Fail before any write or removal if a shard tree or a fan-out directory
    is a symlink: every path below them would lead outside the export."""
    for kind in (NODES_DIRNAME, EDGES_DIRNAME):
        base = output_dir / kind
        if base.is_symlink():
            raise GraphExportError(
                f"{base} is a symlink; the export never follows symlinks. "
                "Remove it or export to another directory"
            )
        if not base.is_dir():
            continue
        for prefix in base.iterdir():
            if _PREFIX_NAME.fullmatch(prefix.name) and prefix.is_symlink():
                raise GraphExportError(
                    f"{prefix} is a symlink; the export never follows symlinks. "
                    "Remove it or export to another directory"
                )


def _sweep_unnamed(output_dir: Path, keep: set[Path]) -> None:
    """Remove shard-named files no manifest names, after a crash left some.

    Only files shaped like a shard, in a two-hex-character directory of the two
    shard trees, are ever removed; anything else the operator keeps there is
    never looked at.
    """
    for kind in (NODES_DIRNAME, EDGES_DIRNAME):
        base = output_dir / kind
        if not base.is_dir():
            continue
        for prefix in sorted(base.iterdir()):
            if not (prefix.is_dir() and _PREFIX_NAME.fullmatch(prefix.name)):
                continue
            for candidate in sorted(prefix.iterdir()):
                if (
                    _SHARD_NAME.fullmatch(candidate.name)
                    and candidate.is_file()
                    and not candidate.is_symlink()
                    and candidate not in keep
                ):
                    candidate.unlink()


def _manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _replace_shards(staged: Staged, stale: Sequence[str]) -> None:
    output_dir = staged.output_dir
    for document_id in staged.changed:
        entry = staged.documents[document_id]
        for kind, rows in _shard_files(entry):
            final = _shard_path(output_dir, kind, entry["shard"])
            if rows:
                final.parent.mkdir(parents=True, exist_ok=True)
                _shard_path(staged.stage_dir, kind, entry["shard"]).replace(final)
            else:
                final.unlink(missing_ok=True)
    for document_id in stale:
        entry = staged.prior.documents[document_id]
        for kind, _rows in _shard_files(entry):
            _shard_path(output_dir, kind, entry["shard"]).unlink(missing_ok=True)
    if not staged.prior.present:
        # No manifest to say what is ours: a crash may have left shards behind.
        keep = {
            _shard_path(output_dir, kind, entry["shard"])
            for entry in staged.documents.values()
            for kind, rows in _shard_files(entry)
            if rows
        }
        _sweep_unnamed(output_dir, keep)


def commit_graph(staged: Staged) -> GraphExportResult:
    """Make a staged export visible: replace the changed shards, remove the stale
    ones, then write the manifest last, so a directory that has one is complete.

    A document is stale when the previous manifest names it and this read did
    not see it (or saw it with nothing to export). Removal happens here, after
    the read succeeded, never during it.
    """
    manifest_path = staged.output_dir / GRAPH_MANIFEST_FILENAME
    payload = _manifest_bytes(staged.manifest)
    stale = sorted(set(staged.prior.documents) - set(staged.documents))
    result = GraphExportResult(
        output_dir=staged.output_dir,
        documents=len(staged.documents),
        node_rows=staged.manifest["totals"]["node_rows"],
        edges=staged.manifest["totals"]["edges"],
        edges_skipped=dict(staged.skipped),
        shards_written=len(staged.changed),
        shards_unchanged=staged.unchanged,
        shards_removed=len(stale),
    )
    try:
        if staged.changed or stale or staged.prior.legacy_files or not staged.prior.present:
            # A manifest left by an earlier export would vouch for files about
            # to be replaced, so it goes first. A v1/v2 manifest vouches only
            # for the two whole files, which nothing here replaces, so it stays
            # until the v3 manifest takes its place.
            if not staged.prior.legacy_files:
                manifest_path.unlink(missing_ok=True)
            _replace_shards(staged, stale)
            _write_file(manifest_path, payload)
            # Only now is the earlier export superseded.
            for name in staged.prior.legacy_files:
                (staged.output_dir / name).unlink(missing_ok=True)
        elif manifest_path.read_bytes() != payload:
            # Only the header moved (a model renamed, a skip count): no shard did.
            _write_file(manifest_path, payload)
    finally:
        staged.discard()
    return result


def write_graph_export(
    documents: Iterable[DocumentRows],
    output_dir: Path,
    *,
    project: str,
    linking_model: str,
    relation_model: str,
    context: GraphContext,
) -> GraphExportResult:
    """Stage and commit an export of an already-joined document stream."""
    staged = stage_graph(
        documents,
        output_dir,
        project=project,
        linking_model=linking_model,
        relation_model=relation_model,
        context=context,
    )
    return commit_graph(staged)


def _rows(snapshot: TableReadSnapshot) -> Iterator[dict[str, Any]]:
    for batch in snapshot:
        yield from batch.to_pylist()


def export_graph(
    project_dir: str | Path,
    *,
    linking_model: str,
    relation_model: str,
    entity_name: str,
    output_dir: Path,
    target: str | None,
    profiles_dir: str | Path | None,
) -> GraphExportResult:
    """Read the two models through the active adapter and write the graph.

    Both tables stream through `table_snapshot`, ordered by `document_id`, so
    no whole-table frame is built. The snapshots are validated when their
    block exits and the files are committed only after that, so a table that
    changed under the read leaves the previous export in place.
    """
    from .adapters import create_adapter
    from .profile import resolve_profile

    # Checked before the project, profile or warehouse is touched.
    entity_name = _clean_entity_name(entity_name)
    project_path = Path(project_dir)
    project, _, _ = load_project(project_path)
    context = GraphContext.create(entity_name, project.vocabularies, project.classes)
    resolved = resolve_profile(
        project, project_path, target=target,
        profiles_dir=Path(profiles_dir) if profiles_dir else None,
    )
    staged: Staged | None = None
    try:
        with create_adapter(resolved.warehouse, project_dir=project_path) as adapter:
            link_available = adapter.read_table_schema(linking_model).columns
            relation_available = adapter.read_table_schema(relation_model).columns
            require_columns(link_available, LINK_COLUMNS, "linking model")
            require_columns(relation_available, RELATION_COLUMNS, "relation model")
            relation_columns = (
                *RELATION_COLUMNS,
                *(name for name in _OPTIONAL_RELATION_COLUMNS if name in relation_available),
            )
            with ExitStack() as stack:
                # Ordered by document so one document at a time is resident (the
                # bounded-memory contract). The key check refuses a repeated or
                # null id before the first row, which also makes the order total.
                try:
                    links = stack.enter_context(
                        adapter.table_snapshot(
                            linking_model,
                            columns=LINK_COLUMNS,
                            batch_size=_READ_BATCH_ROWS,
                            order_by=("document_id", "entity_link_id"),
                            key_column="entity_link_id",
                        )
                    )
                    relations = stack.enter_context(
                        adapter.table_snapshot(
                            relation_model,
                            columns=relation_columns,
                            batch_size=_READ_BATCH_ROWS,
                            order_by=("document_id", "relation_id"),
                            key_column="relation_id",
                        )
                    )
                except AdapterError as error:
                    raise GraphExportError(
                        "could not read the linking and relation models for the "
                        "graph export (entity_link_id and relation_id must each "
                        f"be unique and non-null): {error}"
                    ) from error
                staged = stage_graph(
                    merge_documents(_rows(links), _rows(relations)),
                    output_dir,
                    project=project.name,
                    linking_model=linking_model,
                    relation_model=relation_model,
                    context=context,
                )
        return commit_graph(staged)
    finally:
        if staged is not None:
            staged.discard()
