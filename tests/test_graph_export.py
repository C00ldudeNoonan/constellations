"""Graph export contract (issue #762, ALE-423, ADR-0055; incremental, ADR-0061).

The pure per-document builder, the two-stream join and the incremental file
writer, over hand-built frames. The demo project runs through the CLI in
`test_demo_template.py`, and the bounded read in `test_bounded_memory.py`.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from stel import graph_export
from stel.agent_context import project_entity_link
from stel.config.class_hierarchy import ClassDef
from stel.config.vocabulary import Vocabulary
from stel.graph_export import (
    EDGES_DIRNAME,
    GRAPH_EXPORT_SCHEMA,
    GRAPH_MANIFEST_FILENAME,
    LINK_COLUMNS,
    NODES_DIRNAME,
    RELATION_COLUMNS,
    GraphExportError,
    build_document,
    merge_documents,
    require_columns,
    write_graph_export,
)
from tests.support_graph_export import (
    context as _context,
)
from tests.support_graph_export import (
    documents as _documents,
)
from tests.support_graph_export import (
    export as _export,
)
from tests.support_graph_export import (
    links_frame as _links,
)
from tests.support_graph_export import (
    relations_frame as _relations,
)
from tests.support_graph_export import (
    snapshot as _snapshot,
)

_RECORDED = datetime(2026, 1, 1)


def _build(links: pl.DataFrame, relations: pl.DataFrame, **kwargs: Any) -> Any:
    """Every document's shard merged into one graph, for the builder's rules."""
    require_columns(links.columns, LINK_COLUMNS, "linking model")
    require_columns(relations.columns, RELATION_COLUMNS, "relation model")
    context = _context(**kwargs)
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}
    skipped = {"not_asserted": 0, "unlinked_mention": 0, "self_relation": 0}
    for _document, document_links, document_relations in _documents(links, relations):
        graph = build_document(
            links=document_links, relations=document_relations, context=context
        )
        nodes.update({node["entity_id"]: node for node in graph.nodes})
        edges.update({edge["relation_id"]: edge for edge in graph.edges})
        for reason, count in graph.edges_skipped.items():
            skipped[reason] += count
    return SimpleNamespace(
        nodes=tuple(nodes[key] for key in sorted(nodes)),
        edges=tuple(edges[key] for key in sorted(edges)),
        edges_skipped=skipped,
    )


def test_node_ids_are_the_agent_context_entity_id() -> None:
    """The export's whole value is that a node joins to a governed metric, so
    its id is the one `project_entity_link` writes for the same entity."""
    # A name other than the fixture default, so a node id that ignored the
    # option could not match by coincidence.
    graph = _build(
        _links([{"mention_id": "m1", "canonical_id": "Postgres"}]),
        _relations([]),
        entity_name="datastore",
    )
    expected = project_entity_link(
        context_id="a" * 32,
        entity_namespace="systems",
        entity_name="datastore",
        canonical_id="Postgres",
        relationship_type="mentions",
        link_method="x:1",
        recorded_from=_RECORDED,
    )["entity_id"]
    assert [node["entity_id"] for node in graph.nodes] == [expected]


def test_a_different_entity_name_gives_ids_that_do_not_join() -> None:
    links = _links([{"mention_id": "m1", "canonical_id": "Postgres"}])
    a = _build(links, _relations([]), entity_name="system")
    b = _build(links, _relations([]), entity_name="service")
    assert a.nodes[0]["entity_id"] != b.nodes[0]["entity_id"]


def test_every_edge_names_its_document_and_extractor_version() -> None:
    graph = _build(
        _links(
            [
                {
                    "mention_id": "m1",
                    "canonical_id": "A",
                    "resolver_version": "7",
                    "document_id": "doc-9",
                },
                {"mention_id": "m2", "canonical_id": "B", "document_id": "doc-9"},
            ]
        ),
        _relations(
            [
                {
                    "relation_id": "r1",
                    "subject_mention_id": "m1",
                    "object_mention_id": "m2",
                    "document_id": "doc-9",
                    "extractor_version": "3",
                }
            ]
        ),
    )
    (edge,) = graph.edges
    assert edge["relation_id"] == "r1"
    assert edge["document_id"] == "doc-9"
    assert edge["extractor"] == "co_occurrence"
    assert edge["extractor_version"] == "3"
    by_entity = {node["entity_id"]: node["canonical_id"] for node in graph.nodes}
    assert {by_entity[edge["subject_entity_id"]], by_entity[edge["object_entity_id"]]} == {
        "A",
        "B",
    }
    # Which resolver linked which end travels with the edge.
    assert {edge["subject_resolver_version"], edge["object_resolver_version"]} == {"7", "1"}


def test_an_undirected_pair_is_written_in_entity_id_order_either_way_round() -> None:
    links = _links(
        [
            {"mention_id": "m1", "canonical_id": "A"},
            {"mention_id": "m2", "canonical_id": "B"},
        ]
    )
    forward = _build(
        links,
        _relations(
            [{"relation_id": "r", "subject_mention_id": "m1", "object_mention_id": "m2"}]
        ),
    )
    backward = _build(
        links,
        _relations(
            [{"relation_id": "r", "subject_mention_id": "m2", "object_mention_id": "m1"}]
        ),
    )
    assert forward.edges == backward.edges
    edge = forward.edges[0]
    assert edge["subject_entity_id"] < edge["object_entity_id"]


def test_a_directed_relation_keeps_its_direction() -> None:
    links = _links(
        [
            {"mention_id": "m1", "canonical_id": "A"},
            {"mention_id": "m2", "canonical_id": "B"},
        ]
    )
    graph = _build(
        links,
        _relations(
            [
                {
                    "relation_id": "r",
                    "subject_mention_id": "m2",
                    "object_mention_id": "m1",
                    "directed": True,
                    "relation_type": "depends_on",
                }
            ]
        ),
    )
    by_entity = {node["entity_id"]: node["canonical_id"] for node in graph.nodes}
    edge = graph.edges[0]
    assert (by_entity[edge["subject_entity_id"]], by_entity[edge["object_entity_id"]]) == (
        "B",
        "A",
    )


def test_what_the_graph_leaves_out_is_counted_by_reason() -> None:
    graph = _build(
        _links(
            [
                {"mention_id": "m1", "canonical_id": "A"},
                {"mention_id": "m2", "canonical_id": "B"},
                {"mention_id": "m3", "canonical_id": "A"},
                {"mention_id": "m4", "canonical_id": "C", "status": "ambiguous"},
            ]
        ),
        _relations(
            [
                {"relation_id": "ok", "subject_mention_id": "m1", "object_mention_id": "m2"},
                {
                    "relation_id": "maybe",
                    "subject_mention_id": "m1",
                    "object_mention_id": "m2",
                    "status": "ambiguous",
                },
                # An ambiguous link names no single entity, so this end is unlinked.
                {"relation_id": "amb", "subject_mention_id": "m1", "object_mention_id": "m4"},
                {"relation_id": "gone", "subject_mention_id": "m1", "object_mention_id": "zz"},
                # m1 and m3 are two mentions of the same entity.
                {"relation_id": "self", "subject_mention_id": "m1", "object_mention_id": "m3"},
            ]
        ),
    )
    assert [edge["relation_id"] for edge in graph.edges] == ["ok"]
    assert graph.edges_skipped == {
        "not_asserted": 1,
        "unlinked_mention": 2,
        "self_relation": 1,
    }
    # The ambiguous link's entity is not a node either.
    assert sorted(node["canonical_id"] for node in graph.nodes) == ["A", "B"]


def test_a_node_carries_its_declared_class_and_ancestors() -> None:
    vocabularies = {
        "systems": Vocabulary.model_validate(
            {"terms": [{"label": "Redis", "class": "cache"}, {"label": "Other"}]}
        )
    }
    classes = ("datastore", ClassDef(name="cache", subclass_of="datastore"))
    graph = _build(
        _links(
            [
                {"mention_id": "m1", "canonical_id": "Redis"},
                {"mention_id": "m2", "canonical_id": "Other"},
                {"mention_id": "m3", "canonical_id": "Undeclared"},
            ]
        ),
        _relations([]),
        vocabularies=vocabularies,
        classes=classes,
    )
    nodes = {node["canonical_id"]: node for node in graph.nodes}
    assert (nodes["Redis"]["class"], nodes["Redis"]["class_ancestors"]) == (
        "cache",
        ["datastore"],
    )
    assert (nodes["Other"]["class"], nodes["Other"]["class_ancestors"]) == (None, [])
    assert nodes["Undeclared"]["class"] is None


def test_a_mention_matched_to_two_entities_fails_rather_than_picking_one() -> None:
    with pytest.raises(GraphExportError, match="matched to two different entities"):
        _build(
            _links(
                [
                    {"mention_id": "m1", "canonical_id": "A"},
                    {"mention_id": "m1", "canonical_id": "B"},
                ]
            ),
            _relations([]),
        )


def test_a_duplicate_relation_id_fails() -> None:
    links = _links(
        [
            {"mention_id": "m1", "canonical_id": "A"},
            {"mention_id": "m2", "canonical_id": "B"},
        ]
    )
    row = {"relation_id": "r", "subject_mention_id": "m1", "object_mention_id": "m2"}
    with pytest.raises(GraphExportError, match="appears twice"):
        _build(links, _relations([row, row]))


def test_a_model_without_the_provenance_columns_is_refused_by_name() -> None:
    links = _links([{"mention_id": "m1", "canonical_id": "A"}])
    with pytest.raises(GraphExportError, match=r"relation model is missing.*extractor_version"):
        _build(links, _relations([]).drop("extractor_version"))
    with pytest.raises(GraphExportError, match=r"linking model is missing.*resolver"):
        _build(links.drop("resolver"), _relations([]))


def test_an_empty_entity_name_is_refused() -> None:
    with pytest.raises(GraphExportError, match="entity_name"):
        _build(_links([]), _relations([]), entity_name="  ")


def _corpus(
    documents: int, *, skip: tuple[int, ...] = (), confidence: dict[int, float] | None = None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """`documents` documents, each linking two systems and relating them once."""
    link_rows: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    for index in range(documents):
        if index in skip:
            continue
        document = f"doc{index}"
        link_rows += [
            {"document_id": document, "mention_id": f"{document}-a", "canonical_id": f"A{index}"},
            {"document_id": document, "mention_id": f"{document}-b", "canonical_id": f"B{index}"},
        ]
        relation_rows.append(
            {
                "relation_id": f"r-{document}",
                "document_id": document,
                "subject_mention_id": f"{document}-a",
                "object_mention_id": f"{document}-b",
                "confidence": (confidence or {}).get(index, 0.5),
            }
        )
    return _links(link_rows), _relations(relation_rows)


def _shard(out: Path, document: str, kind: str) -> Path:
    shard = graph_export._shard_id(document)
    return out / kind / shard[:2] / f"{shard}.jsonl"


def _two_node_graph(out: Path) -> Any:
    links = _links(
        [
            {"mention_id": "m1", "canonical_id": "Café"},
            {"mention_id": "m2", "canonical_id": "B"},
        ]
    )
    relations = _relations(
        [
            {
                "relation_id": "r",
                "subject_mention_id": "m1",
                "object_mention_id": "m2",
                "confidence": 0.5,
            }
        ]
    )
    return _export(out, links, relations)


def test_a_document_is_two_shard_files_named_by_a_hash_of_its_id(tmp_path: Path) -> None:
    out = tmp_path / "g"
    result = _two_node_graph(out)

    nodes_path = _shard(out, "d1", NODES_DIRNAME)
    edges_path = _shard(out, "d1", EDGES_DIRNAME)
    assert set(_snapshot(out)) == {
        GRAPH_MANIFEST_FILENAME,
        nodes_path.relative_to(out).as_posix(),
        edges_path.relative_to(out).as_posix(),
    }
    nodes = [json.loads(line) for line in nodes_path.read_text(encoding="utf-8").splitlines()]
    assert "Café" in {node["canonical_id"] for node in nodes}
    assert len(edges_path.read_text(encoding="utf-8").splitlines()) == 1
    assert (result.documents, result.node_rows, result.edges) == (1, 2, 1)

    manifest = json.loads((out / GRAPH_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["schema"] == GRAPH_EXPORT_SCHEMA
    assert manifest["entity_name"] == "system"
    assert manifest["totals"] == {"documents": 1, "node_rows": 2, "edges": 1}
    entry = manifest["documents"]["d1"]
    assert (entry["nodes"], entry["edges"]) == (2, 1)
    assert entry["shard"] == graph_export._shard_id("d1")
    # No timestamp: identical inputs must give identical bytes.
    assert not any("time" in key or "generated" in key for key in manifest)


def test_two_exports_of_unchanged_tables_write_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ALE-601: an unchanged source costs a read, not a rewrite. Every write the
    writer makes goes through `write_bytes`, so none being made is the claim."""
    out = tmp_path / "g"
    links, relations = _corpus(4)
    first = _export(out, links, relations)
    before = _snapshot(out)
    assert first.shards_written == 4

    written: list[str] = []
    real_write = Path.write_bytes

    def spy(self: Path, data: Any) -> int:
        written.append(self.name)
        return real_write(self, data)

    monkeypatch.setattr(Path, "write_bytes", spy)
    again = _export(out, links, relations)

    assert written == []
    assert (again.shards_written, again.shards_unchanged, again.shards_removed) == (0, 4, 0)
    assert _snapshot(out) == before


def test_a_changed_document_rewrites_only_its_shard_and_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ALE-601 done-when: one document moves, one shard and the manifest are
    written, and every other document's files keep their bytes."""
    out = tmp_path / "g"
    links, relations = _corpus(5)
    _export(out, links, relations)
    before = _snapshot(out)

    written: list[str] = []
    real_write = Path.write_bytes

    def spy(self: Path, data: Any) -> int:
        written.append(self.name)
        return real_write(self, data)

    monkeypatch.setattr(Path, "write_bytes", spy)
    links, relations = _corpus(5, confidence={2: 0.9})
    result = _export(out, links, relations)

    shard = f"{graph_export._shard_id('doc2')}.jsonl"
    assert sorted(written) == sorted([shard, shard, "graph_manifest.json.tmp"])
    assert (result.shards_written, result.shards_unchanged) == (1, 4)
    after = _snapshot(out)
    changed = {name for name in after if after[name] != before.get(name)}
    assert changed == {
        GRAPH_MANIFEST_FILENAME,
        _shard(out, "doc2", EDGES_DIRNAME).relative_to(out).as_posix(),
    }
    moved = json.loads(_shard(out, "doc2", EDGES_DIRNAME).read_text(encoding="utf-8"))
    assert moved["confidence"] == 0.9


def test_a_removed_document_loses_its_shards_and_its_manifest_entry(tmp_path: Path) -> None:
    """ALE-601 done-when: a document that left the source has its edges removed
    from the export, and nothing else moves."""
    out = tmp_path / "g"
    _export(out, *_corpus(4))
    before = _snapshot(out)
    gone = [_shard(out, "doc1", kind) for kind in (NODES_DIRNAME, EDGES_DIRNAME)]
    assert all(path.is_file() for path in gone)

    result = _export(out, *_corpus(4, skip=(1,)))

    assert not any(path.exists() for path in gone)
    assert result.shards_removed == 1
    manifest = json.loads((out / GRAPH_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert sorted(manifest["documents"]) == ["doc0", "doc2", "doc3"]
    assert manifest["totals"] == {"documents": 3, "node_rows": 6, "edges": 3}
    after = _snapshot(out)
    kept = {
        name: payload
        for name, payload in before.items()
        if name != GRAPH_MANIFEST_FILENAME and graph_export._shard_id("doc1") not in name
    }
    assert {name: after[name] for name in kept} == kept


def test_an_empty_source_makes_every_shard_stale(tmp_path: Path) -> None:
    """The reconciler's rule (#153): absence is complete even for an empty
    upstream, so an export of nothing leaves no document behind."""
    out = tmp_path / "g"
    _export(out, *_corpus(3))

    result = _export(out, *_corpus(0))

    assert result.shards_removed == 3
    assert [p for p in _snapshot(out) if p != GRAPH_MANIFEST_FILENAME] == []
    manifest = json.loads((out / GRAPH_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["documents"] == {}


def test_a_document_that_keeps_nodes_but_loses_its_edges_loses_only_the_edge_file(
    tmp_path: Path,
) -> None:
    out = tmp_path / "g"
    links, relations = _corpus(2)
    _export(out, links, relations)
    edges_path = _shard(out, "doc0", EDGES_DIRNAME)
    assert edges_path.is_file()

    # The relation is no longer asserted, so the document still links two
    # entities and asserts nothing between them.
    relations = relations.with_columns(
        pl.when(pl.col("document_id") == "doc0")
        .then(pl.lit("ambiguous"))
        .otherwise(pl.col("status"))
        .alias("status")
    )
    result = _export(out, links, relations)

    assert not edges_path.exists()
    assert _shard(out, "doc0", NODES_DIRNAME).is_file()
    assert result.edges_skipped["not_asserted"] == 1
    entry = json.loads((out / GRAPH_MANIFEST_FILENAME).read_text(encoding="utf-8"))["documents"]
    assert (entry["doc0"]["nodes"], entry["doc0"]["edges"]) == (2, 0)


def test_a_document_with_nothing_to_export_is_not_a_shard(tmp_path: Path) -> None:
    """A relation whose mentions no link resolves is skipped and counted, and
    the document it names gets no shard and no manifest entry."""
    out = tmp_path / "g"
    relations = _relations(
        [
            {
                "relation_id": "r",
                "document_id": "orphan",
                "subject_mention_id": "x",
                "object_mention_id": "y",
            }
        ]
    )
    result = _export(out, _links([]), relations)

    assert result.edges_skipped["unlinked_mention"] == 1
    assert [p for p in _snapshot(out) if p != GRAPH_MANIFEST_FILENAME] == []


def test_a_shard_the_manifest_vouches_for_but_the_directory_lost_is_rewritten(
    tmp_path: Path,
) -> None:
    out = tmp_path / "g"
    links, relations = _corpus(3)
    _export(out, links, relations)
    before = _snapshot(out)
    _shard(out, "doc1", EDGES_DIRNAME).unlink()

    result = _export(out, links, relations)

    assert result.shards_written == 1
    assert _snapshot(out) == before


def test_a_failed_read_leaves_the_previous_export_untouched(tmp_path: Path) -> None:
    """The directory is what a consumer loads from, so a run that fails partway
    must not leave it half-updated, whichever way it fails."""
    out = tmp_path / "g"
    _export(out, *_corpus(4))
    before = _snapshot(out)

    def failing() -> Iterator[Any]:
        links, relations = _corpus(4, confidence={0: 0.1, 1: 0.2})
        for index, document in enumerate(_documents(links, relations)):
            if index == 2:
                raise RuntimeError("warehouse went away")
            yield document

    with pytest.raises(RuntimeError, match="went away"):
        write_graph_export(
            failing(),
            out,
            project="p",
            linking_model="l",
            relation_model="r",
            context=_context(),
            graph_format="jsonl",
        )
    assert _snapshot(out) == before

    # A document that fails validation does the same.
    links, relations = _corpus(4, skip=(3,))
    conflicted = pl.concat(
        [
            links,
            _links(
                [
                    {"document_id": "doc0", "mention_id": "doc0-a", "canonical_id": "Z"},
                ]
            ).with_columns(pl.lit("lz").alias("entity_link_id")),
        ]
    )
    with pytest.raises(GraphExportError, match="matched to two different entities"):
        _export(out, conflicted, relations)
    assert _snapshot(out) == before


def test_a_failed_commit_leaves_no_manifest_and_no_scratch_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest from an earlier export must not vouch for half-replaced files."""
    out = tmp_path / "g"
    links, relations = _corpus(2)
    _export(out, links, relations)
    assert (out / GRAPH_MANIFEST_FILENAME).exists()

    real_replace = Path.replace

    def fail_on_shards(self: Path, target: Any) -> Any:
        if Path(target).parent.parent.name == EDGES_DIRNAME:
            raise OSError("disk full")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_on_shards)
    links, relations = _corpus(2, confidence={0: 0.9})
    with pytest.raises(OSError, match="disk full"):
        _export(out, links, relations)
    assert not (out / GRAPH_MANIFEST_FILENAME).exists()
    assert not list(out.rglob("*.tmp"))
    assert not (out / graph_export._STAGE_DIRNAME).exists()


def test_a_crash_that_cost_the_manifest_is_swept_on_the_next_export(tmp_path: Path) -> None:
    """With no manifest to say what is ours, shard-shaped files nobody names are
    removed, and nothing else in the directory is looked at."""
    out = tmp_path / "g"
    _export(out, *_corpus(3))
    stranded = _shard(out, "doc1", EDGES_DIRNAME)
    (out / GRAPH_MANIFEST_FILENAME).unlink()
    keepsake = stranded.parent / "notes.txt"
    keepsake.write_text("mine", encoding="utf-8")
    elsewhere = out / "README.md"
    elsewhere.write_text("mine too", encoding="utf-8")

    _export(out, *_corpus(3, skip=(1,)))

    assert not stranded.exists()
    assert not _shard(out, "doc1", NODES_DIRNAME).exists()
    assert keepsake.read_text(encoding="utf-8") == "mine"
    assert elsewhere.read_text(encoding="utf-8") == "mine too"
    assert _shard(out, "doc0", EDGES_DIRNAME).is_file()


def _legacy_export(out: Path) -> dict[str, bytes]:
    out.mkdir(parents=True)
    (out / "nodes.jsonl").write_bytes(b'{"entity_id":"old"}\n')
    (out / "edges.jsonl").write_bytes(b'{"relation_id":"old"}\n')
    (out / GRAPH_MANIFEST_FILENAME).write_text(
        json.dumps({"schema": "stel.graph_export/v2", "files": {}}), encoding="utf-8"
    )
    return _snapshot(out)


def test_a_v2_export_is_left_alone_until_the_v3_export_has_committed(
    tmp_path: Path,
) -> None:
    """The directory an earlier release wrote is the user's data: a run that
    fails does not touch it, and one that succeeds replaces it whole."""
    out = tmp_path / "g"
    legacy = _legacy_export(out)
    links, relations = _corpus(1)
    with pytest.raises(GraphExportError, match="appears twice"):
        _export(out, links, pl.concat([relations, relations]))
    assert _snapshot(out) == legacy

    result = _export(out, *_corpus(2))

    assert result.shards_written == 2
    assert not (out / "nodes.jsonl").exists()
    assert not (out / "edges.jsonl").exists()
    manifest = json.loads((out / GRAPH_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["schema"] == GRAPH_EXPORT_SCHEMA


def test_a_v2_export_survives_a_failed_manifest_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The earlier export is retained until the v3 manifest is published, so a
    write that fails at the last step (a full disk) does not cost the user both."""
    out = tmp_path / "g"
    legacy = _legacy_export(out)
    real_write = graph_export._write_file

    def fail_manifest(path: Path, payload: bytes) -> None:
        if path.name == GRAPH_MANIFEST_FILENAME:
            raise OSError("disk full")
        real_write(path, payload)

    monkeypatch.setattr(graph_export, "_write_file", fail_manifest)
    with pytest.raises(OSError, match="disk full"):
        _export(out, *_corpus(2))

    after = _snapshot(out)
    assert {name: after.get(name) for name in legacy} == legacy


def _stranded_shard(root: Path, kind: str) -> Path:
    prefix = root / kind / "aa"
    prefix.mkdir(parents=True)
    path = prefix / f"{'a' * 32}.jsonl"
    path.write_bytes(b"{}")
    return path


def test_a_symlinked_shard_directory_is_refused_before_anything_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recovery sweep removes shard-shaped files, so it must never follow a
    link out of the export. `Path.is_symlink` is stood in for here because
    creating a real link needs a privilege some machines lack."""
    out = tmp_path / "g"
    outside = _stranded_shard(tmp_path / "elsewhere", NODES_DIRNAME)
    inside = _stranded_shard(out, EDGES_DIRNAME)
    real = Path.is_symlink
    monkeypatch.setattr(
        Path, "is_symlink", lambda self: self.name == "aa" or real(self)
    )

    with pytest.raises(GraphExportError, match="symlink"):
        _export(out, *_corpus(0))

    assert outside.is_file()
    assert inside.is_file()


@pytest.mark.parametrize("name", [NODES_DIRNAME, graph_export._STAGE_DIRNAME])
def test_a_symlinked_tree_or_scratch_directory_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """The shard trees and the scratch directory are written through, so a link
    in either place leads outside the export. Only the link itself is stood in
    for (see the test above); the directory behind it is real, and the stranded
    shard inside the tree is what the recovery sweep would remove."""
    out = tmp_path / "g"
    inside = _stranded_shard(out, NODES_DIRNAME)
    (out / graph_export._STAGE_DIRNAME).mkdir()
    real = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda self: self.name == name or real(self))

    with pytest.raises(GraphExportError, match="symlink"):
        _export(out, *_corpus(0))

    assert inside.is_file()
    assert not any((out / graph_export._STAGE_DIRNAME).iterdir())


def test_legacy_files_a_v3_manifest_outlived_are_removed_by_the_next_run(
    tmp_path: Path,
) -> None:
    """A run killed after writing the v3 manifest and before removing the old
    whole files leaves both; the next run, which has nothing to write, still
    finishes the removal."""
    out = tmp_path / "g"
    _export(out, *_corpus(2))
    (out / "nodes.jsonl").write_bytes(b"{}")
    (out / "edges.jsonl").write_bytes(b"{}")

    result = _export(out, *_corpus(2))

    assert result.shards_written == 0
    assert not (out / "nodes.jsonl").exists()
    assert not (out / "edges.jsonl").exists()


def test_a_manifest_scratch_file_a_hard_kill_left_is_removed(tmp_path: Path) -> None:
    out = tmp_path / "g"
    _export(out, *_corpus(1))
    leftover = out / (GRAPH_MANIFEST_FILENAME + ".tmp")
    leftover.write_bytes(b"{")

    _export(out, *_corpus(1))

    assert not leftover.exists()


def test_a_real_symlinked_shard_directory_leaves_its_target_alone(tmp_path: Path) -> None:
    out = tmp_path / "g"
    outside = _stranded_shard(tmp_path / "elsewhere", NODES_DIRNAME)
    (out / NODES_DIRNAME).mkdir(parents=True)
    try:
        (out / NODES_DIRNAME / "aa").symlink_to(outside.parent, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are not available here")

    with pytest.raises(GraphExportError, match="symlink"):
        _export(out, *_corpus(0))

    assert outside.is_file()


@pytest.mark.parametrize(
    "manifest",
    [
        "{not json",
        json.dumps({"schema": "stel.graph_export/v9", "documents": {}}),
        json.dumps({"schema": GRAPH_EXPORT_SCHEMA}),
        json.dumps({"schema": GRAPH_EXPORT_SCHEMA, "documents": {"d": {"nodes": 1}}}),
    ],
    ids=["unreadable", "newer-schema", "no-documents", "malformed-entry"],
)
def test_a_manifest_it_cannot_vouch_for_is_refused_and_left_in_place(
    tmp_path: Path, manifest: str
) -> None:
    out = tmp_path / "g"
    out.mkdir()
    (out / GRAPH_MANIFEST_FILENAME).write_text(manifest, encoding="utf-8")
    (out / "mine.txt").write_text("keep", encoding="utf-8")
    before = _snapshot(out)

    with pytest.raises(GraphExportError, match="refusing to overwrite"):
        _export(out, *_corpus(2))
    assert _snapshot(out) == before


def test_a_header_only_change_rewrites_the_manifest_and_no_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "g"
    links, relations = _corpus(3)
    _export(out, links, relations)
    before = _snapshot(out)

    written: list[str] = []
    real_write = Path.write_bytes

    def spy(self: Path, data: Any) -> int:
        written.append(self.name)
        return real_write(self, data)

    monkeypatch.setattr(Path, "write_bytes", spy)
    result = _export(out, links, relations, relation_model="renamed")

    assert written == ["graph_manifest.json.tmp"]
    assert result.shards_written == 0
    after = _snapshot(out)
    assert {name for name in after if after[name] != before[name]} == {GRAPH_MANIFEST_FILENAME}


def test_a_changed_entity_name_rewrites_every_shard(tmp_path: Path) -> None:
    """Node ids are derived from `entity_name`, so a new name moves every shard's
    bytes and the fingerprint notices without being told the context changed."""
    out = tmp_path / "g"
    links, relations = _corpus(3)
    _export(out, links, relations)

    result = _export(out, links, relations, entity_name="service")

    assert result.shards_written == 3


def test_a_changed_class_declaration_rewrites_the_shards_it_touches(tmp_path: Path) -> None:
    out = tmp_path / "g"
    links, relations = _corpus(3)
    _export(out, links, relations)

    vocabularies = {
        "systems": Vocabulary.model_validate({"terms": [{"label": "A1", "class": "cache"}]})
    }
    result = _export(out, links, relations, vocabularies=vocabularies, classes=("cache",))

    assert result.shards_written == 1
    nodes = _shard(out, "doc1", NODES_DIRNAME).read_text(encoding="utf-8")
    assert '"class":"cache"' in nodes


# --- the two-stream join ----------------------------------------------------


def _link_row(document: str, mention: str) -> dict[str, Any]:
    return {"document_id": document, "mention_id": mention}


def test_the_join_pairs_each_document_with_its_own_rows_from_either_table() -> None:
    links = [_link_row("a", "m1"), _link_row("a", "m2"), _link_row("c", "m3")]
    relations = [_link_row("a", "r1"), _link_row("b", "r2"), _link_row("c", "r3")]

    joined = [
        (document, [row["mention_id"] for row in left], [row["mention_id"] for row in right])
        for document, left, right in merge_documents(links, relations)
    ]

    assert joined == [
        ("a", ["m1", "m2"], ["r1"]),
        ("b", [], ["r2"]),
        ("c", ["m3"], ["r3"]),
    ]


def test_the_join_handles_one_table_being_empty() -> None:
    only_links = list(merge_documents([_link_row("a", "m1")], []))
    only_relations = list(merge_documents([], [_link_row("a", "r1")]))
    assert [(d, len(left), len(right)) for d, left, right in only_links] == [("a", 1, 0)]
    assert [(d, len(left), len(right)) for d, left, right in only_relations] == [("a", 0, 1)]
    assert list(merge_documents([], [])) == []


@pytest.mark.parametrize("side", ["links", "relations"])
def test_a_stream_out_of_document_order_fails_instead_of_mispairing(side: str) -> None:
    """The join trusts nothing it cannot check: a warehouse collation that
    disagrees with Python's would otherwise pair the wrong documents silently."""
    ordered = [_link_row("a", "x"), _link_row("c", "y")]
    shuffled = [_link_row("b", "x"), _link_row("a", "y")]
    streams = (shuffled, ordered) if side == "links" else (ordered, shuffled)
    with pytest.raises(GraphExportError, match="ascending document_id order"):
        list(merge_documents(*streams))


def test_a_stream_that_returns_to_a_document_it_left_fails() -> None:
    rows = [_link_row("a", "x"), _link_row("b", "y"), _link_row("a", "z")]
    with pytest.raises(GraphExportError, match="ascending document_id order"):
        list(merge_documents(rows, []))


@pytest.mark.parametrize("missing", [None, ""])
def test_a_row_with_no_document_id_is_refused(missing: str | None) -> None:
    with pytest.raises(GraphExportError, match="no document_id"):
        list(merge_documents([_link_row("a", "x")], [{"document_id": missing}]))


def test_a_link_without_a_namespace_is_not_a_node() -> None:
    links = _links([{"mention_id": "m1", "canonical_id": "A"}])
    links = links.with_columns(pl.lit(None, dtype=pl.String()).alias("entity_namespace"))
    graph = _build(links, _relations([]))
    assert graph.nodes == ()


def test_a_null_resolver_stays_null_and_nan_confidence_is_null() -> None:
    links = _links(
        [
            {"mention_id": "m1", "canonical_id": "A", "resolver": None},
            {"mention_id": "m2", "canonical_id": "B"},
        ]
    )
    graph = _build(
        links,
        _relations(
            [
                {
                    "relation_id": "r",
                    "subject_mention_id": "m1",
                    "object_mention_id": "m2",
                    "confidence": float("nan"),
                }
            ]
        ),
    )
    (edge,) = graph.edges
    assert edge["confidence"] is None
    assert None in {edge["subject_resolver"], edge["object_resolver"]}


def test_a_relation_model_without_confidence_is_refused() -> None:
    links = _links([{"mention_id": "m1", "canonical_id": "A"}])
    with pytest.raises(GraphExportError, match=r"relation model is missing.*confidence"):
        _build(links, _relations([]).drop("confidence"))


def test_lines_end_in_a_bare_newline_on_every_platform(tmp_path: Path) -> None:
    out = tmp_path / "g"
    _two_node_graph(out)
    for kind in (NODES_DIRNAME, EDGES_DIRNAME):
        assert b"\r" not in _shard(out, "d1", kind).read_bytes()
    assert not list(out.rglob("*.tmp"))


# --- validity on edges (ALE-422, ADR-0056): schema v2 -----------------------

_VALIDITY_SCHEMA = {
    "valid_from": pl.Datetime("us", "UTC"),
    "valid_to": pl.Datetime("us", "UTC"),
    "observed_at": pl.Datetime("us", "UTC"),
    "superseded_by": pl.String(),
}
_EDGE_VALIDITY = ("valid_from", "valid_to", "observed_at", "superseded_by")


def _ab_links() -> pl.DataFrame:
    return _links(
        [
            {"mention_id": "m1", "canonical_id": "A"},
            {"mention_id": "m2", "canonical_id": "B"},
        ]
    )


def _validity_relations(**validity: Any) -> pl.DataFrame:
    return _relations(
        [
            {
                "relation_id": "r1",
                "subject_mention_id": "m1",
                "object_mention_id": "m2",
                "directed": True,
            }
        ]
    ).with_columns(
        *(
            pl.Series(name, [validity.get(name)], dtype=dtype)
            for name, dtype in _VALIDITY_SCHEMA.items()
        )
    )


def _validity_graph(**validity: Any) -> Any:
    return _build(_ab_links(), _validity_relations(**validity))


def test_the_export_schema_is_v3_because_the_layout_became_shards() -> None:
    """ADR-0055 said a change of shape arrives as a version bump, not as null
    columns: v2 added edge validity, v3 replaced the two whole files with a
    shard per document (ADR-0061), so a reader that checks the version learns
    the layout changed."""
    assert GRAPH_EXPORT_SCHEMA == "stel.graph_export/v3"


def test_an_edge_carries_its_validity_interval_as_utc_iso_strings() -> None:
    graph = _validity_graph(
        valid_from=datetime(2020, 1, 1, tzinfo=UTC),
        valid_to=datetime(2023, 6, 1, 12, 30, 15, 250000, tzinfo=UTC),
        observed_at=datetime(2020, 1, 2, tzinfo=UTC),
        superseded_by="r2",
    )

    (edge,) = graph.edges
    assert edge["valid_from"] == "2020-01-01T00:00:00Z"
    assert edge["valid_to"] == "2023-06-01T12:30:15.250000Z"
    assert edge["observed_at"] == "2020-01-02T00:00:00Z"
    assert edge["superseded_by"] == "r2"


def test_an_open_interval_has_a_null_end() -> None:
    graph = _validity_graph(valid_from=datetime(2020, 1, 1, tzinfo=UTC))

    (edge,) = graph.edges
    assert edge["valid_to"] is None
    assert edge["superseded_by"] is None


def test_a_relation_model_built_before_validity_exports_null_validity() -> None:
    """A table with no validity columns says the validity is unknown, which is
    what null means here; the edge still has the keys so v2 has one shape."""
    graph = _build(
        _ab_links(),
        _relations([{"relation_id": "r1", "subject_mention_id": "m1", "object_mention_id": "m2"}]),
    )

    (edge,) = graph.edges
    assert {name: edge[name] for name in _EDGE_VALIDITY} == dict.fromkeys(_EDGE_VALIDITY)


def test_validity_survives_the_written_file_byte_for_byte(tmp_path: Path) -> None:
    relations = _validity_relations(
        valid_from=datetime(2020, 1, 1, tzinfo=UTC),
        valid_to=datetime(2023, 1, 1, tzinfo=UTC),
        superseded_by="r2",
    )

    out = tmp_path / "g"
    _export(out, _ab_links(), relations)
    _export(tmp_path / "g2", _ab_links(), relations)

    shard = _shard(out, "d1", EDGES_DIRNAME)
    line = shard.read_text(encoding="utf-8").splitlines()[0]
    assert json.loads(line)["valid_to"] == "2023-01-01T00:00:00Z"
    assert shard.read_bytes() == _shard(tmp_path / "g2", "d1", EDGES_DIRNAME).read_bytes()


def test_a_validity_column_that_is_not_a_timestamp_is_refused_by_name() -> None:
    relations = _relations(
        [{"relation_id": "r1", "subject_mention_id": "m1", "object_mention_id": "m2"}]
    ).with_columns(pl.lit("2020-01-01").alias("valid_from"))

    with pytest.raises(GraphExportError, match="valid_from"):
        _build(
            _ab_links(),
            relations,
        )
