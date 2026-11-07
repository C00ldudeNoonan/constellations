"""Graph export contract (issue #762, ALE-423, ADR-0055).

The pure builder and the file writer, over hand-built frames. The demo project
runs through the CLI in `test_demo_template.py`.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from stel.agent_context import project_entity_link
from stel.config.class_hierarchy import ClassDef
from stel.config.vocabulary import Vocabulary
from stel.graph_export import (
    EDGES_FILENAME,
    GRAPH_EXPORT_SCHEMA,
    GRAPH_MANIFEST_FILENAME,
    NODES_FILENAME,
    GraphExportError,
    build_graph,
    write_graph_export,
)

_RECORDED = datetime(2026, 1, 1)


def _links(rows: list[dict[str, Any]]) -> pl.DataFrame:
    base = {
        "entity_namespace": "systems",
        "status": "matched",
        "resolver": "alias_table",
        "resolver_version": "1",
    }
    return pl.DataFrame(
        [{**base, **row} for row in rows],
        schema={
            "mention_id": pl.String(),
            "canonical_id": pl.String(),
            "entity_namespace": pl.String(),
            "status": pl.String(),
            "resolver": pl.String(),
            "resolver_version": pl.String(),
        },
    )


def _relations(rows: list[dict[str, Any]]) -> pl.DataFrame:
    base = {
        "document_id": "d1",
        "relation_type": "co_occurs_with",
        "directed": False,
        "method": "co_occurrence",
        "status": "asserted",
        "confidence": None,
        "sentence_index": 0,
        "extractor": "co_occurrence",
        "extractor_version": "1",
    }
    return pl.DataFrame(
        [{**base, **row} for row in rows],
        schema={
            "relation_id": pl.String(),
            "document_id": pl.String(),
            "subject_mention_id": pl.String(),
            "object_mention_id": pl.String(),
            "relation_type": pl.String(),
            "directed": pl.Boolean(),
            "method": pl.String(),
            "status": pl.String(),
            "confidence": pl.Float64(),
            "sentence_index": pl.Int64(),
            "extractor": pl.String(),
            "extractor_version": pl.String(),
        },
    )


def _build(links: pl.DataFrame, relations: pl.DataFrame, **kwargs: Any) -> Any:
    options: dict[str, Any] = {
        "project": "p",
        "entity_name": "system",
        "vocabularies": {},
        "classes": (),
    }
    options.update(kwargs)
    return build_graph(links=links, relations=relations, **options)


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
                {"mention_id": "m1", "canonical_id": "A", "resolver_version": "7"},
                {"mention_id": "m2", "canonical_id": "B"},
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


def _two_node_graph() -> Any:
    return _build(
        _links(
            [
                {"mention_id": "m1", "canonical_id": "Café"},
                {"mention_id": "m2", "canonical_id": "B"},
            ]
        ),
        _relations(
            [
                {
                    "relation_id": "r",
                    "subject_mention_id": "m1",
                    "object_mention_id": "m2",
                    "confidence": 0.5,
                }
            ]
        ),
    )


def test_the_files_are_deterministic_and_the_manifest_digests_them(tmp_path: Path) -> None:
    graph = _two_node_graph()
    out = write_graph_export(
        graph, tmp_path / "g", linking_model="l", relation_model="r"
    )
    first = {p.name: p.read_bytes() for p in out.iterdir()}
    write_graph_export(graph, tmp_path / "g", linking_model="l", relation_model="r")
    assert {p.name: p.read_bytes() for p in out.iterdir()} == first
    assert set(first) == {NODES_FILENAME, EDGES_FILENAME, GRAPH_MANIFEST_FILENAME}

    manifest = json.loads(first[GRAPH_MANIFEST_FILENAME])
    assert manifest["schema"] == GRAPH_EXPORT_SCHEMA
    assert manifest["entity_name"] == "system"
    assert manifest["files"][EDGES_FILENAME] == {
        "rows": 1,
        "sha256": hashlib.sha256(first[EDGES_FILENAME]).hexdigest(),
    }
    assert manifest["files"][NODES_FILENAME]["rows"] == 2
    # No timestamp: identical inputs must give identical bytes.
    assert not any("time" in key or "generated" in key for key in manifest)

    nodes = [json.loads(line) for line in first[NODES_FILENAME].decode("utf-8").splitlines()]
    assert "Café" in {node["canonical_id"] for node in nodes}


def test_lines_end_in_a_bare_newline_on_every_platform(tmp_path: Path) -> None:
    out = write_graph_export(
        _two_node_graph(), tmp_path / "g", linking_model="l", relation_model="r"
    )
    for name in (NODES_FILENAME, EDGES_FILENAME):
        assert b"\r" not in (out / name).read_bytes()
    assert not list(out.glob("*.tmp"))
