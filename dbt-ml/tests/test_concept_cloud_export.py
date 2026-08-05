from __future__ import annotations

from pathlib import Path

import polars as pl

from dbt_ml.adapters import create_adapter, parse_warehouse_config
from dbt_ml.concept_cloud import (
    DagNode,
    DagPlane,
    build_concept_cloud,
    dag_plane_from_dbt_manifest,
    dag_plane_from_dbt_ml_manifest,
    render_concept_cloud,
)

_LINKING_NODE = "model.p.link_entities"


def _plane() -> DagPlane:
    return DagPlane(
        nodes=(DagNode(id=_LINKING_NODE, label="link_entities", resource_type="model"),)
    )


def _links() -> pl.DataFrame:
    # m1/m2 -> Acme; m3 -> New York (ambiguous, scored); m4 unmatched (null id).
    return pl.DataFrame(
        {
            "mention_id": ["m1", "m2", "m3", "m4"],
            "canonical_id": ["org:acme", "org:acme", "gpe:ny", None],
            "document_id": ["d1", "d2", "d1", "d1"],
            "status": ["matched", "matched", "ambiguous", "unmatched"],
            "match_score": [None, None, 0.8, None],
            "label": ["ORG", "ORG", "GPE", None],
            "mention_text": ["Acme", "Acme Corp", "New York", None],
        }
    )


def _relations() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "subject_mention_id": ["m1", "m3", "m2"],
            "object_mention_id": ["m3", "m1", "m3"],
            "relation_type": ["co_occurs_with", "co_occurs_with", "co_occurs_with"],
            "directed": [False, False, False],
            "method": ["co_occurrence", "co_occurrence", "co_occurrence"],
            "confidence": [None, None, None],
        }
    )


def test_build_aggregates_concepts_and_drops_unmatched() -> None:
    export = build_concept_cloud(
        project="p", links=_links(), dag_plane=_plane(),
        linking_node_id=_LINKING_NODE, linking_model="link_entities",
    )
    by_id = {c.canonical_id: c for c in export.concepts}
    assert set(by_id) == {"org:acme", "gpe:ny"}  # m4 (unmatched) dropped
    assert by_id["org:acme"].frequency == 2
    assert by_id["org:acme"].display == "Acme"          # first non-null text
    assert by_id["org:acme"].label == "ORG"
    assert by_id["org:acme"].link_status == "matched"
    assert by_id["org:acme"].provenance.documents == 2  # d1, d2
    assert by_id["gpe:ny"].link_status == "ambiguous"
    assert by_id["gpe:ny"].match_score == 0.8
    # Every kept concept gets a cross-layer edge to the linking node.
    assert {e.concept for e in export.cross_layer_edges} == {"org:acme", "gpe:ny"}
    assert all(e.dag_node == _LINKING_NODE for e in export.cross_layer_edges)


def test_build_canonicalizes_and_collapses_undirected_edges() -> None:
    export = build_concept_cloud(
        project="p", links=_links(), relations=_relations(), dag_plane=_plane(),
        linking_node_id=_LINKING_NODE,
    )
    # All three mention-level relations map to the same undirected concept pair.
    assert len(export.concept_edges) == 1
    edge = export.concept_edges[0]
    assert (edge.source, edge.target) == ("gpe:ny", "org:acme")  # sorted pair
    assert edge.directed is False
    assert edge.weight == 3
    assert edge.method == "co_occurrence"


def test_build_top_n_caps_and_prunes_dangling_edges() -> None:
    export = build_concept_cloud(
        project="p", links=_links(), relations=_relations(), dag_plane=_plane(),
        linking_node_id=_LINKING_NODE, top_n=1,
    )
    assert [c.canonical_id for c in export.concepts] == ["org:acme"]  # most frequent
    # The only edge referenced gpe:ny, now dropped -> no dangling edges.
    assert export.concept_edges == ()
    assert len(export.cross_layer_edges) == 1


def test_build_preserves_directed_edges() -> None:
    links = pl.DataFrame(
        {"mention_id": ["m1", "m2"], "canonical_id": ["org:a", "org:b"],
         "document_id": ["d1", "d1"], "status": ["matched", "matched"]}
    )
    relations = pl.DataFrame(
        {"subject_mention_id": ["m1"], "object_mention_id": ["m2"],
         "relation_type": ["acquired"], "directed": [True],
         "method": ["model_assertion"], "confidence": [0.91]}
    )
    export = build_concept_cloud(
        project="p", links=links, relations=relations, dag_plane=_plane(),
    )
    edge = export.concept_edges[0]
    assert (edge.source, edge.target) == ("org:a", "org:b")  # direction preserved
    assert edge.directed is True
    assert edge.confidence == 0.91


def test_dag_plane_from_dbt_ml_manifest() -> None:
    manifest = {
        "dag": {
            "nodes": [
                {"name": "raw", "kind": "source", "unique_id": "source.p.raw"},
                {"name": "link_entities", "kind": "model", "unique_id": "model.p.link_entities"},
            ],
            "edges": [["source.p.raw", "model.p.link_entities"]],
        }
    }
    plane, id_by_name = dag_plane_from_dbt_ml_manifest(manifest)
    assert id_by_name["link_entities"] == "model.p.link_entities"
    assert {n.resource_type for n in plane.nodes} == {"source", "model"}
    assert plane.edges[0].from_ == "source.p.raw"


def test_dag_plane_from_dbt_manifest_filters_and_links() -> None:
    manifest = {
        "sources": {"source.p.raw": {"name": "raw", "resource_type": "source"}},
        "nodes": {
            "model.p.stg": {"name": "stg", "resource_type": "model"},
            "seed.p.cur": {"name": "cur", "resource_type": "seed"},
            "test.p.t1": {"name": "t1", "resource_type": "test"},  # excluded
        },
        "exposures": {},
        "parent_map": {"model.p.stg": ["source.p.raw", "test.p.t1"]},
    }
    plane = dag_plane_from_dbt_manifest(manifest)
    ids = {n.id for n in plane.nodes}
    assert ids == {"source.p.raw", "model.p.stg", "seed.p.cur"}  # test excluded
    # Only the edge to a kept parent survives.
    assert [(e.from_, e.to) for e in plane.edges] == [("source.p.raw", "model.p.stg")]


def test_export_end_to_end_through_duckdb(tmp_path: Path) -> None:
    # The whole join over a real warehouse: write the linking + relation tables to
    # DuckDB, read them back through the adapter, build a bundle, and render it.
    warehouse = parse_warehouse_config(
        {"type": "duckdb", "path": str(tmp_path / "w.duckdb"), "schema": "main"}
    )
    with create_adapter(warehouse) as adapter:
        adapter.materialize_full("link_entities", _links())
        adapter.materialize_full("extract_relations", _relations())
        links = adapter.read_table("link_entities")
        relations = adapter.read_table("extract_relations")

    plane, id_by_name = dag_plane_from_dbt_ml_manifest(
        {"dag": {"nodes": [
            {"name": "link_entities", "kind": "model", "unique_id": _LINKING_NODE}
        ], "edges": []}}
    )
    export = build_concept_cloud(
        project="p", links=links, relations=relations, dag_plane=plane,
        linking_node_id=id_by_name["link_entities"],
    )
    assert {c.canonical_id for c in export.concepts} == {"org:acme", "gpe:ny"}
    assert export.concept_edges[0].weight == 3
    html = render_concept_cloud(export)
    assert "Acme" in html and "__CONCEPT_CLOUD_DATA__" not in html
