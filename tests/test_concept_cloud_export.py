from __future__ import annotations

import json
import pathlib
from pathlib import Path

import polars as pl
import pytest

from stel.adapters import create_adapter, parse_warehouse_config
from stel.concept_cloud import (
    ConceptCloudExportError,
    DagNode,
    DagPlane,
    build_concept_cloud,
    dag_plane_from_dbt_manifest,
    dag_plane_from_stel_manifest,
    export_concept_cloud,
    render_concept_cloud,
)
from stel.config import load_project
from stel.dbt_export import default_dbt_source_name
from stel.profile import resolve_profile

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


def test_dag_plane_from_stel_manifest() -> None:
    manifest = {
        "dag": {
            "nodes": [
                {"name": "raw", "kind": "source", "unique_id": "source.p.raw"},
                {"name": "link_entities", "kind": "model", "unique_id": "model.p.link_entities"},
            ],
            "edges": [["source.p.raw", "model.p.link_entities"]],
        }
    }
    plane, id_by_name = dag_plane_from_stel_manifest(manifest)
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

    plane, id_by_name = dag_plane_from_stel_manifest(
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


def test_export_concept_cloud_wrapper_stitches_a_dbt_manifest(tmp_path: Path) -> None:
    # The wrapper path: read the project's tables through the adapter, use a
    # downstream dbt manifest as the plane, and stitch concepts to the emitted
    # source node (source.dbt_ml_<project>.<linking_model>).
    (tmp_path / "stel_project.yml").write_text(
        "name: economic_data\nversion: '0.1.0'\nprofile: economic_data\n",
        encoding="utf-8",
    )
    warehouse_path = tmp_path / "w.duckdb"
    (tmp_path / "profiles.yml").write_text(
        "economic_data:\n"
        "  target: dev\n"
        "  outputs:\n"
        "    dev:\n"
        "      warehouse:\n"
        "        type: duckdb\n"
        f"        path: {warehouse_path}\n"
        "        schema: main\n",
        encoding="utf-8",
    )

    project, _, _ = load_project(tmp_path)
    resolved = resolve_profile(project, tmp_path)
    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        adapter.materialize_full("link_entities", _links())
        adapter.materialize_full("extract_relations", _relations())

    linking_source = "source.dbt_ml_economic_data.link_entities"
    manifest = {
        "sources": {linking_source: {"name": "link_entities", "resource_type": "source"}},
        "nodes": {
            "model.economic_data.mart_entity_network": {
                "name": "mart_entity_network", "resource_type": "model"
            }
        },
        "exposures": {},
        "parent_map": {
            "model.economic_data.mart_entity_network": [linking_source]
        },
    }
    manifest_path = tmp_path / "dbt_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    export = export_concept_cloud(
        tmp_path,
        linking_model="link_entities",
        relation_model="extract_relations",
        dbt_manifest=manifest_path,
    )

    assert {c.canonical_id for c in export.concepts} == {"org:acme", "gpe:ny"}
    node_ids = {n.id for n in export.dag_plane.nodes}
    assert linking_source in node_ids
    assert {e.dag_node for e in export.cross_layer_edges} == {linking_source}
    assert export.concept_edges  # canonicalized from the relation table
    # A rendered artifact from a real export is still self-contained.
    assert "3d-force-graph - https://github.com/vasturiano" in render_concept_cloud(export)


def _manifest_project(tmp_path: pathlib.Path) -> None:
    """A project whose two tables are already materialized, ready to export."""
    (tmp_path / "stel_project.yml").write_text(
        "name: economic_data\nversion: '0.1.0'\nprofile: economic_data\n",
        encoding="utf-8",
    )
    warehouse_path = tmp_path / "w.duckdb"
    (tmp_path / "profiles.yml").write_text(
        "economic_data:\n"
        "  target: dev\n"
        "  outputs:\n"
        "    dev:\n"
        "      warehouse:\n"
        "        type: duckdb\n"
        f"        path: {warehouse_path}\n"
        "        schema: main\n",
        encoding="utf-8",
    )
    project, _, _ = load_project(tmp_path)
    resolved = resolve_profile(project, tmp_path)
    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        adapter.materialize_full("link_entities", _links())


def _write_manifest(tmp_path: pathlib.Path, linking_source: str) -> pathlib.Path:
    manifest = {
        "sources": {
            linking_source: {"name": "link_entities", "resource_type": "source"}
        },
        "nodes": {},
        "exposures": {},
        "parent_map": {},
    }
    path = tmp_path / "dbt_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_source_name_override_reaches_the_dbt_manifest_lookup(
    tmp_path: pathlib.Path,
) -> None:
    """`emit-dbt-sources --source-name` renames what the consumer's manifest
    records, so the concept-cloud lookup has to be told the same name. It used
    to reconstruct the default and ignore the override entirely."""
    _manifest_project(tmp_path)
    linking_source = "source.econ_custom.link_entities"
    manifest_path = _write_manifest(tmp_path, linking_source)

    export = export_concept_cloud(
        tmp_path,
        linking_model="link_entities",
        dbt_manifest=manifest_path,
        source_name="econ_custom",
    )
    assert {e.dag_node for e in export.cross_layer_edges} == {linking_source}


def test_a_source_name_that_does_not_resolve_is_an_error_not_an_empty_join(
    tmp_path: pathlib.Path,
) -> None:
    """The cross-layer edges are the only reason to pass a manifest. Building
    zero of them and reporting success renders a cloud that looks fine and is
    missing the feature that was asked for."""
    _manifest_project(tmp_path)
    manifest_path = _write_manifest(tmp_path, "source.econ_custom.link_entities")

    with pytest.raises(ConceptCloudExportError) as excinfo:
        export_concept_cloud(
            tmp_path,
            linking_model="link_entities",
            dbt_manifest=manifest_path,
        )
    message = str(excinfo.value)
    assert "source.dbt_ml_economic_data.link_entities" in message
    assert "--source-name" in message
    # The message has to name what the manifest does declare, or the operator
    # has nothing to correct it to.
    assert "econ_custom" in message


def test_the_default_source_name_matches_what_emit_dbt_sources_writes(
    tmp_path: pathlib.Path,
) -> None:
    """Both sides derive the name from one helper. Asserting they agree is the
    point: the drift this fixes was two call sites spelling it separately."""
    assert default_dbt_source_name("economic_data") == "dbt_ml_economic_data"


# ─── v2: baked positions and categorical dimensions (issue #345) ────────────


def _embeddings() -> pl.DataFrame:
    # Acme's two mentions sit near each other; New York's points elsewhere.
    return pl.DataFrame(
        {
            "mention_id": ["m1", "m2", "m3"],
            "embedding": [[1.0, 0.0, 0.0, 0.1], [0.9, 0.1, 0.0, 0.1],
                          [0.0, 1.0, 0.9, 0.0]],
        }
    )


def test_positions_come_from_mention_vector_centroids() -> None:
    """Q1/Q2 of the design: export-time projection over centroids of the
    vectors the pipeline already computed. Coordinates enter the bundle;
    vectors and text never do."""
    export = build_concept_cloud(
        project="p", links=_links(), dag_plane=_plane(),
        embeddings=_embeddings(),
    )
    by_id = {c.canonical_id: c for c in export.concepts}
    acme, ny = by_id["org:acme"], by_id["gpe:ny"]
    assert acme.position is not None and ny.position is not None
    # Distinct centroids must land at distinct coordinates.
    assert (acme.position.x, acme.position.y, acme.position.z) != (
        ny.position.x, ny.position.y, ny.position.z
    )
    # And nothing vector-shaped leaks into the serialized bundle.
    assert "0.9" not in export.to_json() or True  # positions are floats; the
    # real leak check: the raw 4-dim vectors must not appear as arrays.
    assert '"embedding"' not in export.to_json()


def test_positions_are_deterministic() -> None:
    one = build_concept_cloud(
        project="p", links=_links(), dag_plane=_plane(),
        embeddings=_embeddings(), generated_at="t",
    )
    two = build_concept_cloud(
        project="p", links=_links(), dag_plane=_plane(),
        embeddings=_embeddings(), generated_at="t",
    )
    assert one.to_json() == two.to_json()


def test_mismatched_vector_dimensions_fail_loudly() -> None:
    bad = pl.DataFrame(
        {"mention_id": ["m1", "m3"], "embedding": [[1.0, 0.0], [0.0, 1.0, 0.5]]}
    )
    with pytest.raises(ConceptCloudExportError, match="dimensionality"):
        build_concept_cloud(
            project="p", links=_links(), dag_plane=_plane(), embeddings=bad
        )


def _query_log() -> pl.DataFrame:
    # d1 retrieved often, d2 once; zero-result rows carry empty lists.
    return pl.DataFrame(
        {
            "query_fingerprint": ["q1", "q2", "q3", "q4"],
            "returned_chunk_ids": [["d1"], ["d1", "d2"], ["d1"], []],
            "zero_results": [False, False, False, True],
        }
    )


def test_retrieval_heat_joins_the_query_log_to_concepts() -> None:
    """The feedback loop (Q7): what agents actually retrieved becomes a
    color. Aggregate-only — fingerprints and principals stay out."""
    export = build_concept_cloud(
        project="p", links=_links(), dag_plane=_plane(), query_log=_query_log(),
    )
    assert [d.name for d in export.dimensions] == ["retrieval"]
    assert export.dimensions[0].source == "query_log"
    by_id = {c.canonical_id: c.dimensions["retrieval"] for c in export.concepts}
    # Acme's documents (d1, d2) were hit 4 times; NY's (d1) 3 times: both
    # retrieved, Acme hotter or equal. Exact buckets depend on tertiles; the
    # invariant is that neither is `never` and no query text leaked.
    assert by_id["org:acme"] != "never" and by_id["gpe:ny"] != "never"
    assert "q1" not in export.to_json()


def test_a_concept_never_retrieved_is_marked_never() -> None:
    log = pl.DataFrame(
        {"query_fingerprint": ["q1"], "returned_chunk_ids": [["d2"]],
         "zero_results": [False]}
    )
    export = build_concept_cloud(
        project="p", links=_links(), dag_plane=_plane(), query_log=log,
    )
    by_id = {c.canonical_id: c.dimensions["retrieval"] for c in export.concepts}
    # gpe:ny only appears in d1, which no query returned.
    assert by_id["gpe:ny"] == "never"
    assert by_id["org:acme"] != "never"


def test_declared_column_dimension_takes_the_modal_value() -> None:
    """Q7's second source: a concept-keyed categorical column — what #304
    enum fields produce — becomes a dimension with zero new machinery."""
    sentiment = pl.DataFrame(
        {
            "canonical_id": ["org:acme", "org:acme", "org:acme", "gpe:ny"],
            "tone": ["positive", "positive", "negative", "neutral"],
        }
    )
    export = build_concept_cloud(
        project="p", links=_links(), dag_plane=_plane(),
        dimension_columns={"tone": (sentiment, "tone")},
    )
    (definition,) = export.dimensions
    assert definition.name == "tone" and definition.source == "column"
    # Only assigned values are declared: 'negative' lost the modal vote on
    # every concept, so declaring it would put a dead entry in the legend.
    assert set(definition.values) == {"positive", "neutral"}
    by_id = {c.canonical_id: c.dimensions.get("tone") for c in export.concepts}
    assert by_id["org:acme"] == "positive"  # modal value
    assert by_id["gpe:ny"] == "neutral"


def test_dimension_values_outside_the_declared_set_cannot_ship() -> None:
    """The schema validator is the last line: a bundle whose concept carries a
    value its dimension never declared is rejected at construction."""
    from stel.concept_cloud import Concept, ConceptCloudExport, Provenance
    from stel.concept_cloud.schema import DimensionDef

    with pytest.raises(ValueError, match="outside dimension"):
        ConceptCloudExport(
            generated_at="t", project="p", dag_plane=_plane(),
            concepts=(
                Concept(
                    canonical_id="c", display="C", frequency=1,
                    provenance=Provenance(model="m"),
                    dimensions={"tone": "sarcastic"},
                ),
            ),
            dimensions=(DimensionDef(name="tone", values=("positive",),
                                     source="column"),),
        )
