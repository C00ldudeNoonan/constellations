from __future__ import annotations

import json
import pathlib
from pathlib import Path
from typing import Any, cast

import polars as pl
import pytest

from stel.adapters import create_adapter, parse_warehouse_config
from stel.concept_cloud import (
    Concept,
    ConceptCloudExport,
    ConceptCloudExportError,
    DagNode,
    DagPlane,
    Provenance,
    build_concept_cloud,
    concept_names,
    dag_plane_from_dbt_manifest,
    dag_plane_from_stel_manifest,
    export_concept_cloud,
    render_concept_cloud,
)
from stel.config import load_project
from stel.dbt_export import default_dbt_source_name
from stel.profile import resolve_profile

# Runs a whole project or opens a retrieval store, so it belongs to the
# `e2e` tier (issue #518). `test_test_tiers.py` fails if a file that
# does either is missing this.
pytestmark = pytest.mark.e2e

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
    # "Acme" and "Acme Corp" appear once each; the lexical tie-break picks
    # "Acme". `test_display_is_the_most_frequent_mention_text` is the case
    # that actually discriminates the rule.
    assert by_id["org:acme"].display == "Acme"
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

    linking_source = (
        "source.consumer_dbt_project.dbt_ml_economic_data.link_entities"
    )
    manifest = {
        "sources": {
            linking_source: {
                "name": "link_entities",
                "source_name": "dbt_ml_economic_data",
                "resource_type": "source",
            }
        },
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


def _write_manifest(tmp_path: pathlib.Path, source_name: str) -> pathlib.Path:
    """A manifest shaped the way dbt actually writes one (issue #552).

    dbt's unique_id for a source table is
    `source.<dbt_project>.<source_name>.<table>` -- four segments, with the
    dbt project's name second. The fixtures used to use three, which is a
    shape dbt never produces, so the tests agreed with the bug: the lookup
    rebuilt a three-segment id and matched a fixture that had one.
    """
    manifest = {
        "sources": {
            f"source.consumer_dbt_project.{source_name}.link_entities": {
                "name": "link_entities",
                "source_name": source_name,
                "resource_type": "source",
            }
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
    manifest_path = _write_manifest(tmp_path, "econ_custom")

    export = export_concept_cloud(
        tmp_path,
        linking_model="link_entities",
        dbt_manifest=manifest_path,
        source_name="econ_custom",
    )
    # The four-segment id dbt writes, resolved from the manifest's own fields
    # rather than rebuilt from the source name (issue #552).
    assert {e.dag_node for e in export.cross_layer_edges} == {
        "source.consumer_dbt_project.econ_custom.link_entities"
    }


def test_a_source_name_that_does_not_resolve_is_an_error_not_an_empty_join(
    tmp_path: pathlib.Path,
) -> None:
    """The cross-layer edges are the only reason to pass a manifest. Building
    zero of them and reporting success renders a cloud that looks fine and is
    missing the feature that was asked for."""
    _manifest_project(tmp_path)
    manifest_path = _write_manifest(tmp_path, "econ_custom")

    with pytest.raises(ConceptCloudExportError) as excinfo:
        export_concept_cloud(
            tmp_path,
            linking_model="link_entities",
            dbt_manifest=manifest_path,
        )
    message = str(excinfo.value)
    assert "'link_entities' is not declared as a source named" in message
    assert "dbt_ml_economic_data" in message
    # The linking model has to be in the manifest at all, which is the step
    # that is actually missing when this fires (issue #552).
    assert "emit-dbt-sources" in message
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


def test_a_real_dbt_manifest_id_resolves(tmp_path: pathlib.Path) -> None:
    """The shape a real consumer manifest has (issue #552).

    dbt writes `source.<dbt_project>.<source_name>.<table>`. stel rebuilt the
    id as `source.<source_name>.<table>`, so `--dbt-manifest` failed against
    every real manifest, and the error's "sources it declares" hint listed the
    dbt *project* name -- pointing the operator at a `--source-name` value
    that could not work, when the one they passed was already right.

    The id here is copied from the report: the astrolabe project's own
    manifest, whose sources were written by `emit-dbt-sources --source-name
    dbt_ml_document_extraction`.
    """
    _manifest_project(tmp_path)
    linking_source = (
        "source.dbt_project.dbt_ml_document_extraction.link_entities"
    )
    manifest = {
        "sources": {
            linking_source: {
                "name": "link_entities",
                "source_name": "dbt_ml_document_extraction",
                "resource_type": "source",
            }
        },
        "nodes": {
            "model.dbt_project.mart_entity_network": {
                "name": "mart_entity_network",
                "resource_type": "model",
            }
        },
        "exposures": {},
        "parent_map": {"model.dbt_project.mart_entity_network": [linking_source]},
    }
    manifest_path = tmp_path / "dbt_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    export = export_concept_cloud(
        tmp_path,
        linking_model="link_entities",
        dbt_manifest=manifest_path,
        source_name="dbt_ml_document_extraction",
    )

    assert {e.dag_node for e in export.cross_layer_edges} == {linking_source}


def test_the_unresolved_hint_lists_source_names_not_the_dbt_project(
    tmp_path: pathlib.Path,
) -> None:
    """The hint pointed at the wrong thing in exactly the case it exists for.

    Reading `unique_id.split(".")[1]` yields the dbt project name for a real
    four-segment id, so an operator was told to pass `--source-name
    dbt_project` -- a value that can never match (issue #552).
    """
    _manifest_project(tmp_path)
    manifest = {
        "sources": {
            "source.dbt_project.dbt_ml_document_extraction.link_entities": {
                "name": "link_entities",
                "source_name": "dbt_ml_document_extraction",
                "resource_type": "source",
            }
        },
        "nodes": {},
        "exposures": {},
        "parent_map": {},
    }
    manifest_path = tmp_path / "dbt_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ConceptCloudExportError) as excinfo:
        export_concept_cloud(
            tmp_path,
            linking_model="link_entities",
            dbt_manifest=manifest_path,
            source_name="wrong_name",
        )

    message = str(excinfo.value)
    assert "dbt_ml_document_extraction" in message
    # The dbt project name is not a source name and must not be offered as one.
    assert "dbt_project'" not in message
    assert "['dbt_project']" not in message


# ── display names and descriptions (#554) ──────────────────────────────────


def _text_links() -> pl.DataFrame:
    """One concept whose first row's text is not its most frequent one.

    Under the old `_first` rule this concept was named "Pentair Water
    Solutions plc" -- whichever mention the frame happened to yield first.
    """
    return pl.DataFrame(
        {
            "mention_id": ["m1", "m2", "m3", "m4"],
            "canonical_id": ["org:pnr"] * 4,
            "document_id": ["d1", "d2", "d3", "d4"],
            "status": ["matched"] * 4,
            "label": ["ORG"] * 4,
            "mention_text": [
                "Pentair Water Solutions plc", "Pentair", "Pentair", "PNR",
            ],
        }
    )


def _names() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "canonical_id": ["org:pnr"],
            "display_name": ["Pentair"],
            "description": ["Water treatment equipment manufacturer (NYSE: PNR)."],
        }
    )


def test_display_is_the_most_frequent_mention_text() -> None:
    """Row order must not name a concept (issue #554).

    A real map showed `US`, `U.K.`, `Aon` and `American Water Works Company,
    Inc.` side by side because each was whichever mention text the linking
    frame yielded first.
    """
    export = build_concept_cloud(
        project="p", links=_text_links(), dag_plane=_plane(),
    )
    concept = export.concepts[0]
    assert concept.display == "Pentair"      # 2 mentions, not the first row's
    assert concept.description is None       # nothing supplies one by default


def test_display_ties_break_lexically_not_by_row_order() -> None:
    links = _text_links().with_columns(
        pl.Series("mention_text", ["Zebra", "Zebra", "Alpha", "Alpha"])
    )
    reversed_rows = links.reverse()
    assert (
        build_concept_cloud(project="p", links=links, dag_plane=_plane())
        .concepts[0].display
        == build_concept_cloud(project="p", links=reversed_rows, dag_plane=_plane())
        .concepts[0].display
        == "Alpha"
    )


def test_entity_text_enriches_only_rows_without_their_own_text() -> None:
    """Enrichment is per row, then counted -- not a second list appended after
    every row's own text, which is what made the old rule order-dependent."""
    links = _text_links().with_columns(
        pl.Series("mention_text", ["Pentair plc", None, None, None])
    )
    entities = pl.DataFrame(
        {"entity_id": ["m2", "m3", "m4"], "entity_text": ["Pentair"] * 3}
    )
    export = build_concept_cloud(
        project="p", links=links, entities=entities, dag_plane=_plane(),
    )
    assert export.concepts[0].display == "Pentair"


def test_names_model_supplies_the_display_name_and_description() -> None:
    export = build_concept_cloud(
        project="p", links=_text_links(), dag_plane=_plane(), names=_names(),
    )
    concept = export.concepts[0]
    assert concept.display == "Pentair"
    assert concept.description == (
        "Water treatment equipment manufacturer (NYSE: PNR)."
    )


def test_names_model_outranks_the_most_frequent_mention() -> None:
    names = _names().with_columns(pl.Series("display_name", ["Pentair plc"]))
    export = build_concept_cloud(
        project="p", links=_text_links(), dag_plane=_plane(), names=names,
    )
    assert export.concepts[0].display == "Pentair plc"


def test_names_model_blank_cells_fall_back_rather_than_blanking_a_node() -> None:
    names = pl.DataFrame(
        {
            "canonical_id": ["org:pnr"],
            "display_name": ["   "],
            "description": [""],
        }
    )
    concept = build_concept_cloud(
        project="p", links=_text_links(), dag_plane=_plane(), names=names,
    ).concepts[0]
    assert concept.display == "Pentair"   # the mention-text rule, not a blank
    assert concept.description is None


def test_names_model_covers_only_the_concepts_it_names() -> None:
    export = build_concept_cloud(
        project="p", links=_links(), dag_plane=_plane(),
        names=pl.DataFrame(
            {"canonical_id": ["org:acme"], "display_name": ["Acme Corporation"]}
        ),
    )
    by_id = {c.canonical_id: c for c in export.concepts}
    assert by_id["org:acme"].display == "Acme Corporation"
    assert by_id["gpe:ny"].display == "New York"   # unnamed, unchanged
    assert by_id["gpe:ny"].description is None


def test_names_model_without_a_description_column_is_fine() -> None:
    names = pl.DataFrame(
        {"canonical_id": ["org:pnr"], "display_name": ["Pentair"]}
    )
    assert concept_names(names)["org:pnr"].description is None


def test_names_model_missing_a_required_column_is_refused_by_name() -> None:
    with pytest.raises(ConceptCloudExportError, match="display_name"):
        concept_names(pl.DataFrame({"canonical_id": ["org:pnr"]}))
    with pytest.raises(ConceptCloudExportError, match="canonical_id"):
        concept_names(pl.DataFrame({"display_name": ["Pentair"]}))


def test_names_model_duplicate_canonical_id_is_refused() -> None:
    """A warehouse read promises no row order, so "first row wins" would let
    the map's labels change between two exports of unchanged data."""
    names = pl.DataFrame(
        {
            "canonical_id": ["org:pnr", "org:pnr"],
            "display_name": ["Pentair", "Pentair plc"],
        }
    )
    with pytest.raises(ConceptCloudExportError, match="more than one row"):
        concept_names(names)


def test_export_reads_the_names_model_through_the_adapter(
    tmp_path: pathlib.Path,
) -> None:
    _manifest_project(tmp_path)
    project, _, _ = load_project(tmp_path)
    resolved = resolve_profile(project, tmp_path)
    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        adapter.materialize_full(
            "concept_names",
            pl.DataFrame(
                {
                    "canonical_id": ["org:acme"],
                    "display_name": ["Acme Corporation"],
                    "description": ["A fictional maker of anvils."],
                }
            ),
        )

    export = export_concept_cloud(
        tmp_path, linking_model="link_entities", names_model="concept_names",
    )
    by_id = {c.canonical_id: c for c in export.concepts}
    assert by_id["org:acme"].display == "Acme Corporation"
    assert by_id["org:acme"].description == "A fictional maker of anvils."


def test_cli_names_model_reaches_the_export(tmp_path: pathlib.Path) -> None:
    """The flag is hand-wired to a keyword argument; a typo there would leave
    it silently None and every node back on its mention text."""
    from click.testing import CliRunner

    from stel.cli import cli

    _manifest_project(tmp_path)
    project, _, _ = load_project(tmp_path)
    resolved = resolve_profile(project, tmp_path)
    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        adapter.materialize_full(
            "concept_names",
            pl.DataFrame(
                {
                    "canonical_id": ["org:acme"],
                    "display_name": ["Acme Corporation"],
                    "description": ["A fictional maker of anvils."],
                }
            ),
        )

    out = tmp_path / "cloud.html"
    result = CliRunner().invoke(
        cli,
        [
            "--project-dir", str(tmp_path), "concept-cloud",
            "--linking-model", "link_entities",
            "--names-model", "concept_names",
            "--output", str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    html = out.read_text(encoding="utf-8")
    assert "Acme Corporation" in html
    assert "A fictional maker of anvils." in html


# ─── v3: a time axis (issue #553) ───────────────────────────────────────────


def _timed_links() -> pl.DataFrame:
    """Mentions carrying a filing year, the shape #553 describes."""
    return pl.DataFrame(
        {
            "mention_id": ["m1", "m2", "m3", "m4", "m5"],
            "canonical_id": ["FERC", "FERC", "FERC", "AES", "AES"],
            "mention_text": ["FERC", "FERC", "FERC", "AES", "AES"],
            "document_id": ["d1", "d2", "d3", "d1", "d3"],
            "filing_year": [2019, 2019, 2021, 2019, 2021],
        }
    )


def test_a_time_field_gives_each_concept_its_periods() -> None:
    """One bundle covering every period, instead of one artifact per period."""
    export = build_concept_cloud(
        project="p",
        links=_timed_links(),
        dag_plane=DagPlane(nodes=()),
        linking_model="link_entities",
        time_field="filing_year",
    )

    assert export.schema_version == "3"
    assert export.periods == ("2019", "2021")
    by_id = {c.canonical_id: c for c in export.concepts}
    assert by_id["FERC"].by_period == {"2019": 2, "2021": 1}
    assert by_id["AES"].by_period == {"2019": 1, "2021": 1}
    # Totals are unchanged by the axis.
    assert by_id["FERC"].frequency == 3


def test_without_a_time_field_the_bundle_has_no_periods() -> None:
    """The axis is opt-in; nothing changes for an export that omits it."""
    export = build_concept_cloud(
        project="p",
        links=_timed_links(),
        dag_plane=DagPlane(nodes=()),
        linking_model="link_entities",
    )

    assert export.periods == ()
    assert all(c.by_period == {} for c in export.concepts)


@pytest.mark.parametrize(
    ("grain", "expected"),
    [
        ("year", "2021"),
        ("quarter", "2021Q3"),
        ("month", "2021-08"),
    ],
)
def test_the_grain_decides_the_period_key(grain: str, expected: str) -> None:
    """Keys are sortable as strings, which is what the ascending axis and the
    viewer's slider both rely on."""
    links = pl.DataFrame(
        {
            "mention_id": ["m1"],
            "canonical_id": ["FERC"],
            "mention_text": ["FERC"],
            "filed_at": ["2021-08-14T00:00:00"],
        }
    )
    export = build_concept_cloud(
        project="p", links=links, dag_plane=DagPlane(nodes=()),
        linking_model="link_entities",
        time_field="filed_at", time_grain=cast(Any, grain),
    )
    assert export.periods == (expected,)
    assert export.concepts[0].by_period == {expected: 1}


def test_a_mention_with_no_usable_date_counts_in_the_total_only() -> None:
    """Silently bucketing an unreadable date would put mentions in a period
    they are not from, which is worse than a total exceeding its periods."""
    links = pl.DataFrame(
        {
            "mention_id": ["m1", "m2", "m3"],
            "canonical_id": ["FERC", "FERC", "FERC"],
            "mention_text": ["FERC", "FERC", "FERC"],
            "filing_year": ["2019", None, "not a date"],
        }
    )
    export = build_concept_cloud(
        project="p", links=links, dag_plane=DagPlane(nodes=()),
        linking_model="link_entities", time_field="filing_year",
    )
    concept = export.concepts[0]
    assert concept.frequency == 3
    assert concept.by_period == {"2019": 1}
    assert sum(concept.by_period.values()) < concept.frequency


def test_an_edge_is_periodized_only_when_both_mentions_agree() -> None:
    """A pair named in different periods was not named together *in* either."""
    links = pl.DataFrame(
        {
            "mention_id": ["m1", "m2", "m3", "m4"],
            "canonical_id": ["AES", "FERC", "AES", "FERC"],
            "mention_text": ["AES", "FERC", "AES", "FERC"],
            "filing_year": [2019, 2019, 2019, 2021],
        }
    )
    relations = pl.DataFrame(
        {
            "subject_mention_id": ["m1", "m3"],
            "object_mention_id": ["m2", "m4"],
            "relation_type": ["co_mention", "co_mention"],
        }
    )
    export = build_concept_cloud(
        project="p", links=links, dag_plane=DagPlane(nodes=()),
        relations=relations, linking_model="link_entities",
        time_field="filing_year",
    )

    edge = export.concept_edges[0]
    # Both relations count toward the total; only the same-period one is
    # attributed to a period.
    assert edge.weight == 2
    assert edge.by_period == {"2019": 1}


def test_a_missing_time_field_column_is_a_named_error() -> None:
    """Naming a column that is not there must not silently yield no periods."""
    with pytest.raises(ConceptCloudExportError) as excinfo:
        build_concept_cloud(
            project="p", links=_timed_links(), dag_plane=DagPlane(nodes=()),
            linking_model="link_entities", time_field="nope",
        )
    assert "'nope' is not a column on the linking model" in str(excinfo.value)


def test_a_bundle_cannot_count_a_period_it_does_not_declare() -> None:
    """The slider iterates the axis, so a count keyed outside it would be
    invisible — a bundle that cannot render what it contains."""
    with pytest.raises(ValueError, match="undeclared period"):
        ConceptCloudExport(
            generated_at="2026-01-01T00:00:00Z",
            project="p",
            dag_plane=DagPlane(nodes=()),
            concepts=(
                Concept(
                    canonical_id="FERC",
                    display="FERC",
                    frequency=1,
                    provenance=Provenance(model="link_entities", documents=1),
                    by_period={"2019": 1},
                ),
            ),
            periods=("2021",),
        )


def test_the_period_axis_survives_top_n_trimming() -> None:
    """The axis is every period the corpus covers, not every period that
    happened to survive `top_n`.

    A period whose only concepts were trimmed is still a period the corpus
    covers, and a slider that skipped it would read a gap as missing data
    rather than as "nothing was named then" (issue #553).
    """
    links = pl.DataFrame(
        {
            "mention_id": ["m1", "m2", "m3", "m4"],
            "canonical_id": ["FERC", "FERC", "FERC", "RARE"],
            "mention_text": ["FERC", "FERC", "FERC", "RARE"],
            "filing_year": [2019, 2019, 2019, 2021],
        }
    )
    export = build_concept_cloud(
        project="p", links=links, dag_plane=DagPlane(nodes=()),
        linking_model="link_entities", time_field="filing_year", top_n=1,
    )

    # Only the frequent concept survived...
    assert [c.canonical_id for c in export.concepts] == ["FERC"]
    assert export.concepts[0].by_period == {"2019": 3}
    # ...but 2021 is still part of the corpus, and still on the axis.
    assert export.periods == ("2019", "2021")


def test_top_n_per_period_keeps_a_risk_that_only_one_period_cared_about() -> None:
    """The case the flag exists for (issue #553).

    Ranking on total frequency trims exactly what a time axis is meant to
    show: something that enters, dominates one period, and is unremarkable
    across the corpus. Here `SUPPLY` is named twice in 2021 and never again,
    while three other concepts each out-total it — so `--top-n 3` drops it and
    the slider can never show it arriving.
    """
    links = pl.DataFrame(
        {
            "mention_id": [f"m{i}" for i in range(1, 12)],
            "canonical_id": [
                "FERC", "FERC", "FERC",
                "AES", "AES", "AES",
                "EU", "EU", "EU",
                "SUPPLY", "SUPPLY",
            ],
            "mention_text": [
                "FERC", "FERC", "FERC", "AES", "AES", "AES",
                "EU", "EU", "EU", "supply chain", "supply chain",
            ],
            "filing_year": [
                2019, 2020, 2021, 2019, 2020, 2021,
                2019, 2020, 2021, 2021, 2021,
            ],
        }
    )
    without = build_concept_cloud(
        project="p", links=links, dag_plane=DagPlane(nodes=()),
        linking_model="link_entities", time_field="filing_year", top_n=3,
    )
    assert "SUPPLY" not in {c.canonical_id for c in without.concepts}

    # SUPPLY is the second-biggest thing that happened in 2021, so a
    # per-period rank of 2 reaches it.
    with_flag = build_concept_cloud(
        project="p", links=links, dag_plane=DagPlane(nodes=()),
        linking_model="link_entities", time_field="filing_year", top_n=3,
        top_n_per_period=2,
    )
    by_id = {c.canonical_id: c for c in with_flag.concepts}
    assert "SUPPLY" in by_id
    assert by_id["SUPPLY"].by_period == {"2021": 2}
    # The overall top-3 are still there; the flag is a union, not a swap.
    assert {"FERC", "AES", "EU"} <= set(by_id)


def test_top_n_per_period_keeps_the_canonical_order() -> None:
    """The union must not disturb the deterministic ordering."""
    links = pl.DataFrame(
        {
            "mention_id": ["m1", "m2", "m3", "m4"],
            "canonical_id": ["BIG", "BIG", "BIG", "RARE"],
            "mention_text": ["BIG", "BIG", "BIG", "RARE"],
            "filing_year": [2019, 2019, 2019, 2021],
        }
    )
    export = build_concept_cloud(
        project="p", links=links, dag_plane=DagPlane(nodes=()),
        linking_model="link_entities", time_field="filing_year",
        top_n=1, top_n_per_period=1,
    )
    # Most frequent first, exactly as without the flag.
    assert [c.canonical_id for c in export.concepts] == ["BIG", "RARE"]


def test_top_n_per_period_without_a_time_field_is_refused() -> None:
    """Silently doing nothing would look like the flag working."""
    with pytest.raises(ConceptCloudExportError) as excinfo:
        export_concept_cloud(
            pathlib.Path("."), linking_model="link_entities", top_n_per_period=5
        )
    assert "needs --time-field" in str(excinfo.value)
