from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from stel.concept_cloud import (
    CONCEPT_CLOUD_SCHEMA_VERSION,
    ConceptCloudExport,
    parse_concept_cloud_export,
    placeholder_export,
)


def _bundle() -> dict[str, Any]:
    return json.loads(placeholder_export().to_json())


def test_placeholder_export_is_valid_and_round_trips() -> None:
    export = placeholder_export()
    raw = json.loads(export.to_json())
    assert raw["schema_version"] == CONCEPT_CLOUD_SCHEMA_VERSION
    # `from` is emitted as the aliased key, not `from_`.
    assert raw["dag_plane"]["edges"][0] == {
        "from": "source.dbt_ml_invoice_pipeline.link_entities",
        "to": "model.analytics.vendor_facts",
    }
    reparsed = parse_concept_cloud_export(raw)
    assert reparsed == export


def test_parse_rejects_unknown_schema_version() -> None:
    bad = _bundle()
    bad["schema_version"] = "999"
    with pytest.raises(ValueError, match="unsupported concept-cloud schema_version"):
        parse_concept_cloud_export(bad)


def test_parse_rejects_missing_schema_version() -> None:
    # An unversioned bundle must not be silently treated as the current schema.
    bad = _bundle()
    del bad["schema_version"]
    with pytest.raises(ValueError, match="must declare a schema_version"):
        parse_concept_cloud_export(bad)


def test_dag_node_accepts_arbitrary_dbt_resource_types() -> None:
    # Real dbt manifests carry seeds, snapshots, analyses, etc.; the plane must
    # project them, not reject them.
    bundle = _bundle()
    bundle["dag_plane"]["nodes"].append(
        {"id": "seed.p.currencies", "label": "currencies", "resource_type": "seed"}
    )
    export = parse_concept_cloud_export(bundle)
    assert any(n.resource_type == "seed" for n in export.dag_plane.nodes)


def test_parse_rejects_unknown_fields() -> None:
    bad = _bundle()
    bad["surprise"] = True
    with pytest.raises(ValidationError):
        parse_concept_cloud_export(bad)


def test_concept_edge_endpoints_must_be_known_concepts() -> None:
    bad = _bundle()
    bad["concept_edges"].append(
        {"source": "org:acme", "target": "org:does_not_exist", "relation_type": "x"}
    )
    with pytest.raises(ValidationError, match="unknown concept 'org:does_not_exist'"):
        parse_concept_cloud_export(bad)


def test_cross_layer_edge_must_reference_known_dag_node() -> None:
    bad = _bundle()
    bad["cross_layer_edges"].append(
        {"concept": "org:acme", "dag_node": "model.analytics.ghost"}
    )
    with pytest.raises(ValidationError, match="unknown dag node"):
        parse_concept_cloud_export(bad)


def test_dag_edge_must_reference_known_nodes() -> None:
    bad = _bundle()
    bad["dag_plane"]["edges"].append({"from": "model.analytics.vendor_facts", "to": "ghost"})
    with pytest.raises(ValidationError, match="unknown node 'ghost'"):
        parse_concept_cloud_export(bad)


def test_concept_edge_rejects_self_loop() -> None:
    bad = _bundle()
    bad["concept_edges"].append(
        {"source": "org:acme", "target": "org:acme", "relation_type": "x"}
    )
    with pytest.raises(ValidationError, match="two distinct concepts"):
        parse_concept_cloud_export(bad)


def test_frequency_and_scores_are_bounded() -> None:
    with pytest.raises(ValidationError):
        ConceptCloudExport.model_validate(
            {
                **_bundle(),
                "concepts": [
                    {
                        "canonical_id": "org:acme",
                        "display": "Acme",
                        "frequency": 0,  # must be >= 1
                        "provenance": {"model": "link_entities"},
                    }
                ],
                "concept_edges": [],
                "cross_layer_edges": [],
            }
        )


def test_duplicate_canonical_ids_are_rejected() -> None:
    dup = {
        "canonical_id": "org:acme",
        "display": "Acme",
        "frequency": 1,
        "provenance": {"model": "link_entities"},
    }
    with pytest.raises(ValidationError, match="canonical_ids must be unique"):
        ConceptCloudExport.model_validate(
            {
                **_bundle(),
                "concepts": [dup, dup],
                "concept_edges": [],
                "cross_layer_edges": [],
            }
        )
