"""Concept-cloud visualization export contract (issue #255).

The visualization renders stel's extracted concepts as a 3D cloud above a
static 2D plane of the downstream dbt DAG. This package defines the versioned
JSON **export contract** that decouples the two halves of the feature: a
self-contained static artifact renders one of these bundles, and an export job
(reading a project's manifests + entity/linking/relation relations) produces
them. Only the contract lives here — no rendering, no warehouse access.

The bundle is deliberately credential- and PII-safe: it carries only canonical
concept identity, aggregate frequencies, typed edges, and manifest-level DAG
lineage. Human-readable concept text is included only when the operator opted
into it upstream (`include_text`); otherwise `display` falls back to the
canonical id or label.
"""
from __future__ import annotations

from .artifact import render_concept_cloud, write_concept_cloud
from .demo import demo_export
from .export import (
    ConceptCloudExportError,
    build_concept_cloud,
    dag_plane_from_dbt_manifest,
    dag_plane_from_stel_manifest,
    export_concept_cloud,
)
from .schema import (
    CONCEPT_CLOUD_SCHEMA_VERSION,
    Concept,
    ConceptCloudExport,
    ConceptEdge,
    CrossLayerEdge,
    DagEdge,
    DagNode,
    DagPlane,
    Provenance,
    parse_concept_cloud_export,
    placeholder_export,
)

__all__ = [
    "CONCEPT_CLOUD_SCHEMA_VERSION",
    "Concept",
    "ConceptCloudExport",
    "ConceptCloudExportError",
    "ConceptEdge",
    "CrossLayerEdge",
    "DagEdge",
    "DagNode",
    "DagPlane",
    "Provenance",
    "build_concept_cloud",
    "dag_plane_from_dbt_manifest",
    "dag_plane_from_stel_manifest",
    "demo_export",
    "export_concept_cloud",
    "parse_concept_cloud_export",
    "placeholder_export",
    "render_concept_cloud",
    "write_concept_cloud",
]
