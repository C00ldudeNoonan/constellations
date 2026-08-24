"""Versioned Pydantic contract for the concept-cloud export bundle (issue #255).

One `ConceptCloudExport` is the entire input the static visualization needs: the
DAG plane (from the downstream dbt manifest), the canonical concept cloud (from
the entity-linking output), typed concept-to-concept edges (from the relation
grain), and best-effort cross-layer links tying a concept to the dbt node that
serves it. Field names and types mirror what stel actually publishes so the
export job is a straight projection, not a translation.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Bumped when the bundle shape changes so the artifact and the export job can
# evolve independently; the artifact refuses a bundle it does not understand.
# v2 (issue #345): baked 3D positions and categorical dimensions.
CONCEPT_CLOUD_SCHEMA_VERSION = "2"

# Mirrors stel.text.relations.RelationMethod (proximity vs. asserted edges).
ConceptEdgeMethod = Literal["co_occurrence", "rule", "model_assertion"]
# A concept in the cloud is a canonical entity, so it is always linked — either
# a clean match or a match the resolver flagged ambiguous. Entity-linking's third
# status, `unmatched`, has a null canonical id (no canonical concept), so those
# mentions have no node and are excluded by the export job.
LinkStatus = Literal["matched", "ambiguous"]


def _require_non_empty(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must not be empty")
    return normalized


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DagNode(_Frozen):
    """One node on the static DAG plane. `id` is the dbt `unique_id`;
    `resource_type` is the dbt resource kind verbatim (source, model, seed,
    snapshot, exposure, analysis, …) — kept free-form so any downstream dbt
    manifest projects cleanly, with the artifact color-coding known kinds."""

    id: str
    label: str
    resource_type: str
    # Optional coarse grouping for plane layout (e.g. staging/marts); free-form.
    layer: str | None = None

    @field_validator("id", "label", "resource_type")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        return _require_non_empty(value)


class DagEdge(_Frozen):
    """A directed lineage edge between two DAG nodes, by `unique_id`."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    # `from` is a Python keyword, so alias the JSON key.
    from_: str = Field(alias="from")
    to: str

    @field_validator("from_", "to")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        return _require_non_empty(value)


class DagPlane(_Frozen):
    nodes: tuple[DagNode, ...]
    edges: tuple[DagEdge, ...] = ()

    @model_validator(mode="after")
    def _edges_reference_known_nodes(self) -> DagPlane:
        ids = {node.id for node in self.nodes}
        if len(ids) != len(self.nodes):
            raise ValueError("dag_plane node ids must be unique")
        for edge in self.edges:
            for endpoint in (edge.from_, edge.to):
                if endpoint not in ids:
                    raise ValueError(
                        f"dag_plane edge references unknown node '{endpoint}'"
                    )
        return self


class Provenance(_Frozen):
    """Where a concept came from, at model/table grain (per-column provenance is
    not available from the manifest — see the issue's resolved open questions)."""

    model: str
    # The downstream dbt source node this concept's table maps to via
    # `emit-dbt-sources` (`source.dbt_ml_<project>.<model>`); None until stitched.
    source_node: str | None = None
    documents: int = Field(ge=0, default=0)

    @field_validator("model")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        return _require_non_empty(value)


class Position(_Frozen):
    """Baked 3D coordinates from the export-time projection (issue #345).

    Computed as the centroid of the concept's mention embeddings projected to
    3D, so proximity means how the corpus uses a concept — the projection runs
    in Python at export time and only coordinates enter the bundle, never
    text or raw vectors. Null on a concept means no embedding data was
    available; the artifact falls back to force layout for those nodes."""

    x: float
    y: float
    z: float

    @field_validator("x", "y", "z")
    @classmethod
    def _finite(cls, value: float) -> float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("position coordinates must be finite")
        return value


class DimensionDef(_Frozen):
    """One categorical dimension concepts can be colored and filtered by
    (issue #345). The value set is closed and declared here, so the artifact
    can build a stable legend without scanning every concept.

    `source` records where the values came from: `query_log` dimensions are
    aggregates derived from the MCP query log (retrieval heat — never query
    text or principals); `column` dimensions are concept-keyed categorical
    columns from the pipeline, e.g. an `llm:` enum field."""

    name: str
    values: tuple[str, ...]
    source: Literal["query_log", "column"]
    description: str | None = None

    @field_validator("name")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        return _require_non_empty(value)

    @field_validator("values")
    @classmethod
    def _values_non_empty_and_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("dimension must declare at least one value")
        cleaned = tuple(_require_non_empty(value) for value in values)
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("dimension values must be unique")
        return cleaned


class Concept(_Frozen):
    """One canonical entity in the cloud. Keyed on `canonical_id` from the
    entity-linking output; `display` is human-readable only when the operator
    opted into mention/entity text, else it falls back to the id or label."""

    canonical_id: str
    display: str
    label: str | None = None
    namespace: str | None = None
    # COUNT of mentions for this canonical id; drives node size. >= 1.
    frequency: int = Field(ge=1)
    link_status: LinkStatus = "matched"
    # Entity-linking match score; null for exact-alias matches.
    match_score: float | None = None
    provenance: Provenance
    # Baked semantic coordinates; null falls back to force layout (#345).
    position: Position | None = None
    # Dimension name -> value. Every key must name a DimensionDef on the
    # bundle and every value must be in that def's declared set; a concept
    # absent from a dimension simply omits the key.
    dimensions: dict[str, str] = Field(default_factory=dict)

    @field_validator("canonical_id", "display")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        return _require_non_empty(value)

    @field_validator("match_score")
    @classmethod
    def _score_in_unit_range(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("match_score must be between 0 and 1")
        return value


class ConceptEdge(_Frozen):
    """A typed concept-to-concept edge, canonicalized from the relation grain.
    `weight` is the count of underlying relations; `confidence` is populated only
    for `model_assertion` edges."""

    source: str
    target: str
    relation_type: str
    directed: bool = False
    method: ConceptEdgeMethod = "co_occurrence"
    weight: int = Field(ge=1, default=1)
    confidence: float | None = None

    @field_validator("source", "target", "relation_type")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        return _require_non_empty(value)

    @field_validator("confidence")
    @classmethod
    def _confidence_in_unit_range(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return value

    @model_validator(mode="after")
    def _no_self_loops(self) -> ConceptEdge:
        if self.source == self.target:
            raise ValueError("concept edge must connect two distinct concepts")
        return self


class CrossLayerEdge(_Frozen):
    """Links a concept to the DAG node that serves it. `column` is best-effort
    (null unless the operator declared matching fields)."""

    concept: str
    dag_node: str
    column: str | None = None

    @field_validator("concept", "dag_node")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        return _require_non_empty(value)


class ConceptCloudExport(_Frozen):
    """The complete, self-contained input for the concept-cloud artifact."""

    schema_version: Literal["2"] = CONCEPT_CLOUD_SCHEMA_VERSION
    generated_at: str
    project: str
    dag_plane: DagPlane
    concepts: tuple[Concept, ...] = ()
    concept_edges: tuple[ConceptEdge, ...] = ()
    cross_layer_edges: tuple[CrossLayerEdge, ...] = ()
    # Declared categorical dimensions (issue #345); order is picker order.
    dimensions: tuple[DimensionDef, ...] = ()

    @field_validator("generated_at", "project")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        return _require_non_empty(value)

    @model_validator(mode="after")
    def _edges_reference_known_endpoints(self) -> ConceptCloudExport:
        concept_ids = {concept.canonical_id for concept in self.concepts}
        if len(concept_ids) != len(self.concepts):
            raise ValueError("concept canonical_ids must be unique")
        dimension_defs = {dim.name: set(dim.values) for dim in self.dimensions}
        if len(dimension_defs) != len(self.dimensions):
            raise ValueError("dimension names must be unique")
        for concept in self.concepts:
            for name, value in concept.dimensions.items():
                allowed = dimension_defs.get(name)
                if allowed is None:
                    raise ValueError(
                        f"concept '{concept.canonical_id}' uses undeclared "
                        f"dimension '{name}'"
                    )
                if value not in allowed:
                    raise ValueError(
                        f"concept '{concept.canonical_id}' has value {value!r} "
                        f"outside dimension '{name}' declared set"
                    )
        node_ids = {node.id for node in self.dag_plane.nodes}
        for edge in self.concept_edges:
            for endpoint in (edge.source, edge.target):
                if endpoint not in concept_ids:
                    raise ValueError(
                        f"concept_edge references unknown concept '{endpoint}'"
                    )
        for link in self.cross_layer_edges:
            if link.concept not in concept_ids:
                raise ValueError(
                    f"cross_layer_edge references unknown concept '{link.concept}'"
                )
            if link.dag_node not in node_ids:
                raise ValueError(
                    f"cross_layer_edge references unknown dag node '{link.dag_node}'"
                )
        return self

    def to_json(self) -> str:
        """Serialize to the on-disk bundle JSON (aliased keys, e.g. `from`)."""
        return self.model_dump_json(by_alias=True)


def parse_concept_cloud_export(data: dict[str, Any]) -> ConceptCloudExport:
    """Validate a raw bundle mapping into a `ConceptCloudExport`. The bundle must
    declare a `schema_version`, and it must match this build's — an unversioned
    or mismatched artifact is rejected rather than silently treated as current."""
    if not isinstance(data, dict) or "schema_version" not in data:
        raise ValueError(
            "concept-cloud bundle must declare a schema_version; "
            f"this build understands {CONCEPT_CLOUD_SCHEMA_VERSION!r}"
        )
    version = data["schema_version"]
    if version != CONCEPT_CLOUD_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported concept-cloud schema_version {version!r}; "
            f"this build understands {CONCEPT_CLOUD_SCHEMA_VERSION!r}"
        )
    return ConceptCloudExport.model_validate(data)


def placeholder_export() -> ConceptCloudExport:
    """A small, valid bundle for artifact development and tests. It mirrors the
    invoice example: a two-node dbt plane, three vendor concepts, one
    co-occurrence edge, and one cross-layer link."""
    return ConceptCloudExport(
        generated_at="2026-08-04T00:00:00Z",
        project="invoice_pipeline",
        dag_plane=DagPlane(
            nodes=(
                DagNode(
                    id="source.dbt_ml_invoice_pipeline.link_entities",
                    label="link_entities",
                    resource_type="source",
                    layer="stel",
                ),
                DagNode(
                    id="model.analytics.vendor_facts",
                    label="vendor_facts",
                    resource_type="model",
                    layer="marts",
                ),
            ),
            edges=(
                DagEdge.model_validate(
                    {
                        "from": "source.dbt_ml_invoice_pipeline.link_entities",
                        "to": "model.analytics.vendor_facts",
                    }
                ),
            ),
        ),
        concepts=(
            Concept(
                canonical_id="org:acme",
                display="Acme Corp",
                label="ORG",
                frequency=12,
                provenance=Provenance(
                    model="link_entities",
                    source_node="source.dbt_ml_invoice_pipeline.link_entities",
                    documents=8,
                ),
            ),
            Concept(
                canonical_id="org:globex",
                display="Globex",
                label="ORG",
                frequency=5,
                link_status="ambiguous",
                match_score=0.71,
                provenance=Provenance(
                    model="link_entities",
                    source_node="source.dbt_ml_invoice_pipeline.link_entities",
                    documents=4,
                ),
            ),
            Concept(
                canonical_id="gpe:new_york",
                display="New York",
                label="GPE",
                frequency=3,
                provenance=Provenance(model="link_entities", documents=3),
            ),
        ),
        concept_edges=(
            ConceptEdge(
                source="org:acme",
                target="gpe:new_york",
                relation_type="co_occurs_with",
                method="co_occurrence",
                weight=4,
            ),
        ),
        cross_layer_edges=(
            CrossLayerEdge(
                concept="org:acme",
                dag_node="model.analytics.vendor_facts",
            ),
        ),
    )
