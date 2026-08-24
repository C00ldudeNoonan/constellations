"""Build a `ConceptCloudExport` from a project's real artifacts (#255, M3).

This is the "export job": the three-way join the feature turns on. It reads the
entity-linking output (canonical concepts + mention→canonical map), optionally
the NLP entity table (for labels/display text) and the relation table (typed
concept-to-concept edges), and a DAG (the downstream dbt manifest, or stel's
own) for the ground plane. The join logic (`build_concept_cloud`) is a pure
function over polars frames so it is exercised deterministically without a
warehouse; `export_concept_cloud` is the thin wrapper that reads the frames
through the active adapter.

Only intended output tables are read — no credentials, prompts, or raw documents
enter the bundle. Display text is used only if the operator opted into it
upstream; otherwise a concept shows its canonical id.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import polars as pl

from ..config import load_project
from ..manifest import MANIFEST_FILENAME, write_manifest
from .schema import (
    Concept,
    ConceptCloudExport,
    ConceptEdge,
    ConceptEdgeMethod,
    CrossLayerEdge,
    DagEdge,
    DagNode,
    DagPlane,
    DimensionDef,
    Position,
    Provenance,
)

# stel DAG node kind -> a resource_type label for the plane.
_STEL_KIND_RESOURCE = {"source": "source", "model": "model", "search_index": "exposure"}
# dbt resource kinds worth placing on the plane (tests/units add noise).
_DBT_PLANE_RESOURCE_TYPES = {
    "model", "source", "seed", "snapshot", "exposure", "analysis", "metric",
}


class ConceptCloudExportError(Exception):
    pass


def _col(frame: pl.DataFrame, name: str) -> pl.Series | None:
    return frame[name] if name in frame.columns else None


def build_concept_cloud(
    *,
    project: str,
    links: pl.DataFrame,
    dag_plane: DagPlane,
    entities: pl.DataFrame | None = None,
    relations: pl.DataFrame | None = None,
    linking_node_id: str | None = None,
    linking_model: str = "link_entities",
    top_n: int = 200,
    generated_at: str | None = None,
    statuses: tuple[str, ...] = ("matched", "ambiguous"),
    embeddings: pl.DataFrame | None = None,
    vector_field: str = "embedding",
    query_log: pl.DataFrame | None = None,
    dimension_columns: dict[str, tuple[pl.DataFrame, str]] | None = None,
) -> ConceptCloudExport:
    """Assemble a bundle from entity-linking (+optional entities/relations) frames.

    Concepts are aggregated at `canonical_id` grain (unmatched rows, which carry a
    null canonical id, are dropped); the cloud is capped to the `top_n` most
    frequent concepts for readability. Relation rows are canonicalized through the
    mention→canonical map into typed concept edges.
    """
    generated_at = generated_at or datetime.now(UTC).isoformat()
    node_ids = {node.id for node in dag_plane.nodes}

    concepts, canonical_of = _aggregate_concepts(
        links, entities=entities, linking_model=linking_model,
        linking_node_id=linking_node_id if linking_node_id in node_ids else None,
        top_n=top_n, statuses=statuses,
    )
    kept = {c.canonical_id for c in concepts}
    concept_edges = _aggregate_edges(relations, canonical_of, kept)

    if embeddings is not None:
        positions = concept_positions(
            links, embeddings, vector_field=vector_field, kept=kept
        )
        concepts = [
            c.model_copy(update={"position": positions.get(c.canonical_id)})
            for c in concepts
        ]

    dimension_defs: list[DimensionDef] = []
    values_by_concept: dict[str, dict[str, str]] = defaultdict(dict)
    if query_log is not None:
        heat_def, heat_values = retrieval_dimension(links, query_log, kept=kept)
        if heat_def is not None:
            dimension_defs.append(heat_def)
            for cid, value in heat_values.items():
                values_by_concept[cid][heat_def.name] = value
    for name, (frame, value_column) in sorted((dimension_columns or {}).items()):
        column_def, column_values = column_dimension(
            name, frame, value_column, kept=kept
        )
        if column_def is not None:
            dimension_defs.append(column_def)
            for cid, value in column_values.items():
                values_by_concept[cid][name] = value
    if values_by_concept:
        concepts = [
            c.model_copy(update={"dimensions": values_by_concept.get(c.canonical_id, {})})
            for c in concepts
        ]
    cross_layer_edges = (
        tuple(
            CrossLayerEdge(concept=c.canonical_id, dag_node=linking_node_id)
            for c in concepts
        )
        if linking_node_id in node_ids
        else ()
    )
    return ConceptCloudExport(
        generated_at=generated_at,
        project=project,
        dag_plane=dag_plane,
        concepts=tuple(concepts),
        concept_edges=concept_edges,
        cross_layer_edges=cross_layer_edges,
        dimensions=tuple(dimension_defs),
    )


def _aggregate_concepts(
    links: pl.DataFrame,
    *,
    entities: pl.DataFrame | None,
    linking_model: str,
    linking_node_id: str | None,
    top_n: int,
    statuses: tuple[str, ...],
) -> tuple[list[Concept], dict[str, str]]:
    if "canonical_id" not in links.columns or "mention_id" not in links.columns:
        raise ConceptCloudExportError(
            "entity-linking output must have `canonical_id` and `mention_id` columns"
        )
    frame = links.filter(pl.col("canonical_id").is_not_null())
    if "status" in frame.columns:
        frame = frame.filter(pl.col("status").is_in(list(statuses)))
    if frame.height == 0:
        return [], {}

    # Enrich label/text from the entity table when linking did not carry them.
    label_by_mention, text_by_mention = _mention_enrichment(frame, entities)

    canonical_of: dict[str, str] = {}
    rows_by_canonical: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in frame.iter_rows(named=True):
        cid = str(row["canonical_id"])
        mid = str(row["mention_id"])
        canonical_of[mid] = cid
        rows_by_canonical[cid].append(row)

    concepts: list[Concept] = []
    for cid, rows in rows_by_canonical.items():
        documents = {
            r.get("document_id") for r in rows if r.get("document_id") is not None
        }
        label = _first(
            [r.get("label") for r in rows]
            + [label_by_mention.get(str(r["mention_id"])) for r in rows]
        )
        display = _first(
            [r.get("mention_text") for r in rows]
            + [text_by_mention.get(str(r["mention_id"])) for r in rows]
        ) or cid
        ambiguous = any(r.get("status") == "ambiguous" for r in rows)
        scores = [
            float(score) for r in rows
            if isinstance((score := r.get("match_score")), (int, float))
            and not isinstance(score, bool)
        ]
        source_node = linking_node_id if linking_node_id else None
        concepts.append(Concept(
            canonical_id=cid,
            display=str(display),
            label=str(label) if label is not None else None,
            frequency=len(rows),
            link_status="ambiguous" if ambiguous else "matched",
            match_score=max(scores) if scores else None,
            provenance=Provenance(
                model=linking_model, source_node=source_node, documents=len(documents)
            ),
        ))

    # Deterministic top-N: most frequent first, canonical_id breaks ties.
    concepts.sort(key=lambda c: (-c.frequency, c.canonical_id))
    concepts = concepts[: max(0, top_n)]
    kept = {c.canonical_id for c in concepts}
    canonical_of = {m: c for m, c in canonical_of.items() if c in kept}
    return concepts, canonical_of


def _mention_enrichment(
    links: pl.DataFrame, entities: pl.DataFrame | None
) -> tuple[dict[str, str], dict[str, str]]:
    label: dict[str, str] = {}
    text: dict[str, str] = {}
    if entities is None or "entity_id" not in entities.columns:
        return label, text
    lcol = _col(entities, "label")
    tcol = _col(entities, "entity_text")
    ids = entities["entity_id"]
    for i in range(entities.height):
        eid = str(ids[i])
        if lcol is not None and lcol[i] is not None:
            label[eid] = str(lcol[i])
        if tcol is not None and tcol[i] is not None:
            text[eid] = str(tcol[i])
    return label, text


def _aggregate_edges(
    relations: pl.DataFrame | None,
    canonical_of: dict[str, str],
    kept: set[str],
) -> tuple[ConceptEdge, ...]:
    if relations is None or relations.height == 0 or not kept:
        return ()
    required = {"subject_mention_id", "object_mention_id", "relation_type"}
    if not required.issubset(relations.columns):
        return ()
    EdgeKey = tuple[str, str, str, str, bool]
    weights: dict[EdgeKey, int] = defaultdict(int)
    confidences: dict[EdgeKey, float | None] = {}
    for row in relations.iter_rows(named=True):
        s = canonical_of.get(str(row["subject_mention_id"]))
        t = canonical_of.get(str(row["object_mention_id"]))
        if s is None or t is None or s == t:
            continue
        directed = bool(row.get("directed", False))
        # Undirected edges collapse onto a sorted pair so A–B and B–A aggregate.
        pair = (s, t) if directed else tuple(sorted((s, t)))
        method = str(row.get("method") or "co_occurrence")
        key: EdgeKey = (pair[0], pair[1], str(row["relation_type"]), method, directed)
        weights[key] += 1
        conf = row.get("confidence")
        if isinstance(conf, (int, float)) and not isinstance(conf, bool):
            prev = confidences.get(key)
            confidences[key] = float(conf) if prev is None else max(prev, float(conf))
        else:
            confidences.setdefault(key, None)
    return tuple(
        ConceptEdge(
            source=src, target=tgt, relation_type=rtype,
            method=cast(ConceptEdgeMethod, method), directed=directed,
            weight=weights[(src, tgt, rtype, method, directed)],
            confidence=confidences.get((src, tgt, rtype, method, directed)),
        )
        for (src, tgt, rtype, method, directed) in sorted(weights)
    )



# Coordinate radius the projection is scaled to. Matches the artifact's scene
# scale (the DAG plane grid spreads ~60 units per node), so semantic layout
# and lineage mode share one coordinate system.
_POSITION_RADIUS = 120.0
_RETRIEVAL_DIMENSION = "retrieval"


def concept_positions(
    links: pl.DataFrame,
    embeddings: pl.DataFrame,
    *,
    vector_field: str = "embedding",
    kept: set[str],
) -> dict[str, Position]:
    """Semantic coordinates: mention-vector centroids projected to 3D (#345).

    The embed relation already carries every mention's vector (the upstream
    record's columns survive embedding), so this is a join and a mean — no
    provider calls. PCA rather than UMAP: deterministic, dependency-light, and
    at <=2,000 concepts the difference does not earn a heavyweight import.
    Only coordinates leave this function; vectors never enter the bundle.
    """
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - numpy rides other extras
        raise ConceptCloudExportError(
            "Baked positions require numpy (installed by most stel extras); "
            "install it or export without --embed-model"
        ) from error

    if "mention_id" not in embeddings.columns or vector_field not in embeddings.columns:
        raise ConceptCloudExportError(
            f"embed model output must have `mention_id` and `{vector_field}` columns"
        )
    joined = (
        links.filter(pl.col("canonical_id").is_in(sorted(kept)))
        .select("mention_id", "canonical_id")
        .join(
            embeddings.select("mention_id", vector_field),
            on="mention_id",
            how="inner",
        )
    )
    if joined.height == 0:
        return {}
    # Element-wise centroid per concept. Deliberately not `.list.mean()`,
    # which averages *within* one row's vector; the centroid is the mean
    # across a concept's mention vectors, per dimension.
    by_concept: dict[str, list[Any]] = defaultdict(list)
    for row in joined.iter_rows(named=True):
        vector = row[vector_field]
        if vector is not None:
            by_concept[str(row["canonical_id"])].append(vector)
    if len(by_concept) < 2:
        return {}
    ids = sorted(by_concept)
    try:
        matrix = np.array(
            [np.mean(np.array(by_concept[cid], dtype=np.float64), axis=0) for cid in ids]
        )
    except ValueError as error:
        raise ConceptCloudExportError(
            "mention vectors must share one dimensionality to be projected"
        ) from error
    if matrix.ndim != 2:
        return {}
    centered = matrix - matrix.mean(axis=0)
    # Deterministic PCA via SVD; sign fixed so the largest-magnitude loading on
    # each axis is positive, since SVD signs are otherwise arbitrary.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:3]
    for i in range(components.shape[0]):
        if components[i][np.argmax(np.abs(components[i]))] < 0:
            components[i] = -components[i]
    projected = centered @ components.T
    if projected.shape[1] < 3:
        projected = np.pad(projected, ((0, 0), (0, 3 - projected.shape[1])))
    peak = float(np.abs(projected).max()) or 1.0
    projected = projected * (_POSITION_RADIUS / peak)
    return {
        cid: Position(x=float(x), y=float(y), z=float(z))
        for cid, (x, y, z) in zip(ids, projected, strict=True)
    }


def retrieval_dimension(
    links: pl.DataFrame,
    query_log: pl.DataFrame,
    *,
    kept: set[str],
) -> tuple[DimensionDef | None, dict[str, str]]:
    """The usage-feedback dimension: retrieval heat from the MCP query log.

    Joins `returned_chunk_ids` through the linking rows to concepts and
    buckets hit counts into hot/warm/cold/never. Aggregate-only by
    construction — counts are the only thing read; query text and principal
    ids never leave the warehouse frame (issue #345, #329 rule 1).
    `never` is deliberate and arguably the most actionable value on the map:
    a concept agents cannot reach.
    """
    if "returned_chunk_ids" not in query_log.columns:
        return None, {}
    join_column = next(
        (c for c in ("chunk_id", "document_id") if c in links.columns), None
    )
    if join_column is None:
        return None, {}
    hits = (
        query_log.select(pl.col("returned_chunk_ids").alias("returned"))
        .explode("returned")
        .drop_nulls()
        .group_by("returned")
        .len()
        .rename({"returned": join_column, "len": "hits"})
    )
    per_concept = (
        links.filter(pl.col("canonical_id").is_in(sorted(kept)))
        .select("canonical_id", join_column)
        .join(hits, on=join_column, how="left")
        .group_by("canonical_id")
        .agg(pl.col("hits").fill_null(0).sum().alias("hits"))
    )
    counts = {
        str(row["canonical_id"]): int(row["hits"])
        for row in per_concept.iter_rows(named=True)
    }
    # Every kept concept gets a value; ones with no joinable rows are `never`.
    counts = {cid: counts.get(cid, 0) for cid in kept}
    positive = sorted(count for count in counts.values() if count > 0)
    definition = DimensionDef(
        name=_RETRIEVAL_DIMENSION,
        values=("hot", "warm", "cold", "never"),
        source="query_log",
        description="How often agents' queries returned this concept's chunks",
    )
    if not positive:
        return definition, {cid: "never" for cid in counts}
    # Tertiles over concepts that were retrieved at all; deterministic bounds.
    low = positive[len(positive) // 3]
    high = positive[(2 * len(positive)) // 3]

    def bucket(count: int) -> str:
        if count == 0:
            return "never"
        if count <= low:
            return "cold"
        if count <= high:
            return "warm"
        return "hot"

    return definition, {cid: bucket(count) for cid, count in counts.items()}


def column_dimension(
    name: str,
    frame: pl.DataFrame,
    value_column: str,
    *,
    kept: set[str],
) -> tuple[DimensionDef | None, dict[str, str]]:
    """A declared pipeline dimension: any concept-keyed categorical column.

    This is where `llm:` enum fields (#304) flow in for free — the label set
    is already declared and validated upstream; here it just becomes a color.
    A concept with multiple rows takes the most frequent value, ties broken
    lexically for determinism.
    """
    if "canonical_id" not in frame.columns or value_column not in frame.columns:
        raise ConceptCloudExportError(
            f"dimension '{name}' needs `canonical_id` and `{value_column}` columns"
        )
    rows = (
        frame.filter(
            pl.col("canonical_id").is_in(sorted(kept))
            & pl.col(value_column).is_not_null()
        )
        .group_by("canonical_id", value_column)
        .len()
        .sort(["canonical_id", "len", value_column], descending=[False, True, False])
        .group_by("canonical_id", maintain_order=True)
        .first()
    )
    values = {
        str(row["canonical_id"]): str(row[value_column])
        for row in rows.iter_rows(named=True)
    }
    if not values:
        return None, {}
    definition = DimensionDef(
        name=name,
        values=tuple(sorted(set(values.values()))),
        source="column",
    )
    return definition, values


def _first(values: list[object]) -> object | None:
    for v in values:
        if v is not None and str(v).strip():
            return v
    return None


def dag_plane_from_stel_manifest(manifest: dict[str, Any]) -> tuple[DagPlane, dict[str, str]]:
    """Plane from stel's own manifest. Returns the plane and a name→node-id map
    (so the caller can resolve the linking model's node)."""
    dag = manifest.get("dag", {})
    id_by_name: dict[str, str] = {}
    nodes: list[DagNode] = []
    for entry in dag.get("nodes", []):
        name = entry["name"]
        node_id = entry.get("unique_id") or name
        id_by_name[name] = node_id
        nodes.append(DagNode(
            id=node_id, label=name,
            resource_type=_STEL_KIND_RESOURCE.get(entry.get("kind", "model"), "model"),
        ))
    edges = []
    for edge in dag.get("edges", []):
        a, b = edge[0], edge[1]
        # v1 edges are name pairs; v2 are unique_id pairs. Normalize to ids.
        edges.append(DagEdge.model_validate(
            {"from": id_by_name.get(a, a), "to": id_by_name.get(b, b)}
        ))
    return DagPlane(nodes=tuple(nodes), edges=tuple(edges)), id_by_name


def dag_plane_from_dbt_manifest(manifest: dict[str, Any]) -> DagPlane:
    """Plane from a downstream dbt `manifest.json` (sources + models + exposures,
    lineage from `parent_map`)."""
    entries: dict[str, dict[str, Any]] = {}
    for section in ("nodes", "sources", "exposures"):
        entries.update(manifest.get(section, {}))
    keep = {
        uid: e for uid, e in entries.items()
        if e.get("resource_type") in _DBT_PLANE_RESOURCE_TYPES
    }
    nodes = tuple(
        DagNode(id=uid, label=e.get("name", uid), resource_type=e["resource_type"])
        for uid, e in keep.items()
    )
    edges = []
    for child, parents in manifest.get("parent_map", {}).items():
        if child not in keep:
            continue
        for parent in parents:
            if parent in keep:
                edges.append(DagEdge.model_validate({"from": parent, "to": child}))
    return DagPlane(nodes=nodes, edges=tuple(edges))


def export_concept_cloud(
    project_dir: str | Path,
    *,
    linking_model: str,
    relation_model: str | None = None,
    entity_model: str | None = None,
    dbt_manifest: str | Path | None = None,
    source_name: str | None = None,
    target: str | None = None,
    profiles_dir: str | Path | None = None,
    top_n: int = 200,
    embed_model: str | None = None,
    vector_field: str = "embedding",
    with_query_log: bool = False,
    dimension_specs: dict[str, str] | None = None,
) -> ConceptCloudExport:
    """Read the project's tables through the active adapter and build a bundle.

    `source_name` must match whatever `emit-dbt-sources --source-name` wrote
    into the downstream project, since that is the name its manifest records.
    """
    from ..adapters import create_adapter
    from ..dbt_export import default_dbt_source_name
    from ..profile import resolve_profile

    project_path = Path(project_dir)
    project, _, _ = load_project(project_path)
    resolved = resolve_profile(
        project, project_path, target=target,
        profiles_dir=Path(profiles_dir) if profiles_dir else None,
    )

    if dbt_manifest is not None:
        manifest = json.loads(Path(dbt_manifest).read_text(encoding="utf-8"))
        dag_plane = dag_plane_from_dbt_manifest(manifest)
        source = source_name or default_dbt_source_name(project.name)
        linking_node_id = f"source.{source}.{linking_model}"
        # The cross-layer edges are the entire reason to pass a dbt manifest,
        # and they are built only for a node id that resolves. A name that does
        # not match what the consumer's project actually declares used to
        # render a cloud with the DAG join silently missing, which looks like a
        # working export.
        if linking_node_id not in {node.id for node in dag_plane.nodes}:
            declared = sorted(
                node.id.split(".")[1]
                for node in dag_plane.nodes
                if node.id.startswith("source.")
            )
            raise ConceptCloudExportError(
                f"'{linking_node_id}' is not in the dbt manifest, so the "
                f"concept-to-DAG edges would be empty. Pass --source-name to "
                f"match what `emit-dbt-sources --source-name` wrote into that "
                f"project. Sources it declares: {sorted(set(declared)) or '(none)'}."
            )
    else:
        target_dir = (project_path / project.target_path).resolve()
        manifest_path = target_dir / MANIFEST_FILENAME
        if not manifest_path.exists():
            write_manifest(project_path, target=target,
                           profiles_dir=Path(profiles_dir) if profiles_dir else None)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dag_plane, id_by_name = dag_plane_from_stel_manifest(manifest)
        linking_node_id = id_by_name.get(linking_model)

    dimension_columns: dict[str, tuple[pl.DataFrame, str]] = {}
    with create_adapter(resolved.warehouse, project_dir=project_path) as adapter:
        links = adapter.read_table(linking_model)
        relations = adapter.read_table(relation_model) if relation_model else None
        entities = adapter.read_table(entity_model) if entity_model else None
        embeddings = adapter.read_table(embed_model) if embed_model else None
        query_log = None
        if with_query_log:
            # The log is opt-in and may not exist yet; an absent relation is
            # "no usage data", not an export failure.
            log_config = resolved.mcp_query_log
            relation = log_config.relation if log_config else "stel_mcp_query_log"
            try:
                query_log = adapter.read_table(relation)
            except Exception:
                query_log = None
        for name, spec in sorted((dimension_specs or {}).items()):
            model_name, _, value_column = spec.partition(".")
            if not model_name or not value_column:
                raise ConceptCloudExportError(
                    f"dimension '{name}' must be `model.column`, got {spec!r}"
                )
            dimension_columns[name] = (
                adapter.read_table(model_name),
                value_column,
            )

    return build_concept_cloud(
        project=project.name,
        links=links,
        dag_plane=dag_plane,
        entities=entities,
        relations=relations,
        linking_node_id=linking_node_id,
        linking_model=linking_model,
        top_n=top_n,
        embeddings=embeddings,
        vector_field=vector_field,
        query_log=query_log,
        dimension_columns=dimension_columns or None,
    )
