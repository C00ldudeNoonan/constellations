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
    target: str | None = None,
    profiles_dir: str | Path | None = None,
    top_n: int = 200,
) -> ConceptCloudExport:
    """Read the project's tables through the active adapter and build a bundle."""
    from ..adapters import create_adapter
    from ..profile import resolve_profile

    project_path = Path(project_dir)
    project, _, _ = load_project(project_path)
    resolved = resolve_profile(
        project, project_path, target=target,
        profiles_dir=Path(profiles_dir) if profiles_dir else None,
    )

    if dbt_manifest is not None:
        manifest = json.loads(Path(dbt_manifest).read_text())
        dag_plane = dag_plane_from_dbt_manifest(manifest)
        linking_node_id = f"source.dbt_ml_{project.name}.{linking_model}"
    else:
        target_dir = (project_path / project.target_path).resolve()
        manifest_path = target_dir / MANIFEST_FILENAME
        if not manifest_path.exists():
            write_manifest(project_path, target=target,
                           profiles_dir=Path(profiles_dir) if profiles_dir else None)
        manifest = json.loads(manifest_path.read_text())
        dag_plane, id_by_name = dag_plane_from_stel_manifest(manifest)
        linking_node_id = id_by_name.get(linking_model)

    with create_adapter(resolved.warehouse, project_dir=project_path) as adapter:
        links = adapter.read_table(linking_model)
        relations = adapter.read_table(relation_model) if relation_model else None
        entities = adapter.read_table(entity_model) if entity_model else None

    return build_concept_cloud(
        project=project.name,
        links=links,
        dag_plane=dag_plane,
        entities=entities,
        relations=relations,
        linking_node_id=linking_node_id,
        linking_model=linking_model,
        top_n=top_n,
    )
