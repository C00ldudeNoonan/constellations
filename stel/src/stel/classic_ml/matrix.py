"""Matrix assembly and unsupervised models (issue #190, Workstream B).

Document-matrix assembly from features or embeddings, clustering, topic
modeling and nearest-neighbor fitting, their geometry/metrics, output row
shaping, and the matrix-artifact reader/writer. numpy and scikit-learn stay
behind lazy optional-dependency imports.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import polars as pl

from ..adapters import WarehouseAdapter
from ..config.model import MLConfig, ModelConfig
from ..config.project import ProjectConfig
from ..dag import parse_ref
from ..hashing import HASH_DIGEST_SIZE
from ..ml_contracts import ExecutableMLContract
from ..optional_dependencies import import_optional_dependency
from ..versioning import compute_code_version
from .artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    IncompatibleClassicMLArtifactError,
    _artifact_files_hash,
    _artifact_publication,
    _artifact_version,
    _hash_json,
    _new_artifact_staging_path,
    _read_artifact_json,
    _read_metadata,
    _remove_path,
    _runtime_versions,
    _validate_artifact_payload,
    _validate_metadata,
    _validated_persisted_options,
    _write_metadata,
)
from .common import ClassicMLRun, _canonical_row_key, _project_metrics

_ML_FEATURE = "Clustering and topic modeling"


def _numpy() -> Any:
    return import_optional_dependency("numpy", extra="ml", feature=_ML_FEATURE)


def _sklearn(module: str) -> Any:
    return import_optional_dependency(module, extra="ml", feature=_ML_FEATURE)


@dataclass
class _MatrixDoc:
    row_index: int
    row_id: str
    document_id: Any
    source_path: Any


def _run_matrix_model(
    *,
    model: ModelConfig,
    ml: MLConfig,
    contract: ExecutableMLContract,
    project: ProjectConfig,
    project_dir: Path,
    adapter: WarehouseAdapter,
) -> ClassicMLRun:
    if not model.depends_on:
        raise ValueError(f"ML model '{model.name}' must declare depends_on.")
    np = _numpy()
    provider = contract.provider
    task = contract.task
    artifact_path = contract.artifact_path
    is_predict = ml.mode in {"predict", "load_pretrained"}

    persisted = _read_matrix_artifact(artifact_path, provider, ml) if is_predict else None
    # predict/load_pretrained take the trained options (input, normalize, vocabulary)
    # from the persisted artifact — `ml.options` is empty in those modes.
    options = persisted["options"] if persisted is not None else contract.options
    fixed_features = persisted["feature_names"] if persisted is not None else None

    source_name = parse_ref(model.depends_on[0])
    source_df = adapter.query_df(f"SELECT * FROM {adapter.table_ref(source_name)}")
    docs, matrix, feature_names = _assemble_matrix(
        source_df, source_name, options, model.name, np, fixed_features
    )
    training_input = _matrix_training_input(model.depends_on, docs, matrix)
    code_version = compute_code_version(
        extraction=None, transform=None, ml=ml, project_dir=project_dir
    )

    raw_matrix = matrix
    if options.get("normalize") == "l2":
        matrix = _l2_normalize(matrix, np)

    if persisted is not None:
        fitted = _apply_matrix_model(task, provider, options, matrix, persisted, np)
    else:
        fitted = _fit_matrix_model(task, provider, options, matrix, np)

    all_metrics, primary_rows, secondary, empty_primary = _matrix_outputs(
        task=task,
        provider=provider,
        options=options,
        docs=docs,
        matrix=matrix,
        raw_matrix=raw_matrix,
        fitted=fitted,
        feature_names=feature_names,
        source_name=source_name,
        np=np,
    )

    if persisted is not None:
        metadata = _read_metadata(artifact_path)
        publication = None
    else:
        model_payload = _matrix_model_payload(
            provider, task, feature_names, fitted, options
        )
        metadata = _matrix_metadata(
            model=model,
            ml=ml,
            provider=provider,
            task=task,
            training_input=training_input,
            model_payload=model_payload,
            options=options,
            metrics=all_metrics,
            code_version=code_version,
        )
        staged_path = _new_artifact_staging_path(artifact_path)
        try:
            _write_matrix_artifact(staged_path, metadata, model_payload)
            publication = _artifact_publication(
                project=project,
                project_dir=project_dir,
                model=model,
                artifact_path=artifact_path,
                staged_path=staged_path,
                metadata=metadata,
            )
        except BaseException:
            _remove_path(staged_path)
            raise

    if ml.mode == "fit":
        primary_df = pl.DataFrame([_matrix_fit_summary(task, fitted, metadata)])
        secondary = {}
    else:
        primary_df = pl.DataFrame(primary_rows) if primary_rows else empty_primary

    return ClassicMLRun(
        df=primary_df,
        artifact_path=artifact_path,
        artifact_version=str(metadata["artifact_version"]),
        training_input=metadata.get("training_input", training_input),
        metrics=_project_metrics(ml, all_metrics),
        artifact_metadata=metadata,
        secondary_tables=secondary,
        _publication=publication,
    )


def _matrix_outputs(
    *,
    task: str,
    provider: str,
    options: dict[str, Any],
    docs: list[_MatrixDoc],
    matrix: Any,
    raw_matrix: Any,
    fitted: dict[str, Any],
    feature_names: list[str],
    source_name: str,
    np: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, pl.DataFrame], pl.DataFrame]:
    if task == "cluster":
        metrics = _cluster_metrics(matrix, fitted, np)
        primary_rows = _cluster_primary_rows(docs, fitted, provider, source_name)
        secondary = {
            "topics": _dataframe_or_empty(
                _cluster_term_rows(feature_names, raw_matrix, fitted, options, provider, np),
                _empty_topics_df(),
            ),
            "representative_docs": _dataframe_or_empty(
                _representative_docs_rows(docs, fitted, options, provider, source_name),
                _empty_representative_docs_df(),
            ),
        }
        if options.get("nearest_neighbors", 0):
            secondary["neighbors"] = _dataframe_or_empty(
                _neighbor_rows(docs, matrix, options, provider, source_name),
                _empty_neighbors_df(),
            )
        return metrics, primary_rows, secondary, _empty_cluster_df()

    metrics = _topic_metrics(matrix, fitted, np)
    primary_rows = _topic_document_rows(docs, fitted, provider, source_name)
    secondary = {
        "topics": _dataframe_or_empty(
            _topic_term_rows(feature_names, fitted, options, provider),
            _empty_topics_df(),
        )
    }
    return metrics, primary_rows, secondary, _empty_document_topics_df()


def _dataframe_or_empty(rows: list[dict[str, Any]], empty: pl.DataFrame) -> pl.DataFrame:
    return pl.DataFrame(rows) if rows else empty


def _assemble_matrix(
    source_df: pl.DataFrame,
    source_name: str,
    options: dict[str, Any],
    model_name: str,
    np: Any,
    fixed_features: list[str] | None = None,
) -> tuple[list[_MatrixDoc], Any, list[str]]:
    if options.get("input") == "embedding":
        return _assemble_embedding_matrix(
            source_df, source_name, options, model_name, np, fixed_features
        )
    return _assemble_feature_matrix(
        source_df, source_name, options, model_name, np, fixed_features
    )


def _assemble_feature_matrix(
    source_df: pl.DataFrame,
    source_name: str,
    options: dict[str, Any],
    model_name: str,
    np: Any,
    fixed_features: list[str] | None,
) -> tuple[list[_MatrixDoc], Any, list[str]]:
    term_field = options["term_field"]
    value_field = options["value_field"]
    required = {"row_id", "row_index", term_field, value_field}
    missing = sorted(required - set(source_df.columns))
    if missing:
        raise ValueError(
            f"ML model '{model_name}' input '{source_name}' is missing feature "
            f"columns {missing}; expected a `features` model output."
        )
    has_term_index = "term_index" in source_df.columns
    docs_by_id: dict[str, _MatrixDoc] = {}
    doc_terms: dict[str, dict[str, float]] = {}
    term_order: dict[str, int] = {}
    for row in source_df.iter_rows(named=True):
        row_id = str(row["row_id"])
        if row_id not in docs_by_id:
            docs_by_id[row_id] = _MatrixDoc(
                row_index=int(row["row_index"]),
                row_id=row_id,
                document_id=row.get("document_id"),
                source_path=row.get("source_path"),
            )
            doc_terms[row_id] = {}
        term = str(row[term_field])
        value = row[value_field]
        doc_terms[row_id][term] = 0.0 if value is None else float(value)
        if term not in term_order:
            term_order[term] = int(row["term_index"]) if has_term_index else len(term_order)
    docs = sorted(docs_by_id.values(), key=lambda d: (d.row_index, d.row_id))
    # predict/load_pretrained align new documents onto the trained vocabulary:
    # unseen terms are dropped, missing terms stay zero.
    feature_names = (
        list(fixed_features)
        if fixed_features is not None
        else sorted(term_order, key=lambda t: (term_order[t], t))
    )
    col_of = {term: i for i, term in enumerate(feature_names)}
    matrix = np.zeros((len(docs), len(feature_names)), dtype=float)
    for i, doc in enumerate(docs):
        for term, value in doc_terms[doc.row_id].items():
            column = col_of.get(term)
            if column is not None:
                matrix[i, column] = value
    return docs, matrix, feature_names


def _assemble_embedding_matrix(
    source_df: pl.DataFrame,
    source_name: str,
    options: dict[str, Any],
    model_name: str,
    np: Any,
    fixed_features: list[str] | None,
) -> tuple[list[_MatrixDoc], Any, list[str]]:
    field = options["embedding_field"]
    if field not in source_df.columns:
        raise ValueError(
            f"ML model '{model_name}' embedding_field '{field}' is not present in "
            f"'{source_name}'."
        )
    expected_dim = len(fixed_features) if fixed_features is not None else None
    ordered = sorted(source_df.iter_rows(named=True), key=_canonical_row_key)
    docs: list[_MatrixDoc] = []
    vectors: list[list[float]] = []
    dim: int | None = expected_dim
    for index, row in enumerate(ordered):
        vector = row[field]
        if vector is None:
            raise ValueError(
                f"ML model '{model_name}' embedding_field '{field}' has a null vector "
                f"at row {index}."
            )
        values = [float(v) for v in vector]
        if dim is None:
            dim = len(values)
            if dim == 0:
                raise ValueError(
                    f"ML model '{model_name}' embedding_field '{field}' is empty."
                )
        elif len(values) != dim:
            raise ValueError(
                f"ML model '{model_name}' embedding_field '{field}' has inconsistent "
                f"vector lengths ({len(values)} vs {dim})."
            )
        row_id = str(row.get("document_id") or row.get("id") or index)
        docs.append(
            _MatrixDoc(
                row_index=index,
                row_id=row_id,
                document_id=row.get("document_id"),
                source_path=row.get("source_path"),
            )
        )
        vectors.append(values)
    feature_names = (
        list(fixed_features)
        if fixed_features is not None
        else [f"dim_{i}" for i in range(dim or 0)]
    )
    matrix = np.array(vectors, dtype=float) if vectors else np.zeros((0, dim or 0))
    return docs, matrix, feature_names


def _matrix_training_input(
    depends_on: list[str],
    docs: list[_MatrixDoc],
    matrix: Any,
) -> dict[str, Any]:
    content = [
        {"row_id": doc.row_id, "vector": [round(float(v), 6) for v in matrix[i]]}
        for i, doc in enumerate(docs)
    ]
    raw = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return {
        "refs": [parse_ref(ref) for ref in depends_on],
        "row_count": len(docs),
        "content_hash": hashlib.blake2b(
            raw.encode(), digest_size=HASH_DIGEST_SIZE
        ).hexdigest(),
    }


def _read_matrix_artifact(
    path: Path,
    provider: str,
    ml: MLConfig,
) -> dict[str, Any]:
    metadata = _read_metadata(path)
    _validate_metadata(
        metadata, path, provider, ml, expected_files=("metadata.json", "model.json")
    )
    _validate_artifact_payload(metadata, path, {})
    payload = _read_artifact_json(path / "model.json", path, "matrix model")
    feature_names = payload.get("feature_names")
    if not isinstance(feature_names, list) or any(
        not isinstance(name, str) for name in feature_names
    ):
        raise IncompatibleClassicMLArtifactError(
            f"incompatible matrix artifact at {path}: feature_names must be strings"
        )
    if payload.get("provider") != provider:
        raise IncompatibleClassicMLArtifactError(
            f"incompatible matrix artifact at {path}: expected provider {provider}, "
            f"found {payload.get('provider')!r}"
        )
    payload["options"] = _validated_persisted_options(
        cast(Any, provider), payload.get("options"), path, surface="matrix payload"
    )
    return payload


def _apply_matrix_model(
    task: str,
    provider: str,
    options: dict[str, Any],
    matrix: Any,
    payload: dict[str, Any],
    np: Any,
) -> dict[str, Any]:
    n_samples = int(matrix.shape[0])
    n_features = int(matrix.shape[1]) if matrix.ndim == 2 else 0
    fitted: dict[str, Any] = {"n_features": n_features, "n_samples": n_samples}
    if task == "cluster":
        if n_samples == 0:
            fitted.update({"labels": [], "centroids": {}, "distances": []})
            return fitted
        centers = payload["cluster_centers"]
        labels_sorted = sorted(int(label) for label in centers)
        center_arr = np.array([centers[str(label)] for label in labels_sorted])
        distances_to_centers = np.linalg.norm(
            matrix[:, None, :] - center_arr[None, :, :], axis=2
        )
        nearest = distances_to_centers.argmin(axis=1)
        fitted["labels"] = [int(labels_sorted[i]) for i in nearest]
        fitted["distances"] = [
            float(distances_to_centers[i, nearest[i]]) for i in range(n_samples)
        ]
        fitted["centroids"] = {
            str(label): center_arr[i].tolist()
            for i, label in enumerate(labels_sorted)
        }
        return fitted

    components = np.array(payload["components"])
    if n_samples == 0:
        fitted.update(
            {
                "components": components.tolist(),
                "doc_topics": [],
                "n_components": int(components.shape[0]),
            }
        )
        return fitted
    decomposition = _sklearn("sklearn.decomposition")
    weights, _, _ = decomposition.non_negative_factorization(
        matrix,
        H=components,
        n_components=int(components.shape[0]),
        update_H=False,
        init="custom",
        max_iter=options["max_iter"],
        random_state=options["random_state"],
    )
    fitted["components"] = components.tolist()
    fitted["doc_topics"] = _normalize_rows(weights, np).tolist()
    fitted["n_components"] = int(components.shape[0])
    return fitted


def _fit_matrix_model(
    task: str,
    provider: str,
    options: dict[str, Any],
    matrix: Any,
    np: Any,
) -> dict[str, Any]:
    n_samples = int(matrix.shape[0])
    n_features = int(matrix.shape[1]) if matrix.ndim == 2 else 0
    fitted: dict[str, Any] = {"n_features": n_features, "n_samples": n_samples}
    if n_samples == 0:
        if task == "cluster":
            fitted.update({"labels": [], "centroids": {}, "distances": []})
        else:
            fitted.update({"components": [], "doc_topics": [], "n_components": 0})
        return fitted

    if task == "cluster":
        labels, extra = _fit_cluster(provider, options, matrix, np, n_samples)
        fitted.update(extra)
        fitted["labels"] = [int(label) for label in labels]
        centroids, distances = _cluster_geometry(matrix, fitted["labels"], np)
        fitted["centroids"] = {str(k): v.tolist() for k, v in centroids.items()}
        fitted["distances"] = distances
        return fitted

    components, doc_topics, extra = _fit_topics(
        provider, options, matrix, np, n_samples, n_features
    )
    fitted.update(extra)
    fitted["components"] = components.tolist()
    fitted["doc_topics"] = doc_topics.tolist()
    fitted["n_components"] = int(components.shape[0])
    return fitted


def _fit_cluster(
    provider: str,
    options: dict[str, Any],
    matrix: Any,
    np: Any,
    n_samples: int,
) -> tuple[Any, dict[str, Any]]:
    cluster = _sklearn("sklearn.cluster")
    if provider == "builtin.kmeans":
        k = options["n_clusters"]
        if k > n_samples:
            raise ValueError(
                f"n_clusters ({k}) cannot exceed the document count ({n_samples})."
            )
        estimator = cluster.KMeans(
            n_clusters=k,
            max_iter=options["max_iter"],
            n_init=options["n_init"],
            random_state=options["random_state"],
        )
        labels = estimator.fit_predict(matrix)
        return labels, {"inertia": float(estimator.inertia_)}
    if provider == "builtin.dbscan":
        estimator = cluster.DBSCAN(
            eps=options["eps"],
            min_samples=options["min_samples"],
            metric=options["metric"],
        )
        return estimator.fit_predict(matrix), {}
    if provider == "builtin.hdbscan":
        estimator = cluster.HDBSCAN(
            min_cluster_size=options["min_cluster_size"],
            min_samples=options["min_samples"],
            metric=options["metric"],
            copy=True,
        )
        labels = estimator.fit_predict(matrix)
        return labels, {"probabilities": [float(p) for p in estimator.probabilities_]}
    raise ValueError(f"Unsupported cluster provider '{provider}'.")


def _fit_topics(
    provider: str,
    options: dict[str, Any],
    matrix: Any,
    np: Any,
    n_samples: int,
    n_features: int,
) -> tuple[Any, Any, dict[str, Any]]:
    k = options["n_topics"]
    if k > n_samples:
        raise ValueError(
            f"n_topics ({k}) cannot exceed the document count ({n_samples})."
        )
    if k > n_features:
        raise ValueError(
            f"n_topics ({k}) cannot exceed the feature count ({n_features})."
        )
    if float(matrix.min()) < 0.0:
        raise ValueError(
            "topic modeling requires a non-negative feature matrix (counts or TF-IDF)."
        )
    decomposition = _sklearn("sklearn.decomposition")
    if provider == "builtin.nmf":
        estimator = decomposition.NMF(
            n_components=k,
            max_iter=options["max_iter"],
            random_state=options["random_state"],
            init="nndsvda",
        )
        weights = estimator.fit_transform(matrix)
        return (
            estimator.components_,
            _normalize_rows(weights, np),
            {"reconstruction_error": float(estimator.reconstruction_err_)},
        )
    if provider == "builtin.lda":
        estimator = decomposition.LatentDirichletAllocation(
            n_components=k,
            max_iter=options["max_iter"],
            learning_method=options["learning_method"],
            random_state=options["random_state"],
        )
        weights = estimator.fit_transform(matrix)
        return (
            estimator.components_,
            _normalize_rows(weights, np),
            {"perplexity": float(estimator.perplexity(matrix))},
        )
    raise ValueError(f"Unsupported topic provider '{provider}'.")


def _normalize_rows(weights: Any, np: Any) -> Any:
    totals = weights.sum(axis=1, keepdims=True)
    return np.divide(weights, totals, out=np.zeros_like(weights), where=totals > 0)


def _l2_normalize(matrix: Any, np: Any) -> Any:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def _cluster_geometry(
    matrix: Any,
    labels: list[int],
    np: Any,
) -> tuple[dict[int, Any], list[float | None]]:
    centroids: dict[int, Any] = {}
    for label in sorted({label for label in labels if label != -1}):
        members = matrix[np.array(labels) == label]
        centroids[label] = members.mean(axis=0)
    distances: list[float | None] = []
    for i, label in enumerate(labels):
        if label == -1:
            distances.append(None)
        else:
            distances.append(float(np.linalg.norm(matrix[i] - centroids[label])))
    return centroids, distances


def _cluster_primary_rows(
    docs: list[_MatrixDoc],
    fitted: dict[str, Any],
    provider: str,
    source_name: str,
) -> list[dict[str, Any]]:
    labels = fitted["labels"]
    distances = fitted["distances"]
    probabilities = fitted.get("probabilities")
    rows: list[dict[str, Any]] = []
    for i, doc in enumerate(docs):
        label = int(labels[i])
        row: dict[str, Any] = {
            "source_model": source_name,
            "row_index": doc.row_index,
            "row_id": doc.row_id,
            "provider": provider,
            "cluster": label,
            "distance": distances[i],
            "probability": (
                float(probabilities[i]) if probabilities is not None else None
            ),
            "is_noise": label == -1,
        }
        _attach_doc_identity(row, doc)
        rows.append(row)
    return rows


def _representative_docs_rows(
    docs: list[_MatrixDoc],
    fitted: dict[str, Any],
    options: dict[str, Any],
    provider: str,
    source_name: str,
) -> list[dict[str, Any]]:
    top_n = options["representative_docs"]
    if top_n == 0:
        return []
    labels = fitted["labels"]
    distances = fitted["distances"]
    by_cluster: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        if label != -1:
            by_cluster.setdefault(int(label), []).append(i)
    rows: list[dict[str, Any]] = []
    for cluster in sorted(by_cluster):
        members = sorted(
            by_cluster[cluster],
            key=lambda i: (distances[i] if distances[i] is not None else 0.0, docs[i].row_id),
        )
        for rank, i in enumerate(members[:top_n]):
            doc = docs[i]
            row: dict[str, Any] = {
                "source_model": source_name,
                "provider": provider,
                "cluster": cluster,
                "rank": rank,
                "row_id": doc.row_id,
                "distance": distances[i],
            }
            _attach_doc_identity(row, doc)
            rows.append(row)
    return rows


def _topic_document_rows(
    docs: list[_MatrixDoc],
    fitted: dict[str, Any],
    provider: str,
    source_name: str,
) -> list[dict[str, Any]]:
    doc_topics = fitted["doc_topics"]
    rows: list[dict[str, Any]] = []
    for i, doc in enumerate(docs):
        weights = doc_topics[i]
        for topic, weight in enumerate(weights):
            row: dict[str, Any] = {
                "source_model": source_name,
                "row_index": doc.row_index,
                "row_id": doc.row_id,
                "provider": provider,
                "topic": topic,
                "weight": float(weight),
            }
            _attach_doc_identity(row, doc)
            rows.append(row)
    return rows


def _topic_term_rows(
    feature_names: list[str],
    fitted: dict[str, Any],
    options: dict[str, Any],
    provider: str,
) -> list[dict[str, Any]]:
    top_terms = options["top_terms"]
    rows: list[dict[str, Any]] = []
    for topic, weights in enumerate(fitted["components"]):
        ranked = sorted(
            range(len(weights)),
            key=lambda j: (-weights[j], feature_names[j]),
        )
        for rank, j in enumerate(ranked[:top_terms]):
            rows.append(
                {
                    "provider": provider,
                    "topic": topic,
                    "rank": rank,
                    "term": feature_names[j],
                    "weight": float(weights[j]),
                }
            )
    return rows


def _cluster_term_rows(
    feature_names: list[str],
    raw_matrix: Any,
    fitted: dict[str, Any],
    options: dict[str, Any],
    provider: str,
    np: Any,
) -> list[dict[str, Any]]:
    """Label clusters with their most distinctive terms via c-TF-IDF: treat each
    cluster as one document and rank terms by term frequency within the cluster
    weighted by inverse cluster frequency (BERTopic's cluster-labeling trick)."""
    top_terms = options.get("top_terms", 0)
    if top_terms == 0 or not feature_names or options.get("input") == "embedding":
        return []
    labels = fitted["labels"]
    clusters = sorted({label for label in labels if label != -1})
    if not clusters:
        return []
    label_arr = np.array(labels)
    class_freq = np.array(
        [raw_matrix[label_arr == cluster].sum(axis=0) for cluster in clusters]
    )
    class_totals = class_freq.sum(axis=1, keepdims=True)
    tf = class_freq / np.clip(class_totals, 1e-12, None)
    term_freq = class_freq.sum(axis=0)
    average_terms = float(class_freq.sum(axis=1).mean())
    idf = np.log(1 + average_terms / np.clip(term_freq, 1e-12, None))
    ctfidf = tf * idf
    rows: list[dict[str, Any]] = []
    for row_index, cluster in enumerate(clusters):
        weights = ctfidf[row_index]
        ranked = sorted(
            range(len(weights)), key=lambda j: (-weights[j], feature_names[j])
        )
        for rank, j in enumerate(ranked[:top_terms]):
            rows.append(
                {
                    "provider": provider,
                    "topic": cluster,
                    "rank": rank,
                    "term": feature_names[j],
                    "weight": float(weights[j]),
                }
            )
    return rows


def _neighbor_rows(
    docs: list[_MatrixDoc],
    matrix: Any,
    options: dict[str, Any],
    provider: str,
    source_name: str,
) -> list[dict[str, Any]]:
    k = options.get("nearest_neighbors", 0)
    if k == 0 or len(docs) < 2:
        return []
    np = _numpy()
    distances = np.linalg.norm(matrix[:, None, :] - matrix[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    rows: list[dict[str, Any]] = []
    for i, doc in enumerate(docs):
        order = list(np.argsort(distances[i])[:k])
        for rank, j in enumerate(order):
            if not np.isfinite(distances[i][j]):
                continue
            row: dict[str, Any] = {
                "source_model": source_name,
                "provider": provider,
                "row_id": doc.row_id,
                "neighbor_row_id": docs[j].row_id,
                "rank": rank,
                "distance": float(distances[i][j]),
            }
            _attach_doc_identity(row, doc)
            rows.append(row)
    return rows


def _attach_doc_identity(row: dict[str, Any], doc: _MatrixDoc) -> None:
    if doc.document_id is not None:
        row["document_id"] = doc.document_id
    if doc.source_path is not None:
        row["source_path"] = doc.source_path


def _cluster_metrics(matrix: Any, fitted: dict[str, Any], np: Any) -> dict[str, Any]:
    labels = fitted["labels"]
    non_noise = sorted({label for label in labels if label != -1})
    metrics: dict[str, Any] = {
        "row_count": len(labels),
        "n_clusters": len(non_noise),
        "noise_points": sum(1 for label in labels if label == -1),
    }
    if "inertia" in fitted:
        metrics["inertia"] = fitted["inertia"]
    metrics["silhouette"] = _silhouette(matrix, labels, np)
    return metrics


def _silhouette(matrix: Any, labels: list[int], np: Any) -> float | None:
    mask = np.array([label != -1 for label in labels])
    kept_labels = [label for label in labels if label != -1]
    if len(set(kept_labels)) < 2 or len(kept_labels) <= len(set(kept_labels)):
        return None
    sklearn_metrics = _sklearn("sklearn.metrics")
    return float(
        sklearn_metrics.silhouette_score(matrix[mask], np.array(kept_labels))
    )


def _topic_metrics(matrix: Any, fitted: dict[str, Any], np: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "row_count": fitted["n_samples"],
        "n_topics": fitted["n_components"],
    }
    if "reconstruction_error" in fitted:
        metrics["reconstruction_error"] = fitted["reconstruction_error"]
    if "perplexity" in fitted:
        metrics["perplexity"] = fitted["perplexity"]
    metrics["topic_coherence"] = _topic_coherence(matrix, fitted, np)
    return metrics


def _topic_coherence(matrix: Any, fitted: dict[str, Any], np: Any) -> float | None:
    components = np.array(fitted["components"])
    if components.size == 0:
        return None
    occurrence = matrix > 0
    doc_freq = occurrence.sum(axis=0)
    top = min(10, components.shape[1])
    coherences: list[float] = []
    for weights in components:
        top_terms = list(np.argsort(weights)[::-1][:top])
        pair_scores: list[float] = []
        for a in range(1, len(top_terms)):
            for b in range(a):
                wi, wj = top_terms[a], top_terms[b]
                co = int(np.logical_and(occurrence[:, wi], occurrence[:, wj]).sum())
                pair_scores.append(math.log((co + 1) / (int(doc_freq[wj]) + 1)))
        if pair_scores:
            coherences.append(sum(pair_scores) / len(pair_scores))
    return float(sum(coherences) / len(coherences)) if coherences else None


def _matrix_model_payload(
    provider: str,
    task: str,
    feature_names: list[str],
    fitted: dict[str, Any],
    options: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "provider": provider,
        "task": task,
        "input": options.get("input", "features"),
        "feature_names": feature_names,
        "n_features": fitted["n_features"],
        "options": dict(options),
    }
    if task == "cluster":
        payload["cluster_centers"] = fitted.get("centroids", {})
    else:
        payload["components"] = fitted.get("components", [])
        payload["n_components"] = fitted.get("n_components", 0)
    return payload


def _matrix_metadata(
    *,
    model: ModelConfig,
    ml: MLConfig,
    provider: str,
    task: str,
    training_input: dict[str, Any],
    model_payload: dict[str, Any],
    options: dict[str, Any],
    metrics: dict[str, Any],
    code_version: str,
) -> dict[str, Any]:
    integrity = {"feature_count": model_payload["n_features"]}
    if task == "cluster":
        integrity["cluster_count"] = metrics.get("n_clusters", 0)
    else:
        integrity["topic_count"] = model_payload.get("n_components", 0)
    metadata: dict[str, Any] = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "classic_ml",
        "model_name": model.name,
        "task": task,
        "provider": provider,
        "mode": ml.mode,
        "text_field": ml.text_field,
        "code_version": code_version,
        "config_hash": _hash_json(
            {"task": task, "provider": provider, "options": options}
        ),
        "runtime": _runtime_versions(provider),
        "training_input": training_input,
        "integrity": integrity,
        "files": ["metadata.json", "model.json"],
        "options": options,
        "model_hash": _hash_json(model_payload),
    }
    if ml.artifact.include_metrics:
        metadata["metrics"] = _project_metrics(ml, metrics)
    return metadata


def _write_matrix_artifact(
    path: Path,
    metadata: dict[str, Any],
    model_payload: dict[str, Any],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.json").write_text(json.dumps(model_payload, indent=2, sort_keys=True))
    payload_files = ["model.json"]
    metadata["files"] = ["metadata.json", *payload_files]
    metadata["artifact_files_hash"] = _artifact_files_hash(path, payload_files, model_payload)
    metadata["artifact_version"] = _artifact_version(metadata)
    _write_metadata(path, metadata)


def _matrix_fit_summary(
    task: str,
    fitted: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "artifact_version": metadata["artifact_version"],
        "row_count": fitted["n_samples"],
        "feature_count": fitted["n_features"],
    }
    if task == "cluster":
        summary["cluster_count"] = len(
            {label for label in fitted["labels"] if label != -1}
        )
    else:
        summary["topic_count"] = fitted["n_components"]
    return summary


def _empty_cluster_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "source_model": pl.String,
            "row_index": pl.Int64,
            "row_id": pl.String,
            "provider": pl.String,
            "cluster": pl.Int64,
            "distance": pl.Float64,
            "probability": pl.Float64,
            "is_noise": pl.Boolean,
        }
    )


def _empty_representative_docs_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "source_model": pl.String,
            "provider": pl.String,
            "cluster": pl.Int64,
            "rank": pl.Int64,
            "row_id": pl.String,
            "distance": pl.Float64,
        }
    )


def _empty_document_topics_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "source_model": pl.String,
            "row_index": pl.Int64,
            "row_id": pl.String,
            "provider": pl.String,
            "topic": pl.Int64,
            "weight": pl.Float64,
        }
    )


def _empty_topics_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "provider": pl.String,
            "topic": pl.Int64,
            "rank": pl.Int64,
            "term": pl.String,
            "weight": pl.Float64,
        }
    )


def _empty_neighbors_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "source_model": pl.String,
            "provider": pl.String,
            "row_id": pl.String,
            "neighbor_row_id": pl.String,
            "rank": pl.Int64,
            "distance": pl.Float64,
        }
    )
