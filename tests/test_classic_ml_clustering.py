"""Clustering and topic-modeling matrix tasks (issue #42).

These tasks consume a document-feature matrix (a `features` model pivoted to
documents x terms, or a dense embedding column) and emit a primary
per-document table plus companion tables. Fitting is deterministic under a
fixed `random_state`, and results are invariant to warehouse row order.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import polars as pl
import pytest

pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from stel.adapters import WarehouseAdapter
from stel.classic_ml import run_classic_ml_model
from stel.config.model import ModelConfig
from stel.config.project import ProjectConfig
from stel.ml_contracts import MLContractError, validate_ml_contract

# Three clearly separable themes; each doc is a bag of term counts.
_THEMES: list[tuple[str, dict[str, int]]] = [
    ("a1", {"goal": 3, "team": 2}),
    ("a2", {"goal": 2, "team": 3}),
    ("a3", {"goal": 3, "team": 3}),
    ("b1", {"stock": 3, "market": 2}),
    ("b2", {"stock": 2, "market": 3}),
    ("b3", {"stock": 3, "market": 3}),
    ("c1", {"recipe": 3, "flour": 2}),
    ("c2", {"recipe": 2, "flour": 3}),
    ("c3", {"recipe": 3, "flour": 3}),
]


def _features_df(docs: list[tuple[str, dict[str, int]]]) -> pl.DataFrame:
    term_index: dict[str, int] = {}
    for _, terms in docs:
        for term in sorted(terms):
            term_index.setdefault(term, len(term_index))
    rows: list[dict[str, Any]] = []
    for idx, (row_id, terms) in enumerate(docs):
        for term, value in terms.items():
            rows.append(
                {
                    "source_model": "doc_tfidf",
                    "row_index": idx,
                    "row_id": row_id,
                    "provider": "builtin.tfidf",
                    "term": term,
                    "term_index": term_index[term],
                    "value": float(value),
                    "document_id": row_id,
                }
            )
    return pl.DataFrame(rows)


def _model(ml: dict[str, Any]) -> ModelConfig:
    return ModelConfig(name="derived", depends_on=["ref('doc_tfidf')"], ml=ml)


def _run(tmp_path: Path, ml: dict[str, Any], source_df: pl.DataFrame) -> Any:
    adapter = Mock(spec=WarehouseAdapter)
    adapter.table_ref.return_value = '"p"."doc_tfidf"'
    adapter.query_df.return_value = source_df
    return run_classic_ml_model(
        model=_model(ml),
        project=ProjectConfig(name="p"),
        project_dir=tmp_path,
        adapter=adapter,
    )


def _cluster_ml(**options: Any) -> dict[str, Any]:
    return {
        "task": "cluster",
        "mode": "fit_transform",
        "provider": options.pop("provider", "builtin.kmeans"),
        "options": options,
    }


def _topic_ml(**options: Any) -> dict[str, Any]:
    return {
        "task": "topic_model",
        "mode": "fit_transform",
        "provider": options.pop("provider", "builtin.nmf"),
        "options": options,
    }


def _km3(**extra: Any) -> dict[str, Any]:
    return {**_cluster_ml(n_clusters=3, normalize="l2"), **extra}


def _partition(rows: list[dict[str, Any]], key: str = "cluster") -> set[frozenset[str]]:
    groups: dict[int, set[str]] = {}
    for row in rows:
        groups.setdefault(row[key], set()).add(row["row_id"])
    return {frozenset(members) for members in groups.values()}


# ─── clustering ──────────────────────────────────────────────────────────────


def test_kmeans_separates_themes_and_emits_representative_docs(tmp_path: Path) -> None:
    output = _run(
        tmp_path,
        _cluster_ml(n_clusters=3, normalize="l2", representative_docs=2),
        _features_df(_THEMES),
    )

    primary = output.df.to_dicts()
    assert len(primary) == len(_THEMES)
    assert {r["row_id"] for r in primary} == {rid for rid, _ in _THEMES}
    assert not any(r["is_noise"] for r in primary)
    # Each theme lands in its own cluster.
    assert _partition(primary) == {
        frozenset({"a1", "a2", "a3"}),
        frozenset({"b1", "b2", "b3"}),
        frozenset({"c1", "c2", "c3"}),
    }

    reps = output.secondary_tables["representative_docs"].to_dicts()
    assert {r["cluster"] for r in reps} == {0, 1, 2}
    assert all(r["rank"] in (0, 1) for r in reps)
    assert len(reps) == 6  # 3 clusters x 2 representatives


def test_kmeans_is_deterministic(tmp_path: Path) -> None:
    first = _run(tmp_path / "a", _km3(), _features_df(_THEMES))
    second = _run(tmp_path / "b", _km3(), _features_df(_THEMES))
    assert first.df.to_dicts() == second.df.to_dicts()
    assert first.artifact_version == second.artifact_version


def test_kmeans_is_invariant_to_row_order(tmp_path: Path) -> None:
    shuffled = [_THEMES[i] for i in (5, 0, 8, 3, 1, 7, 2, 6, 4)]
    ordered = _run(tmp_path / "o", _km3(), _features_df(_THEMES))
    scrambled = _run(tmp_path / "s", _km3(), _features_df(shuffled))
    assert _partition(ordered.df.to_dicts()) == _partition(scrambled.df.to_dicts())


@pytest.mark.parametrize(
    ("provider", "options"),
    [
        ("builtin.dbscan", {"eps": 0.5, "min_samples": 2}),
        ("builtin.hdbscan", {"min_cluster_size": 2}),
    ],
)
def test_density_clustering_separates_themes(
    tmp_path: Path, provider: str, options: dict[str, Any]
) -> None:
    output = _run(
        tmp_path,
        _cluster_ml(provider=provider, normalize="l2", **options),
        _features_df(_THEMES),
    )
    non_noise = [r for r in output.df.to_dicts() if not r["is_noise"]]
    assert _partition(non_noise) == {
        frozenset({"a1", "a2", "a3"}),
        frozenset({"b1", "b2", "b3"}),
        frozenset({"c1", "c2", "c3"}),
    }


def test_cluster_metrics_reported(tmp_path: Path) -> None:
    output = _run(
        tmp_path,
        _km3(metrics=["n_clusters", "inertia", "silhouette"]),
        _features_df(_THEMES),
    )
    assert output.metrics["n_clusters"] == 3
    assert output.metrics["inertia"] >= 0
    assert -1.0 <= output.metrics["silhouette"] <= 1.0


# ─── topic modeling ──────────────────────────────────────────────────────────


def test_nmf_emits_document_topics_and_topic_terms(tmp_path: Path) -> None:
    output = _run(
        tmp_path,
        _topic_ml(n_topics=3, top_terms=2, representative_docs=0),
        _features_df(_THEMES),
    )
    doc_topics = output.df.to_dicts()
    assert len(doc_topics) == len(_THEMES) * 3  # one row per doc x topic
    # Per-document topic weights are normalized proportions.
    by_doc: dict[str, float] = {}
    for row in doc_topics:
        by_doc[row["row_id"]] = by_doc.get(row["row_id"], 0.0) + row["weight"]
    assert all(abs(total - 1.0) < 1e-6 for total in by_doc.values())

    topics = output.secondary_tables["topics"].to_dicts()
    assert {t["topic"] for t in topics} == {0, 1, 2}
    # Each topic's top terms come from a single theme's vocabulary.
    theme_terms = [{"goal", "team"}, {"stock", "market"}, {"recipe", "flour"}]
    for topic in {t["topic"] for t in topics}:
        terms = {t["term"] for t in topics if t["topic"] == topic}
        assert any(terms <= theme for theme in theme_terms)


def test_lda_topic_model_runs(tmp_path: Path) -> None:
    ml = _topic_ml(provider="builtin.lda", n_topics=3, max_iter=20)
    ml["metrics"] = ["perplexity", "n_topics"]
    output = _run(tmp_path, ml, _features_df(_THEMES))
    assert output.metrics["n_topics"] == 3
    assert output.metrics["perplexity"] > 0


# ─── embedding input ─────────────────────────────────────────────────────────


def test_embedding_input_clusters_dense_vectors(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "document_id": ["a", "b", "c", "d"],
            "embedding": [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [0.05, 0.95]],
        }
    )
    ml = _cluster_ml(
        provider="builtin.kmeans",
        n_clusters=2,
        input="embedding",
        embedding_field="embedding",
    )
    output = _run(tmp_path, ml, df)
    assert _partition(output.df.to_dicts()) == {frozenset({"a", "b"}), frozenset({"c", "d"})}


# ─── contract validation ─────────────────────────────────────────────────────


def test_default_matrix_providers_resolve(tmp_path: Path) -> None:
    cluster = validate_ml_contract(_model({"task": "cluster"}), ProjectConfig(name="p"), tmp_path)
    topic = validate_ml_contract(_model({"task": "topic_model"}), ProjectConfig(name="p"), tmp_path)
    assert cluster.provider == "builtin.kmeans"
    assert topic.provider == "builtin.nmf"


def test_topic_model_rejects_embedding_input(tmp_path: Path) -> None:
    with pytest.raises(MLContractError, match="input='features'"):
        validate_ml_contract(
            _model(_topic_ml(input="embedding", embedding_field="vec")),
            ProjectConfig(name="p"),
            tmp_path,
        )


@pytest.mark.parametrize(
    ("task", "provider"),
    [
        ("cluster", "builtin.dbscan"),
        ("cluster", "builtin.hdbscan"),
        ("topic_model", "builtin.lda"),
    ],
)
def test_non_parametric_providers_reject_prediction(
    tmp_path: Path, task: str, provider: str
) -> None:
    with pytest.raises(MLContractError, match="cannot assign new documents"):
        validate_ml_contract(
            _model({"task": task, "mode": "predict", "provider": provider}),
            ProjectConfig(name="p"),
            tmp_path,
        )


def test_cluster_rejects_label_field(tmp_path: Path) -> None:
    with pytest.raises(MLContractError, match="does not use"):
        validate_ml_contract(
            _model({"task": "cluster", "label_field": "y"}),
            ProjectConfig(name="p"),
            tmp_path,
        )


# ─── cluster labeling (c-TF-IDF) and nearest neighbors ───────────────────────


def test_cluster_emits_ctfidf_topic_terms(tmp_path: Path) -> None:
    ml = _cluster_ml(n_clusters=3, normalize="l2", top_terms=2)
    output = _run(tmp_path, ml, _features_df(_THEMES))
    topics = output.secondary_tables["topics"].to_dicts()
    assert {t["topic"] for t in topics} == {0, 1, 2}
    theme_terms = [{"goal", "team"}, {"stock", "market"}, {"recipe", "flour"}]
    for cluster in {t["topic"] for t in topics}:
        terms = {t["term"] for t in topics if t["topic"] == cluster}
        assert any(terms <= theme for theme in theme_terms)


def test_cluster_emits_nearest_neighbors(tmp_path: Path) -> None:
    ml = _cluster_ml(n_clusters=3, normalize="l2", nearest_neighbors=2)
    output = _run(tmp_path, ml, _features_df(_THEMES))
    neighbors = output.secondary_tables["neighbors"].to_dicts()
    by_doc: dict[str, list[tuple[int, str]]] = {}
    for row in neighbors:
        by_doc.setdefault(row["row_id"], []).append((row["rank"], row["neighbor_row_id"]))
    assert all(len(v) <= 2 for v in by_doc.values())
    # a document's nearest neighbor shares its theme prefix (a/b/c).
    for row_id, entries in by_doc.items():
        nearest = min(entries)[1]
        assert nearest[0] == row_id[0]


# ─── prediction / artifact reload ────────────────────────────────────────────


def test_kmeans_predict_reuses_persisted_artifact(tmp_path: Path) -> None:
    trained = _run(tmp_path, _km3(), _features_df(_THEMES))
    trained.publish_artifact()

    predicted = _run(
        tmp_path,
        {"task": "cluster", "mode": "load_pretrained", "provider": "builtin.kmeans"},
        _features_df(_THEMES),
    )
    assert predicted.artifact_version == trained.artifact_version
    assert _partition(predicted.df.to_dicts()) == _partition(trained.df.to_dicts())


def test_nmf_predict_reuses_persisted_components(tmp_path: Path) -> None:
    trained = _run(tmp_path, _topic_ml(n_topics=3), _features_df(_THEMES))
    trained.publish_artifact()

    predicted = _run(
        tmp_path,
        {"task": "topic_model", "mode": "load_pretrained", "provider": "builtin.nmf"},
        _features_df(_THEMES),
    )
    assert predicted.artifact_version == trained.artifact_version
    assert predicted.df.height == len(_THEMES) * 3

    def dominant(rows: list[dict[str, Any]]) -> dict[str, int]:
        best: dict[str, tuple[float, int]] = {}
        for row in rows:
            key = row["row_id"]
            if key not in best or row["weight"] > best[key][0]:
                best[key] = (row["weight"], row["topic"])
        return {k: v[1] for k, v in best.items()}

    assert dominant(predicted.df.to_dicts()) == dominant(trained.df.to_dicts())


def test_predict_before_fit_reports_missing_artifact(tmp_path: Path) -> None:
    from stel.classic_ml import MissingClassicMLArtifactError

    with pytest.raises(MissingClassicMLArtifactError):
        _run(
            tmp_path,
            {"task": "cluster", "mode": "load_pretrained", "provider": "builtin.kmeans"},
            _features_df(_THEMES),
        )


# ─── empty corpus ────────────────────────────────────────────────────────────


def test_empty_corpus_yields_empty_tables(tmp_path: Path) -> None:
    # A features model that matched zero documents emits its schema with no rows.
    empty = pl.DataFrame(
        schema={
            "source_model": pl.String,
            "row_index": pl.Int64,
            "row_id": pl.String,
            "term": pl.String,
            "term_index": pl.Int64,
            "value": pl.Float64,
        }
    )
    cluster = _run(tmp_path / "c", _km3(), empty)
    assert cluster.df.height == 0
    assert cluster.secondary_tables["representative_docs"].height == 0
    assert cluster.metrics["n_clusters"] == 0

    topic = _run(tmp_path / "t", _topic_ml(n_topics=3), empty)
    assert topic.df.height == 0
    assert topic.secondary_tables["topics"].height == 0
    assert topic.metrics["n_topics"] == 0
