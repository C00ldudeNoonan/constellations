"""Classic ML transformation primitives.

Dispatch only: `run_classic_ml_model` routes a validated ML contract to its
algorithm family. Responsibilities live in sibling modules (issue #190,
Workstream B):

- `artifacts` — versioned artifact envelope, validation, atomic publication,
  recovery, and the registry
- `common` — run contract, deterministic source rows, metrics projection
- `text` — text feature providers (analyzer, vectorizers, feature rows)
- `classifier` — supervised classification
- `matrix` — matrix assembly and unsupervised models (cluster, topic, kNN)

Every name those modules define is re-exported below, so the historical flat
`dbt_ml.classic_ml` namespace keeps working for callers and tests.
"""

from __future__ import annotations

from pathlib import Path

from ..adapters import WarehouseAdapter
from ..config.model import ModelConfig
from ..config.project import ProjectConfig
from ..ml_contracts import validate_ml_contract
from .artifacts import ARTIFACT_REGISTRY_FILENAME as ARTIFACT_REGISTRY_FILENAME
from .artifacts import ARTIFACT_SCHEMA_VERSION as ARTIFACT_SCHEMA_VERSION
from .artifacts import ClassicMLArtifactError as ClassicMLArtifactError
from .artifacts import ClassicMLArtifactPublication as ClassicMLArtifactPublication
from .artifacts import IncompatibleClassicMLArtifactError as IncompatibleClassicMLArtifactError
from .artifacts import MissingClassicMLArtifactError as MissingClassicMLArtifactError
from .artifacts import StaleClassicMLArtifactError as StaleClassicMLArtifactError
from .artifacts import _artifact_files_hash as _artifact_files_hash
from .artifacts import _artifact_publication as _artifact_publication
from .artifacts import _artifact_version as _artifact_version
from .artifacts import _artifact_version_at as _artifact_version_at
from .artifacts import _atomic_write_json as _atomic_write_json
from .artifacts import _cleanup_committed_publication as _cleanup_committed_publication
from .artifacts import _display_path as _display_path
from .artifacts import _exclusive_file_lock as _exclusive_file_lock
from .artifacts import _hash_json as _hash_json
from .artifacts import _new_artifact_staging_path as _new_artifact_staging_path
from .artifacts import _package_version as _package_version
from .artifacts import _publish_registry as _publish_registry
from .artifacts import _publish_staged_artifact as _publish_staged_artifact
from .artifacts import _read_artifact_json as _read_artifact_json
from .artifacts import _read_artifact_registry as _read_artifact_registry
from .artifacts import _read_json_object as _read_json_object
from .artifacts import _read_metadata as _read_metadata
from .artifacts import _recover_artifact_publications as _recover_artifact_publications
from .artifacts import _recover_pending_publications as _recover_pending_publications
from .artifacts import _remove_path as _remove_path
from .artifacts import _rollback_publication as _rollback_publication
from .artifacts import _runtime_versions as _runtime_versions
from .artifacts import _validate_artifact_payload as _validate_artifact_payload
from .artifacts import _validate_metadata as _validate_metadata
from .artifacts import _validated_journal_paths as _validated_journal_paths
from .artifacts import _validated_persisted_options as _validated_persisted_options
from .artifacts import _write_artifact_payload as _write_artifact_payload
from .artifacts import _write_metadata as _write_metadata
from .classifier import _classifier_metadata as _classifier_metadata
from .classifier import _classifier_metrics as _classifier_metrics
from .classifier import _classifier_payload as _classifier_payload
from .classifier import _classifier_prediction_rows as _classifier_prediction_rows
from .classifier import _empty_prediction_df as _empty_prediction_df
from .classifier import _fit_naive_bayes as _fit_naive_bayes
from .classifier import _read_classifier_artifact as _read_classifier_artifact
from .classifier import _run_classifier as _run_classifier
from .classifier import _softmax as _softmax
from .classifier import _validate_classifier_payload as _validate_classifier_payload
from .classifier import _write_classifier_artifact as _write_classifier_artifact
from .common import ClassicMLRun as ClassicMLRun
from .common import _canonical_row_key as _canonical_row_key
from .common import _project_metrics as _project_metrics
from .common import _source_rows as _source_rows
from .common import _training_input as _training_input
from .contracts import Analyzer as Analyzer
from .contracts import ClassifierProvider as ClassifierProvider
from .contracts import FeatureProvider as FeatureProvider
from .matrix import _ML_FEATURE as _ML_FEATURE
from .matrix import _apply_matrix_model as _apply_matrix_model
from .matrix import _assemble_embedding_matrix as _assemble_embedding_matrix
from .matrix import _assemble_feature_matrix as _assemble_feature_matrix
from .matrix import _assemble_matrix as _assemble_matrix
from .matrix import _attach_doc_identity as _attach_doc_identity
from .matrix import _cluster_geometry as _cluster_geometry
from .matrix import _cluster_metrics as _cluster_metrics
from .matrix import _cluster_primary_rows as _cluster_primary_rows
from .matrix import _cluster_term_rows as _cluster_term_rows
from .matrix import _dataframe_or_empty as _dataframe_or_empty
from .matrix import _empty_cluster_df as _empty_cluster_df
from .matrix import _empty_document_topics_df as _empty_document_topics_df
from .matrix import _empty_neighbors_df as _empty_neighbors_df
from .matrix import _empty_representative_docs_df as _empty_representative_docs_df
from .matrix import _empty_topics_df as _empty_topics_df
from .matrix import _fit_cluster as _fit_cluster
from .matrix import _fit_matrix_model as _fit_matrix_model
from .matrix import _fit_topics as _fit_topics
from .matrix import _l2_normalize as _l2_normalize
from .matrix import _matrix_fit_summary as _matrix_fit_summary
from .matrix import _matrix_metadata as _matrix_metadata
from .matrix import _matrix_model_payload as _matrix_model_payload
from .matrix import _matrix_outputs as _matrix_outputs
from .matrix import _matrix_training_input as _matrix_training_input
from .matrix import _MatrixDoc as _MatrixDoc
from .matrix import _neighbor_rows as _neighbor_rows
from .matrix import _normalize_rows as _normalize_rows
from .matrix import _numpy as _numpy
from .matrix import _read_matrix_artifact as _read_matrix_artifact
from .matrix import _representative_docs_rows as _representative_docs_rows
from .matrix import _run_matrix_model as _run_matrix_model
from .matrix import _silhouette as _silhouette
from .matrix import _sklearn as _sklearn
from .matrix import _topic_coherence as _topic_coherence
from .matrix import _topic_document_rows as _topic_document_rows
from .matrix import _topic_metrics as _topic_metrics
from .matrix import _topic_term_rows as _topic_term_rows
from .matrix import _write_matrix_artifact as _write_matrix_artifact
from .text import _ENGLISH_STOP_WORDS as _ENGLISH_STOP_WORDS
from .text import _TOKEN_RE as _TOKEN_RE
from .text import TextOptions as TextOptions
from .text import _analyze as _analyze
from .text import _base_feature_row as _base_feature_row
from .text import _char_ngrams as _char_ngrams
from .text import _char_wb_ngrams as _char_wb_ngrams
from .text import _df_threshold as _df_threshold
from .text import _empty_feature_df as _empty_feature_df
from .text import _feature_rows as _feature_rows
from .text import _fit_hashing_vectorizer as _fit_hashing_vectorizer
from .text import _fit_vectorizer as _fit_vectorizer
from .text import _hashed_feature_rows as _hashed_feature_rows
from .text import _metadata as _metadata
from .text import _ngram_range as _ngram_range
from .text import _optional_int as _optional_int
from .text import _read_artifact as _read_artifact
from .text import _run_features as _run_features
from .text import _select_terms as _select_terms
from .text import _stop_words as _stop_words
from .text import _text_options as _text_options
from .text import _token_ngrams as _token_ngrams
from .text import _write_artifact as _write_artifact


def run_classic_ml_model(
    *,
    model: ModelConfig,
    project: ProjectConfig,
    project_dir: Path,
    adapter: WarehouseAdapter,
) -> ClassicMLRun:
    assert model.ml is not None
    _recover_artifact_publications(project, project_dir)
    contract = validate_ml_contract(model, project, project_dir)
    if contract.task == "features":
        return _run_features(
            model=model,
            ml=model.ml,
            contract=contract,
            project=project,
            project_dir=project_dir,
            adapter=adapter,
        )
    if contract.task in {"cluster", "topic_model"}:
        return _run_matrix_model(
            model=model,
            ml=model.ml,
            contract=contract,
            project=project,
            project_dir=project_dir,
            adapter=adapter,
        )
    return _run_classifier(
        model=model,
        ml=model.ml,
        contract=contract,
        project=project,
        project_dir=project_dir,
        adapter=adapter,
    )
