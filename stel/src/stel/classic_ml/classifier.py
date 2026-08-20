"""Supervised text classifiers (issue #190, Workstream B).

Multinomial naive-Bayes fitting, prediction row shaping, classification
metrics, and the classifier-artifact reader/writer and payload validation.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, cast

import polars as pl

from ..adapters import WarehouseAdapter
from ..config.model import MLConfig, ModelConfig
from ..config.project import ProjectConfig
from ..dag import parse_ref
from ..ml_contracts import ExecutableMLContract
from ..versioning import compute_code_version
from .artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    ClassicMLArtifactError,
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
from .common import ClassicMLRun, _project_metrics, _source_rows, _training_input
from .contracts import ClassifierProvider
from .text import TextOptions, _analyze, _select_terms, _text_options


def _fit_naive_bayes(
    rows: list[dict[str, Any]],
    provider: ClassifierProvider,
    options: TextOptions,
    raw_options: dict[str, Any],
) -> dict[str, Any]:
    labeled_rows = [row for row in rows if row.get("label")]
    if not labeled_rows:
        raise ValueError("Classifier fitting requires at least one non-null label.")

    doc_tokens = [_analyze(row["text"], options) for row in labeled_rows]
    doc_freq: Counter[str] = Counter()
    for tokens in doc_tokens:
        doc_freq.update(set(tokens))
    vocabulary = _select_terms(doc_freq, len(labeled_rows), options)
    vocab_set = set(vocabulary)
    alpha = float(raw_options.get("alpha", 1.0))
    if alpha <= 0:
        raise ValueError("ml.options.alpha must be positive for builtin.naive_bayes.")

    class_doc_counts: Counter[str] = Counter(str(row["label"]) for row in labeled_rows)
    class_token_counts: dict[str, Counter[str]] = {
        label: Counter() for label in sorted(class_doc_counts)
    }
    class_total_tokens: Counter[str] = Counter()
    for row, tokens in zip(labeled_rows, doc_tokens, strict=True):
        label = str(row["label"])
        counts = Counter(token for token in tokens if token in vocab_set)
        class_token_counts[label].update(counts)
        class_total_tokens[label] += sum(counts.values())

    classes = sorted(class_doc_counts)
    n_docs = len(labeled_rows)
    n_classes = len(classes)
    vocab_size = max(1, len(vocabulary))
    class_log_prior = {
        label: math.log((class_doc_counts[label] + alpha) / (n_docs + alpha * n_classes))
        for label in classes
    }
    feature_log_prob: dict[str, dict[str, float]] = {}
    default_log_prob: dict[str, float] = {}
    for label in classes:
        denom = class_total_tokens[label] + alpha * vocab_size
        default_log_prob[label] = math.log(alpha / denom)
        feature_log_prob[label] = {
            term: math.log((class_token_counts[label][term] + alpha) / denom)
            for term in vocabulary
        }

    return {
        "provider": provider,
        "classes": classes,
        "vocabulary": vocabulary,
        "n_features": len(vocabulary),
        "options": dict(raw_options),
        "class_doc_counts": dict(class_doc_counts),
        "class_log_prior": class_log_prior,
        "feature_log_prob": feature_log_prob,
        "default_log_prob": default_log_prob,
        "alpha": alpha,
    }


def _classifier_prediction_rows(
    rows: list[dict[str, Any]],
    classifier: dict[str, Any],
    source_name: str,
) -> list[dict[str, Any]]:
    options = _text_options(classifier["options"])
    vocabulary = set(str(term) for term in classifier["vocabulary"])
    classes = [str(label) for label in classifier["classes"]]
    predictions: list[dict[str, Any]] = []
    for row in rows:
        counts = Counter(token for token in _analyze(row["text"], options) if token in vocabulary)
        log_scores: dict[str, float] = {}
        for label in classes:
            score = float(classifier["class_log_prior"][label])
            default = float(classifier["default_log_prob"][label])
            term_probs = classifier["feature_log_prob"][label]
            for term, count in counts.items():
                score += count * float(term_probs.get(term, default))
            log_scores[label] = score
        probabilities = _softmax(log_scores)
        prediction = max(probabilities, key=probabilities.__getitem__)
        actual_label = row.get("label")
        prediction_row: dict[str, Any] = {
            "source_model": source_name,
            "row_index": row["row_index"],
            "row_id": row["row_id"],
            "provider": classifier["provider"],
            "prediction": prediction,
            "score": probabilities[prediction],
            "probabilities": json.dumps(probabilities, sort_keys=True),
        }
        if actual_label is not None:
            prediction_row["label"] = actual_label
            prediction_row["correct"] = actual_label == prediction
        if "document_id" in row:
            prediction_row["document_id"] = row["document_id"]
        if "source_path" in row:
            prediction_row["source_path"] = row["source_path"]
        predictions.append(prediction_row)
    return predictions


def _softmax(log_scores: dict[str, float]) -> dict[str, float]:
    max_score = max(log_scores.values())
    exp_scores = {
        label: math.exp(score - max_score)
        for label, score in log_scores.items()
    }
    total = sum(exp_scores.values()) or 1.0
    return {
        label: exp_scores[label] / total
        for label in sorted(exp_scores)
    }


def _classifier_metrics(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    classifier: dict[str, Any],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "row_count": len(rows),
        "prediction_rows": len(predictions),
        "class_count": len(classifier["classes"]),
        "vocabulary_size": len(classifier["vocabulary"]),
    }
    labeled = [row for row in predictions if "correct" in row]
    if labeled:
        correct = sum(1 for row in labeled if row["correct"])
        metrics["accuracy"] = correct / len(labeled)
        metrics["labeled_row_count"] = len(labeled)
    return metrics


def _classifier_metadata(
    *,
    model: ModelConfig,
    ml: MLConfig,
    provider: ClassifierProvider,
    training_input: dict[str, Any],
    classifier: dict[str, Any],
    metrics: dict[str, Any],
    code_version: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "classic_ml",
        "model_name": model.name,
        "task": ml.task,
        "provider": provider,
        "mode": ml.mode,
        "text_field": ml.text_field,
        "label_field": ml.label_field,
        "code_version": code_version,
        "config_hash": _hash_json(
            {
                "task": ml.task,
                "provider": provider,
                "text_field": ml.text_field,
                "label_field": ml.label_field,
                "options": classifier["options"],
            }
        ),
        "runtime": _runtime_versions(provider),
        "training_input": training_input,
        "integrity": {
            "class_count": metrics["class_count"],
            "feature_count": len(classifier["vocabulary"]),
        },
        "files": ["metadata.json", "model.json"],
        "options": classifier["options"],
        "classes_hash": _hash_json(classifier["classes"]),
        "vocabulary_hash": _hash_json(classifier["vocabulary"]),
        "model_hash": _hash_json(_classifier_payload(classifier)),
    }
    if ml.artifact.include_metrics:
        metadata["metrics"] = _project_metrics(ml, metrics)
    return metadata


def _write_classifier_artifact(
    path: Path,
    metadata: dict[str, Any],
    classifier: dict[str, Any],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload = _classifier_payload(classifier)
    (path / "model.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    payload_files = ["model.json"]
    metadata["files"] = ["metadata.json", *payload_files]
    metadata["artifact_files_hash"] = _artifact_files_hash(path, payload_files, classifier)
    metadata["artifact_version"] = _artifact_version(metadata)
    _write_metadata(path, metadata)


def _read_classifier_artifact(
    path: Path,
    provider: ClassifierProvider,
    ml: MLConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = _read_metadata(path)
    _validate_metadata(
        metadata,
        path,
        provider,
        ml,
        expected_files=("metadata.json", "model.json"),
    )
    metadata_options_payload = metadata.get("options")
    metadata_options = _validated_persisted_options(
        provider,
        metadata_options_payload,
        path,
        surface="metadata",
    )
    model_path = path / "model.json"
    _validate_artifact_payload(metadata, path, {})
    classifier = _read_artifact_json(model_path, path, "classifier model")
    payload_options_raw = classifier.get("options")
    if isinstance(payload_options_raw, dict) and "alpha" not in payload_options_raw:
        payload_options_raw = {**payload_options_raw, "alpha": classifier.get("alpha")}
    payload_options = _validated_persisted_options(
        provider,
        payload_options_raw,
        path,
        surface="classifier payload",
    )
    if payload_options != metadata_options:
        raise IncompatibleClassicMLArtifactError(
            f"incompatible classifier options at {path}: metadata and payload differ"
        )
    classifier["options"] = payload_options
    classifier["alpha"] = float(payload_options["alpha"])
    _validate_classifier_payload(classifier, path, provider)
    integrity = metadata.get("integrity")
    if integrity is not None:
        if not isinstance(integrity, dict):
            raise IncompatibleClassicMLArtifactError(
                f"incompatible classifier integrity at {path}: expected an object"
            )
        class_count = integrity.get("class_count")
        feature_count = integrity.get("feature_count")
        if (
            isinstance(class_count, bool)
            or not isinstance(class_count, int)
            or class_count != len(classifier["classes"])
        ):
            raise IncompatibleClassicMLArtifactError(
                f"incompatible classifier integrity at {path}: class_count mismatch"
            )
        if (
            isinstance(feature_count, bool)
            or not isinstance(feature_count, int)
            or feature_count != len(classifier["vocabulary"])
        ):
            raise IncompatibleClassicMLArtifactError(
                f"incompatible classifier integrity at {path}: feature_count mismatch"
            )
    return metadata, classifier


def _validate_classifier_payload(
    classifier: dict[str, Any],
    path: Path,
    provider: ClassifierProvider,
) -> None:
    try:
        if classifier.get("provider") != provider:
            raise ValueError(
                f"expected provider {provider}, found {classifier.get('provider')!r}"
            )
        classes = classifier["classes"]
        vocabulary = classifier["vocabulary"]
        if (
            not isinstance(classes, list)
            or not classes
            or any(not isinstance(label, str) for label in classes)
            or len(classes) != len(set(classes))
        ):
            raise ValueError("classes must be a non-empty list of unique strings")
        if (
            not isinstance(vocabulary, list)
            or any(not isinstance(term, str) for term in vocabulary)
            or len(vocabulary) != len(set(vocabulary))
        ):
            raise ValueError("vocabulary must be a list of unique strings")
        if (
            isinstance(classifier["n_features"], bool)
            or not isinstance(classifier["n_features"], int)
            or classifier["n_features"] != len(vocabulary)
        ):
            raise ValueError("n_features must match vocabulary length")
        alpha = float(classifier["alpha"])
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")

        class_fields = ("class_doc_counts", "class_log_prior", "default_log_prob")
        for field in class_fields:
            values = classifier[field]
            if not isinstance(values, dict) or set(values) != set(classes):
                raise ValueError(f"{field} keys must match classes")
            if any(
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                for value in values.values()
            ):
                raise ValueError(f"{field} values must be finite numbers")

        probabilities = classifier["feature_log_prob"]
        if not isinstance(probabilities, dict) or set(probabilities) != set(classes):
            raise ValueError("feature_log_prob keys must match classes")
        for label in classes:
            values = probabilities[label]
            if not isinstance(values, dict) or set(values) != set(vocabulary):
                raise ValueError(
                    f"feature_log_prob[{label!r}] keys must match vocabulary"
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                for value in values.values()
            ):
                raise ValueError(
                    f"feature_log_prob[{label!r}] values must be finite numbers"
                )
    except ClassicMLArtifactError:
        raise
    except (KeyError, TypeError, ValueError) as e:
        raise IncompatibleClassicMLArtifactError(
            f"malformed classifier payload at {path}: {e}"
        ) from e


def _classifier_payload(classifier: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": classifier["provider"],
        "classes": classifier["classes"],
        "vocabulary": classifier["vocabulary"],
        "n_features": classifier["n_features"],
        "options": classifier["options"],
        "alpha": classifier["alpha"],
        "class_doc_counts": classifier["class_doc_counts"],
        "class_log_prior": classifier["class_log_prior"],
        "feature_log_prob": classifier["feature_log_prob"],
        "default_log_prob": classifier["default_log_prob"],
    }


def _empty_prediction_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "source_model": pl.String,
            "row_index": pl.Int64,
            "row_id": pl.String,
            "provider": pl.String,
            "prediction": pl.String,
            "score": pl.Float64,
            "probabilities": pl.String,
            "label": pl.String,
            "correct": pl.Boolean,
        }
    )


def _run_classifier(
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
    if not ml.text_field:
        raise ValueError(f"ML model '{model.name}' requires ml.text_field.")
    if ml.mode in {"fit_transform", "fit"} and not ml.label_field:
        raise ValueError(f"Classifier model '{model.name}' requires ml.label_field for fitting.")

    provider = cast(ClassifierProvider, contract.provider)
    options = _text_options(contract.options)
    artifact_path = contract.artifact_path
    if ml.mode in {"predict", "load_pretrained"}:
        metadata, classifier = _read_classifier_artifact(artifact_path, provider, ml)

    source_name = parse_ref(model.depends_on[0])
    source_df = adapter.read_table(source_name)
    if ml.text_field not in source_df.columns:
        raise ValueError(
            f"ML model '{model.name}' text_field '{ml.text_field}' "
            f"is not present in '{source_name}'."
        )
    if ml.label_field and ml.label_field not in source_df.columns:
        raise ValueError(
            f"ML model '{model.name}' label_field '{ml.label_field}' "
            f"is not present in '{source_name}'."
        )

    rows = _source_rows(source_df, ml.text_field, ml.label_field)
    training_input = _training_input(model.depends_on, rows)
    code_version = compute_code_version(
        extraction=None,
        transform=None,
        ml=ml,
        project_dir=project_dir,
    )

    if ml.mode in {"fit_transform", "fit"}:
        classifier = _fit_naive_bayes(rows, provider, options, contract.options)
        predictions = _classifier_prediction_rows(rows, classifier, source_name)
        all_metrics = _classifier_metrics(rows, predictions, classifier)
        metadata = _classifier_metadata(
            model=model,
            ml=ml,
            provider=provider,
            training_input=training_input,
            classifier=classifier,
            metrics=all_metrics,
            code_version=code_version,
        )
        staged_path = _new_artifact_staging_path(artifact_path)
        try:
            _write_classifier_artifact(staged_path, metadata, classifier)
            metadata, _ = _read_classifier_artifact(staged_path, provider, ml)
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
    elif ml.mode in {"predict", "load_pretrained"}:
        predictions = _classifier_prediction_rows(rows, classifier, source_name)
        all_metrics = _classifier_metrics(rows, predictions, classifier)
        publication = None

    if ml.mode == "fit":
        df = pl.DataFrame(
            [
                {
                    "artifact_version": metadata["artifact_version"],
                    "row_count": len(rows),
                    "class_count": len(classifier["classes"]),
                    "vocabulary_size": len(classifier["vocabulary"]),
                    "accuracy": all_metrics.get("accuracy"),
                }
            ]
        )
    else:
        df = pl.DataFrame(predictions) if predictions else _empty_prediction_df()

    return ClassicMLRun(
        df=df,
        artifact_path=artifact_path,
        artifact_version=str(metadata["artifact_version"]),
        training_input=metadata.get("training_input", training_input),
        metrics=_project_metrics(ml, all_metrics),
        artifact_metadata=metadata,
        _publication=publication,
    )
