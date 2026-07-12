from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import polars as pl

from .adapters import WarehouseAdapter
from .config.model import MLConfig, ModelConfig
from .config.project import ProjectConfig
from .dag import parse_ref
from .paths import resolve_within_project
from .versioning import compute_code_version

# v2 (issue #122): canonical training-row order, vectorizer-convention
# min_df/max_df rounding, and an independent hashing sign bit — features and
# hashes from v1 artifacts are not comparable, so v1 artifacts are rejected
# with a refit hint rather than silently reused.
ARTIFACT_SCHEMA_VERSION = 2
ARTIFACT_REGISTRY_FILENAME = "registry.json"
_TOKEN_RE = re.compile(r"\w+")
_FEATURE_PROVIDERS = {"builtin.count", "builtin.tfidf", "builtin.hashing"}
_CLASSIFIER_PROVIDERS = {"builtin.naive_bayes"}
_ENGLISH_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}

FeatureProvider = Literal["builtin.count", "builtin.tfidf", "builtin.hashing"]
ClassifierProvider = Literal["builtin.naive_bayes"]
Analyzer = Literal["word", "char", "char_wb"]


class TextOptions(TypedDict):
    analyzer: Analyzer
    lowercase: bool
    token_pattern: str
    ngram_range: tuple[int, int]
    stop_words: set[str]
    min_df: int | float
    max_df: int | float | None
    max_features: int | None
    binary: bool
    n_features: int
    alternate_sign: bool


@dataclass
class ClassicMLRun:
    df: pl.DataFrame
    artifact_path: Path
    artifact_version: str
    training_input: dict[str, Any]
    metrics: dict[str, Any]
    artifact_metadata: dict[str, Any]


class ClassicMLArtifactError(ValueError):
    pass


class MissingClassicMLArtifactError(ClassicMLArtifactError, FileNotFoundError):
    pass


class StaleClassicMLArtifactError(ClassicMLArtifactError):
    pass


class IncompatibleClassicMLArtifactError(ClassicMLArtifactError):
    pass


def run_classic_ml_model(
    *,
    model: ModelConfig,
    project: ProjectConfig,
    project_dir: Path,
    adapter: WarehouseAdapter,
) -> ClassicMLRun:
    assert model.ml is not None
    if model.ml.task == "features":
        provider = _feature_provider(model.ml.provider)
        return _run_features(
            model=model,
            ml=model.ml,
            provider=provider,
            project=project,
            project_dir=project_dir,
            adapter=adapter,
        )
    if model.ml.task == "classifier":
        classifier_provider = _classifier_provider(model.ml.provider)
        return _run_classifier(
            model=model,
            ml=model.ml,
            provider=classifier_provider,
            project=project,
            project_dir=project_dir,
            adapter=adapter,
        )
    raise NotImplementedError(
        f"ML task '{model.ml.task}' is not executable yet; "
        "supported tasks are 'features' and 'classifier'."
    )


def _feature_provider(provider: str | None) -> FeatureProvider:
    provider = provider or "builtin.tfidf"
    if provider not in _FEATURE_PROVIDERS:
        raise NotImplementedError(
            f"ML provider '{provider}' is not executable yet for task 'features'; "
            "supported feature providers are builtin.count, builtin.tfidf, and builtin.hashing."
        )
    return cast(FeatureProvider, provider)


def _classifier_provider(provider: str | None) -> ClassifierProvider:
    provider = provider or "builtin.naive_bayes"
    if provider not in _CLASSIFIER_PROVIDERS:
        raise NotImplementedError(
            f"ML provider '{provider}' is not executable yet for task 'classifier'; "
            "supported classifier provider is builtin.naive_bayes."
        )
    return cast(ClassifierProvider, provider)


def _run_features(
    *,
    model: ModelConfig,
    ml: MLConfig,
    provider: FeatureProvider,
    project: ProjectConfig,
    project_dir: Path,
    adapter: WarehouseAdapter,
) -> ClassicMLRun:
    if not model.depends_on:
        raise ValueError(f"ML model '{model.name}' must declare depends_on.")
    if not ml.text_field:
        raise ValueError(f"ML model '{model.name}' requires ml.text_field.")

    source_name = parse_ref(model.depends_on[0])
    source_df = adapter.query_df(f"SELECT * FROM {adapter.table_ref(source_name)}")
    if ml.text_field not in source_df.columns:
        raise ValueError(
            f"ML model '{model.name}' text_field '{ml.text_field}' "
            f"is not present in '{source_name}'."
        )

    options = _text_options(ml.options)
    artifact_path = _artifact_path(ml, model, project, project_dir)
    rows = _source_rows(source_df, ml.text_field)
    training_input = _training_input(model.depends_on, rows)
    code_version = compute_code_version(
        extraction=None,
        transform=None,
        ml=ml,
        project_dir=project_dir,
    )

    if ml.mode in {"fit_transform", "fit"}:
        vectorizer = _fit_vectorizer(rows, provider, options)
        metadata = _metadata(
            model=model,
            ml=ml,
            provider=provider,
            training_input=training_input,
            vectorizer=vectorizer,
            options=options,
            code_version=code_version,
        )
        _write_artifact(artifact_path, metadata, vectorizer)
        metadata = _read_metadata(artifact_path)
        _write_artifact_registry(
            project=project,
            project_dir=project_dir,
            model=model,
            artifact_path=artifact_path,
            metadata=metadata,
        )
    elif ml.mode in {"predict", "load_pretrained"}:
        metadata, vectorizer = _read_artifact(artifact_path, provider, ml)
        options = _text_options(vectorizer["options"])
    else:
        raise ValueError(f"Unsupported ML mode: {ml.mode}")

    doc_tokens = [_analyze(row["text"], options) for row in rows]
    features = _feature_rows(rows, doc_tokens, vectorizer, source_name)
    metrics = {
        "row_count": len(rows),
        "vocabulary_size": len(vectorizer["vocabulary"]),
        "feature_rows": len(features),
    }
    if provider == "builtin.hashing":
        metrics["hash_buckets"] = vectorizer["n_features"]

    if ml.mode == "fit":
        df = pl.DataFrame(
            [
                {
                    "artifact_version": metadata["artifact_version"],
                    "row_count": len(rows),
                    "vocabulary_size": len(vectorizer["vocabulary"]),
                    "feature_rows": len(features),
                }
            ]
        )
    else:
        df = pl.DataFrame(features) if features else _empty_feature_df()

    return ClassicMLRun(
        df=df,
        artifact_path=artifact_path,
        artifact_version=str(metadata["artifact_version"]),
        training_input=metadata.get("training_input", training_input),
        metrics=metrics,
        artifact_metadata=metadata,
    )


def _run_classifier(
    *,
    model: ModelConfig,
    ml: MLConfig,
    provider: ClassifierProvider,
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

    source_name = parse_ref(model.depends_on[0])
    source_df = adapter.query_df(f"SELECT * FROM {adapter.table_ref(source_name)}")
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

    options = _text_options(ml.options)
    artifact_path = _artifact_path(ml, model, project, project_dir)
    rows = _source_rows(source_df, ml.text_field, ml.label_field)
    training_input = _training_input(model.depends_on, rows)
    code_version = compute_code_version(
        extraction=None,
        transform=None,
        ml=ml,
        project_dir=project_dir,
    )

    if ml.mode in {"fit_transform", "fit"}:
        classifier = _fit_naive_bayes(rows, provider, options, ml.options)
        predictions = _classifier_prediction_rows(rows, classifier, source_name)
        metrics = _classifier_metrics(rows, predictions, classifier)
        metadata = _classifier_metadata(
            model=model,
            ml=ml,
            provider=provider,
            training_input=training_input,
            classifier=classifier,
            metrics=metrics,
            code_version=code_version,
        )
        _write_classifier_artifact(artifact_path, metadata, classifier)
        metadata = _read_metadata(artifact_path)
        _write_artifact_registry(
            project=project,
            project_dir=project_dir,
            model=model,
            artifact_path=artifact_path,
            metadata=metadata,
        )
    elif ml.mode in {"predict", "load_pretrained"}:
        metadata, classifier = _read_classifier_artifact(artifact_path, provider, ml)
        predictions = _classifier_prediction_rows(rows, classifier, source_name)
        metrics = _classifier_metrics(rows, predictions, classifier)
    else:
        raise ValueError(f"Unsupported ML mode: {ml.mode}")

    if ml.mode == "fit":
        df = pl.DataFrame(
            [
                {
                    "artifact_version": metadata["artifact_version"],
                    "row_count": len(rows),
                    "class_count": len(classifier["classes"]),
                    "vocabulary_size": len(classifier["vocabulary"]),
                    "accuracy": metrics.get("accuracy"),
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
        metrics=metrics,
        artifact_metadata=metadata,
    )


def _artifact_path(
    ml: MLConfig,
    model: ModelConfig,
    project: ProjectConfig,
    project_dir: Path,
) -> Path:
    if ml.artifact.path is not None:
        return resolve_within_project(
            ml.artifact.path,
            project_dir,
            surface=f"Model '{model.name}' ml.artifact.path",
            external=ml.artifact.external,
            hint="Set `external: true` on the artifact block to allow it.",
        )
    return project_dir / project.target_path / "artifacts" / model.name


def _canonical_row_key(row: dict[str, Any]) -> tuple[int, str, str]:
    """Warehouses return `SELECT *` in arbitrary order; training input must
    not depend on it. Order by the stable row identifier when present —
    chunk_id before document_id, since chunk models repeat document_id
    across a document's chunks — with canonical row content breaking any
    remaining ties (fully identical rows are interchangeable)."""
    content = json.dumps(row, sort_keys=True, default=str)
    for key in ("chunk_id", "document_id", "id"):
        value = row.get(key)
        if value is not None:
            return (0, str(value), content)
    return (1, content, "")


def _source_rows(
    df: pl.DataFrame,
    text_field: str,
    label_field: str | None = None,
) -> list[dict[str, Any]]:
    ordered = sorted(df.iter_rows(named=True), key=_canonical_row_key)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(ordered):
        text = "" if row[text_field] is None else str(row[text_field])
        row_id = str(row.get("document_id") or row.get("id") or index)
        payload: dict[str, Any] = {"row_index": index, "row_id": row_id, "text": text}
        if label_field is not None:
            payload["label"] = None if row[label_field] is None else str(row[label_field])
        if "document_id" in row:
            payload["document_id"] = row["document_id"]
        if "source_path" in row:
            payload["source_path"] = row["source_path"]
        rows.append(payload)
    return rows


def _training_input(depends_on: list[str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    content = [
        {
            key: row[key]
            for key in ("row_id", "text", "label")
            if key in row
        }
        for row in rows
    ]
    raw = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return {
        "refs": [parse_ref(ref) for ref in depends_on],
        "row_count": len(rows),
        "content_hash": hashlib.blake2b(raw.encode(), digest_size=8).hexdigest(),
    }


def _text_options(options: dict[str, Any]) -> TextOptions:
    analyzer = str(options.get("analyzer", "word"))
    if analyzer not in {"word", "char", "char_wb"}:
        raise ValueError("ml.options.analyzer must be one of: word, char, char_wb")
    ngram_range = _ngram_range(options.get("ngram_range", [1, 1]))
    return {
        "analyzer": analyzer,  # type: ignore[typeddict-item]
        "lowercase": bool(options.get("lowercase", True)),
        "token_pattern": str(options.get("token_pattern", _TOKEN_RE.pattern)),
        "ngram_range": ngram_range,
        "stop_words": _stop_words(options.get("stop_words")),
        "min_df": options.get("min_df", 1),
        "max_df": options.get("max_df"),
        "max_features": _optional_int(options.get("max_features")),
        "binary": bool(options.get("binary", False)),
        "n_features": int(options.get("n_features", 2**20)),
        "alternate_sign": bool(options.get("alternate_sign", True)),
    }


def _ngram_range(value: Any) -> tuple[int, int]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError("ml.options.ngram_range must be a two-item list.")
    min_n = int(value[0])
    max_n = int(value[1])
    if min_n <= 0 or max_n < min_n:
        raise ValueError("ml.options.ngram_range must be positive and ordered.")
    return min_n, max_n


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _stop_words(value: Any) -> set[str]:
    if value is None:
        return set()
    if value == "english":
        return set(_ENGLISH_STOP_WORDS)
    if not isinstance(value, list):
        raise ValueError("ml.options.stop_words must be a list of terms or 'english'.")
    return {str(term).lower() for term in value}


def _fit_vectorizer(
    rows: list[dict[str, Any]],
    provider: FeatureProvider,
    options: TextOptions,
) -> dict[str, Any]:
    if provider == "builtin.hashing":
        return _fit_hashing_vectorizer(provider, options)

    doc_tokens = [_analyze(row["text"], options) for row in rows]
    doc_freq: Counter[str] = Counter()
    for tokens in doc_tokens:
        doc_freq.update(set(tokens))

    terms = _select_terms(doc_freq, len(rows), options)
    idf_by_term: dict[str, float] = {}
    if provider == "builtin.tfidf":
        n_docs = max(1, len(rows))
        idf_by_term = {
            term: math.log((1 + n_docs) / (1 + doc_freq[term])) + 1
            for term in terms
        }
    return {
        "provider": provider,
        "vocabulary": terms,
        "idf": idf_by_term,
        "n_features": len(terms),
        "options": _serializable_options(options),
    }


def _fit_hashing_vectorizer(provider: FeatureProvider, options: TextOptions) -> dict[str, Any]:
    n_features = options["n_features"]
    if n_features <= 0:
        raise ValueError("ml.options.n_features must be positive for builtin.hashing.")
    return {
        "provider": provider,
        "vocabulary": [],
        "idf": {},
        "n_features": n_features,
        "options": _serializable_options(options),
    }


def _select_terms(
    doc_freq: Counter[str],
    n_docs: int,
    options: TextOptions,
) -> list[str]:
    if n_docs == 0:
        return []
    min_count = _df_threshold(options["min_df"], n_docs, default=1, ceiling=True)
    max_count = _df_threshold(options["max_df"], n_docs, default=n_docs, ceiling=False)
    terms = [
        term for term, count in doc_freq.items()
        if count >= min_count and count <= max_count
    ]
    terms.sort(key=lambda t: (-doc_freq[t], t))
    if options["max_features"] is not None:
        terms = terms[: options["max_features"]]
    terms.sort()
    return terms


def _df_threshold(
    value: int | float | None,
    n_docs: int,
    *,
    default: int,
    ceiling: bool,
) -> int:
    """Vectorizer semantics: a proportional min_df keeps terms appearing in
    at least that fraction of documents (df >= ceil(min_df * n)), and a
    proportional max_df keeps terms in at most that fraction
    (df <= floor(max_df * n))."""
    if value is None:
        return default
    if isinstance(value, float) and 0 < value <= 1:
        scaled = value * n_docs
        return math.ceil(scaled) if ceiling else math.floor(scaled)
    return int(value)


def _analyze(text: str, options: TextOptions) -> list[str]:
    if options["lowercase"]:
        text = text.lower()
    if options["analyzer"] == "word":
        tokens = re.findall(options["token_pattern"], text)
        tokens = [token for token in tokens if token not in options["stop_words"]]
        return _token_ngrams(tokens, options["ngram_range"])
    if options["analyzer"] == "char_wb":
        return _char_wb_ngrams(text, options["ngram_range"])
    return _char_ngrams(text, options["ngram_range"])


def _token_ngrams(tokens: list[str], ngram_range: tuple[int, int]) -> list[str]:
    min_n, max_n = ngram_range
    out: list[str] = []
    for n in range(min_n, max_n + 1):
        if len(tokens) < n:
            continue
        out.extend(" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return out


def _char_ngrams(text: str, ngram_range: tuple[int, int]) -> list[str]:
    min_n, max_n = ngram_range
    out: list[str] = []
    for n in range(min_n, max_n + 1):
        if len(text) < n:
            continue
        out.extend(text[i : i + n] for i in range(len(text) - n + 1))
    return out


def _char_wb_ngrams(text: str, ngram_range: tuple[int, int]) -> list[str]:
    out: list[str] = []
    for token in text.split():
        out.extend(_char_ngrams(f" {token} ", ngram_range))
    return out


def _feature_rows(
    rows: list[dict[str, Any]],
    doc_tokens: list[list[str]],
    vectorizer: dict[str, Any],
    source_name: str,
) -> list[dict[str, Any]]:
    provider = str(vectorizer["provider"])
    if provider == "builtin.hashing":
        return _hashed_feature_rows(rows, doc_tokens, vectorizer, source_name)

    vocabulary = [str(term) for term in vectorizer["vocabulary"]]
    term_index = {term: i for i, term in enumerate(vocabulary)}
    vocab_set = set(vocabulary)
    idf_by_term = {str(k): float(v) for k, v in vectorizer["idf"].items()}
    features: list[dict[str, Any]] = []
    for row, tokens in zip(rows, doc_tokens, strict=True):
        counts = Counter(t for t in tokens if t in vocab_set)
        binary = bool(vectorizer["options"]["binary"])
        total = (len(counts) if binary else sum(counts.values())) or 1
        for term in sorted(counts):
            count = 1 if binary else counts[term]
            tf = count / total
            idf = idf_by_term.get(term)
            value = tf * idf if idf is not None else float(count)
            features.append(
                _base_feature_row(
                    row=row,
                    source_name=source_name,
                    provider=provider,
                    feature_name=term,
                    term_index=term_index[term],
                    count=count,
                    tf=tf,
                    idf=idf,
                    value=value,
                    hash_bucket=None,
                )
            )
    return features


def _hashed_feature_rows(
    rows: list[dict[str, Any]],
    doc_tokens: list[list[str]],
    vectorizer: dict[str, Any],
    source_name: str,
) -> list[dict[str, Any]]:
    options = vectorizer["options"]
    n_features = int(vectorizer["n_features"])
    features: list[dict[str, Any]] = []
    for row, tokens in zip(rows, doc_tokens, strict=True):
        bucket_values: Counter[int] = Counter()
        for token in tokens:
            # The sign bit comes from a digest byte the bucket never sees:
            # deriving both from one value ties sign to bucket parity
            # whenever n_features is even, biasing collisions.
            digest = hashlib.blake2b(token.encode(), digest_size=9).digest()
            hashed = int.from_bytes(digest[:8], byteorder="big", signed=False)
            bucket = hashed % n_features
            sign = -1 if options["alternate_sign"] and digest[8] & 1 else 1
            bucket_values[bucket] += sign
        for bucket in sorted(bucket_values):
            value = float(bucket_values[bucket])
            features.append(
                _base_feature_row(
                    row=row,
                    source_name=source_name,
                    provider=str(vectorizer["provider"]),
                    feature_name=f"hash_{bucket}",
                    term_index=bucket,
                    count=int(abs(bucket_values[bucket])),
                    tf=None,
                    idf=None,
                    value=value,
                    hash_bucket=bucket,
                )
            )
    return features


def _base_feature_row(
    *,
    row: dict[str, Any],
    source_name: str,
    provider: str,
    feature_name: str,
    term_index: int,
    count: int,
    tf: float | None,
    idf: float | None,
    value: float,
    hash_bucket: int | None,
) -> dict[str, Any]:
    feature_row: dict[str, Any] = {
        "source_model": source_name,
        "row_index": row["row_index"],
        "row_id": row["row_id"],
        "provider": provider,
        "term": feature_name,
        "term_index": term_index,
        "count": count,
        "tf": tf,
        "idf": idf,
        "tfidf": value if provider == "builtin.tfidf" else None,
        "value": value,
        "hash_bucket": hash_bucket,
    }
    if "document_id" in row:
        feature_row["document_id"] = row["document_id"]
    if "source_path" in row:
        feature_row["source_path"] = row["source_path"]
    return feature_row


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
        "options": _serializable_options(options),
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


def _metadata(
    *,
    model: ModelConfig,
    ml: MLConfig,
    provider: FeatureProvider,
    training_input: dict[str, Any],
    vectorizer: dict[str, Any],
    options: TextOptions,
    code_version: str,
) -> dict[str, Any]:
    files = ["metadata.json"]
    if provider != "builtin.hashing":
        files.append("vocabulary.json")
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "classic_ml",
        "model_name": model.name,
        "task": ml.task,
        "provider": provider,
        "mode": ml.mode,
        "text_field": ml.text_field,
        "code_version": code_version,
        "config_hash": _hash_json(
            {
                "task": ml.task,
                "provider": provider,
                "text_field": ml.text_field,
                "options": _serializable_options(options),
            }
        ),
        "runtime": _runtime_versions(provider),
        "training_input": training_input,
        "metrics": {
            "row_count": training_input["row_count"],
            "vocabulary_size": len(vectorizer["vocabulary"]),
            "feature_count": vectorizer["n_features"],
        },
        "files": files,
        "options": _serializable_options(options),
        "vocabulary_hash": _hash_json(vectorizer["vocabulary"]),
        "idf_hash": _hash_json(vectorizer["idf"]),
    }


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
    return {
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
                "options": {
                    "text": classifier["options"],
                    "alpha": classifier["alpha"],
                },
            }
        ),
        "runtime": _runtime_versions(provider),
        "training_input": training_input,
        "metrics": {
            "row_count": metrics["row_count"],
            "class_count": metrics["class_count"],
            "vocabulary_size": metrics["vocabulary_size"],
            "accuracy": metrics.get("accuracy"),
        },
        "files": ["metadata.json", "model.json"],
        "options": classifier["options"],
        "classifier_options": {"alpha": classifier["alpha"]},
        "classes_hash": _hash_json(classifier["classes"]),
        "vocabulary_hash": _hash_json(classifier["vocabulary"]),
        "model_hash": _hash_json(_classifier_payload(classifier)),
    }


def _write_artifact(
    path: Path,
    metadata: dict[str, Any],
    vectorizer: dict[str, Any],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload_files = _write_artifact_payload(path, vectorizer)
    metadata["files"] = ["metadata.json", *payload_files]
    metadata["artifact_files_hash"] = _artifact_files_hash(path, payload_files, vectorizer)
    metadata["artifact_version"] = _artifact_version(metadata)
    _write_metadata(path, metadata)


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


def _read_artifact(
    path: Path,
    provider: FeatureProvider,
    ml: MLConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = _read_metadata(path)
    _validate_metadata(metadata, path, provider, ml)
    if provider == "builtin.hashing":
        vectorizer = {
            "provider": provider,
            "vocabulary": [],
            "idf": {},
            "n_features": metadata["metrics"]["feature_count"],
            "options": metadata["options"],
        }
        _validate_artifact_payload(metadata, path, vectorizer)
        return metadata, vectorizer

    vocab_path = path / "vocabulary.json"
    if not vocab_path.exists():
        raise MissingClassicMLArtifactError(
            f"missing artifact payload 'vocabulary.json' at {path}; "
            "run fit or fit_transform again"
        )
    vocab_payload = json.loads(vocab_path.read_text())
    vectorizer = {
        "provider": provider,
        "vocabulary": [str(t) for t in vocab_payload["terms"]],
        "idf": {str(k): float(v) for k, v in vocab_payload["idf"].items()},
        "n_features": len(vocab_payload["terms"]),
        "options": vocab_payload["options"],
    }
    _validate_artifact_payload(metadata, path, vectorizer)
    return metadata, vectorizer


def _read_classifier_artifact(
    path: Path,
    provider: ClassifierProvider,
    ml: MLConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = _read_metadata(path)
    _validate_metadata(metadata, path, provider, ml)
    model_path = path / "model.json"
    if not model_path.exists():
        raise MissingClassicMLArtifactError(
            f"missing artifact payload 'model.json' at {path}; run fit or fit_transform again"
        )
    classifier = cast(dict[str, Any], json.loads(model_path.read_text()))
    _validate_artifact_payload(metadata, path, classifier)
    return metadata, classifier


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


def _write_artifact_payload(path: Path, vectorizer: dict[str, Any]) -> list[str]:
    if vectorizer["provider"] == "builtin.hashing":
        return []
    payload = {
        "provider": vectorizer["provider"],
        "terms": vectorizer["vocabulary"],
        "idf": vectorizer["idf"],
        "options": vectorizer["options"],
    }
    (path / "vocabulary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    return ["vocabulary.json"]


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))


def _read_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        raise MissingClassicMLArtifactError(
            f"missing artifact metadata at {metadata_path}; run fit or fit_transform first"
        )
    return cast(dict[str, Any], json.loads(metadata_path.read_text()))


def _validate_metadata(
    metadata: dict[str, Any],
    path: Path,
    provider: str,
    ml: MLConfig,
) -> None:
    schema_version = metadata.get("artifact_schema_version")
    if schema_version != ARTIFACT_SCHEMA_VERSION:
        raise IncompatibleClassicMLArtifactError(
            f"incompatible artifact schema at {path}: expected "
            f"{ARTIFACT_SCHEMA_VERSION}, found {schema_version!r}; "
            "feature semantics changed - run fit or fit_transform to rebuild"
        )
    if metadata.get("artifact_type") != "classic_ml":
        raise IncompatibleClassicMLArtifactError(
            f"incompatible artifact type at {path}: {metadata.get('artifact_type')!r}"
        )
    if metadata.get("provider") != provider:
        raise IncompatibleClassicMLArtifactError(
            f"incompatible artifact provider at {path}: expected {provider}, "
            f"found {metadata.get('provider')!r}"
        )
    if metadata.get("task") != ml.task:
        raise IncompatibleClassicMLArtifactError(
            f"incompatible artifact task at {path}: expected {ml.task}, "
            f"found {metadata.get('task')!r}"
        )
    expected_version = _artifact_version(metadata)
    if metadata.get("artifact_version") != expected_version:
        raise StaleClassicMLArtifactError(
            f"stale artifact metadata at {path}: artifact_version does not match metadata"
        )


def _validate_artifact_payload(
    metadata: dict[str, Any],
    path: Path,
    vectorizer: dict[str, Any],
) -> None:
    payload_files = [f for f in metadata.get("files", []) if f != "metadata.json"]
    actual_hash = _artifact_files_hash(path, payload_files, vectorizer)
    expected_hash = metadata.get("artifact_files_hash")
    if actual_hash != expected_hash:
        raise StaleClassicMLArtifactError(
            f"stale artifact payload at {path}: artifact_files_hash does not match files"
        )


def _artifact_version(metadata: dict[str, Any]) -> str:
    payload = {
        key: value for key, value in metadata.items()
        if key != "artifact_version"
    }
    return _hash_json(payload)


def _artifact_files_hash(
    path: Path,
    payload_files: list[str],
    vectorizer: dict[str, Any],
) -> str:
    if not payload_files:
        return _hash_json(
            {
                "provider": vectorizer["provider"],
                "options": vectorizer["options"],
                "n_features": vectorizer["n_features"],
            }
        )
    h = hashlib.blake2b(digest_size=8)
    for filename in sorted(payload_files):
        file_path = path / filename
        if not file_path.exists():
            raise MissingClassicMLArtifactError(
                f"missing artifact payload '{filename}' at {path}; "
                "run fit or fit_transform again"
            )
        h.update(filename.encode())
        h.update(file_path.read_bytes())
    return h.hexdigest()


def _write_artifact_registry(
    *,
    project: ProjectConfig,
    project_dir: Path,
    model: ModelConfig,
    artifact_path: Path,
    metadata: dict[str, Any],
) -> None:
    registry_dir = project_dir / project.target_path / "artifacts"
    registry_dir.mkdir(parents=True, exist_ok=True)
    registry_path = registry_dir / ARTIFACT_REGISTRY_FILENAME
    registry = _read_artifact_registry(registry_path)
    registry["artifacts"][model.name] = {
        "model_name": model.name,
        "artifact_path": _display_path(artifact_path, project_dir),
        "artifact_version": metadata["artifact_version"],
        "provider": metadata["provider"],
        "task": metadata["task"],
        "code_version": metadata["code_version"],
        "config_hash": metadata["config_hash"],
        "artifact_files_hash": metadata["artifact_files_hash"],
        "training_input": metadata["training_input"],
        "metrics": metadata["metrics"],
    }
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True))


def _read_artifact_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"artifact_schema_version": ARTIFACT_SCHEMA_VERSION, "artifacts": {}}
    registry = json.loads(path.read_text())
    if not isinstance(registry, dict):
        return {"artifact_schema_version": ARTIFACT_SCHEMA_VERSION, "artifacts": {}}
    registry.setdefault("artifact_schema_version", ARTIFACT_SCHEMA_VERSION)
    registry.setdefault("artifacts", {})
    return cast(dict[str, Any], registry)


def _display_path(path: Path, project_dir: Path) -> str:
    try:
        return path.relative_to(project_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _runtime_versions(provider: str) -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "dbt_ml": _package_version("dbt-ml"),
        "polars": _package_version("polars"),
        "provider": provider,
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _serializable_options(options: TextOptions) -> dict[str, Any]:
    return {
        "analyzer": options["analyzer"],
        "lowercase": options["lowercase"],
        "token_pattern": options["token_pattern"],
        "ngram_range": list(options["ngram_range"]),
        "stop_words": sorted(options["stop_words"]),
        "min_df": options["min_df"],
        "max_df": options["max_df"],
        "max_features": options["max_features"],
        "binary": options["binary"],
        "n_features": options["n_features"],
        "alternate_sign": options["alternate_sign"],
    }


def _empty_feature_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "source_model": pl.String,
            "row_index": pl.Int64,
            "row_id": pl.String,
            "provider": pl.String,
            "term": pl.String,
            "term_index": pl.Int64,
            "count": pl.Int64,
            "tf": pl.Float64,
            "idf": pl.Float64,
            "tfidf": pl.Float64,
            "value": pl.Float64,
            "hash_bucket": pl.Int64,
        }
    )


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


def _hash_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(raw.encode(), digest_size=8).hexdigest()
