from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import sys
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Literal, TypedDict, cast
from uuid import uuid4

import polars as pl

from .adapters import WarehouseAdapter
from .config.model import MLConfig, ModelConfig
from .config.project import ProjectConfig
from .dag import parse_ref
from .hashing import HASH_DIGEST_SIZE
from .ml_contracts import (
    ExecutableMLContract,
    MLContractError,
    validate_ml_contract,
    validate_persisted_ml_options,
)
from .optional_dependencies import import_optional_dependency
from .versioning import compute_code_version

# v2 (issue #122): canonical training-row order, vectorizer-convention
# min_df/max_df rounding, and an independent hashing sign bit — features and
# hashes from v1 artifacts are not comparable, so v1 artifacts are rejected
# with a refit hint rather than silently reused.
ARTIFACT_SCHEMA_VERSION = 2
ARTIFACT_REGISTRY_FILENAME = "registry.json"
_TOKEN_RE = re.compile(r"\w+")
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
    # Companion tables materialized as `<model>__<key>` alongside the primary
    # table (e.g. topic_model emits `topics`; cluster emits `representative_docs`).
    secondary_tables: dict[str, pl.DataFrame] = field(default_factory=dict)
    _publication: ClassicMLArtifactPublication | None = field(default=None, repr=False)

    def publish_artifact(self) -> None:
        if self._publication is not None:
            self._publication.publish()

    def discard_staged_artifact(self) -> None:
        if self._publication is not None:
            self._publication.discard()


@dataclass
class ClassicMLArtifactPublication:
    final_path: Path
    staged_path: Path
    registry_path: Path
    model_name: str
    registry_entry: dict[str, Any]
    _finished: bool = field(default=False, init=False, repr=False)

    def publish(self) -> None:
        if self._finished:
            return
        _publish_staged_artifact(self)
        self._finished = True

    def discard(self) -> None:
        if self._finished:
            return
        _remove_path(self.staged_path)
        self._finished = True


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


def _run_features(
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

    provider = cast(FeatureProvider, contract.provider)
    options = _text_options(contract.options)
    artifact_path = contract.artifact_path
    if ml.mode in {"predict", "load_pretrained"}:
        metadata, vectorizer = _read_artifact(artifact_path, provider, ml)
        options = _text_options(vectorizer["options"])

    source_name = parse_ref(model.depends_on[0])
    source_df = adapter.query_df(f"SELECT * FROM {adapter.table_ref(source_name)}")
    if ml.text_field not in source_df.columns:
        raise ValueError(
            f"ML model '{model.name}' text_field '{ml.text_field}' "
            f"is not present in '{source_name}'."
        )

    rows = _source_rows(source_df, ml.text_field)
    training_input = _training_input(model.depends_on, rows)
    code_version = compute_code_version(
        extraction=None,
        transform=None,
        ml=ml,
        project_dir=project_dir,
    )

    if ml.mode in {"fit_transform", "fit"}:
        vectorizer = _fit_vectorizer(rows, provider, options, contract.options)

    doc_tokens = [_analyze(row["text"], options) for row in rows]
    features = _feature_rows(rows, doc_tokens, vectorizer, source_name)
    all_metrics = {
        "row_count": len(rows),
        "vocabulary_size": len(vectorizer["vocabulary"]),
        "feature_rows": len(features),
    }
    if provider == "builtin.hashing":
        all_metrics["hash_buckets"] = vectorizer["n_features"]

    if ml.mode in {"fit_transform", "fit"}:
        metadata = _metadata(
            model=model,
            ml=ml,
            provider=provider,
            training_input=training_input,
            vectorizer=vectorizer,
            provider_options=contract.options,
            metrics=all_metrics,
            code_version=code_version,
        )
        staged_path = _new_artifact_staging_path(artifact_path)
        try:
            _write_artifact(staged_path, metadata, vectorizer)
            metadata, _ = _read_artifact(staged_path, provider, ml)
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
    else:
        publication = None

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
        metrics=_project_metrics(ml, all_metrics),
        artifact_metadata=metadata,
        _publication=publication,
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
        "content_hash": hashlib.blake2b(
            raw.encode(), digest_size=HASH_DIGEST_SIZE
        ).hexdigest(),
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
    provider_options: dict[str, Any],
) -> dict[str, Any]:
    if provider == "builtin.hashing":
        return _fit_hashing_vectorizer(provider, options, provider_options)

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
        "options": dict(provider_options),
    }


def _fit_hashing_vectorizer(
    provider: FeatureProvider,
    options: TextOptions,
    provider_options: dict[str, Any],
) -> dict[str, Any]:
    n_features = options["n_features"]
    if n_features <= 0:
        raise ValueError("ml.options.n_features must be positive for builtin.hashing.")
    return {
        "provider": provider,
        "vocabulary": [],
        "idf": {},
        "n_features": n_features,
        "options": dict(provider_options),
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
        pattern = re.compile(options["token_pattern"])
        matches = list(pattern.finditer(text))
        if any(match.start() == match.end() for match in matches):
            raise ValueError("ml.options.token_pattern produced an empty match")
        group = 1 if pattern.groups == 1 else 0
        tokens = [match.group(group) for match in matches]
        if any(not token for token in tokens):
            raise ValueError("ml.options.token_pattern produced an empty token")
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


def _project_metrics(ml: MLConfig, metrics: dict[str, Any]) -> dict[str, Any]:
    if not ml.metrics:
        return dict(metrics)
    return {name: metrics.get(name) for name in ml.metrics}


def _metadata(
    *,
    model: ModelConfig,
    ml: MLConfig,
    provider: FeatureProvider,
    training_input: dict[str, Any],
    vectorizer: dict[str, Any],
    provider_options: dict[str, Any],
    metrics: dict[str, Any],
    code_version: str,
) -> dict[str, Any]:
    files = ["metadata.json"]
    if provider != "builtin.hashing":
        files.append("vocabulary.json")
    metadata: dict[str, Any] = {
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
                "options": provider_options,
            }
        ),
        "runtime": _runtime_versions(provider),
        "training_input": training_input,
        "integrity": {
            "feature_count": vectorizer["n_features"],
        },
        "files": files,
        "options": provider_options,
        "vocabulary_hash": _hash_json(vectorizer["vocabulary"]),
        "idf_hash": _hash_json(vectorizer["idf"]),
    }
    if ml.artifact.include_metrics:
        metadata["metrics"] = _project_metrics(ml, metrics)
    return metadata


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
    expected_files = (
        ("metadata.json",)
        if provider == "builtin.hashing"
        else ("metadata.json", "vocabulary.json")
    )
    _validate_metadata(metadata, path, provider, ml, expected_files=expected_files)
    metadata_options = _validated_persisted_options(
        provider,
        metadata.get("options"),
        path,
        surface="metadata",
    )
    if provider == "builtin.hashing":
        integrity = metadata.get("integrity")
        if isinstance(integrity, dict) and "feature_count" in integrity:
            feature_count = integrity["feature_count"]
            if feature_count != metadata_options["n_features"]:
                raise IncompatibleClassicMLArtifactError(
                    f"incompatible artifact integrity at {path}: feature_count does "
                    "not match persisted n_features"
                )
        else:
            legacy_metrics = metadata.get("metrics")
            if not isinstance(legacy_metrics, dict):
                raise IncompatibleClassicMLArtifactError(
                    f"incompatible hashing artifact integrity at {path}: missing "
                    "feature_count"
                )
            feature_count = legacy_metrics.get("feature_count")
        if isinstance(feature_count, bool) or not isinstance(feature_count, int):
            raise IncompatibleClassicMLArtifactError(
                f"incompatible hashing artifact integrity at {path}: feature_count "
                "must be an integer"
            )
        if feature_count != metadata_options["n_features"]:
            raise IncompatibleClassicMLArtifactError(
                f"incompatible hashing artifact integrity at {path}: expected "
                f"{metadata_options['n_features']} features, found {feature_count!r}"
            )
        vectorizer = {
            "provider": provider,
            "vocabulary": [],
            "idf": {},
            "n_features": metadata_options["n_features"],
            "options": metadata_options,
        }
        _validate_artifact_payload(metadata, path, vectorizer)
        return metadata, vectorizer

    vocab_path = path / "vocabulary.json"
    _validate_artifact_payload(metadata, path, {})
    vocab_payload = _read_artifact_json(vocab_path, path, "vocabulary")
    try:
        if vocab_payload.get("provider") != provider:
            raise IncompatibleClassicMLArtifactError(
                f"incompatible vocabulary provider at {path}: expected {provider}, "
                f"found {vocab_payload.get('provider')!r}"
            )
        vocabulary = vocab_payload["terms"]
        idf_payload = vocab_payload["idf"]
        if (
            not isinstance(vocabulary, list)
            or any(not isinstance(term, str) for term in vocabulary)
            or len(vocabulary) != len(set(vocabulary))
        ):
            raise IncompatibleClassicMLArtifactError(
                f"incompatible vocabulary terms at {path}: expected unique strings"
            )
        if not isinstance(idf_payload, dict):
            raise IncompatibleClassicMLArtifactError(
                f"incompatible vocabulary idf values at {path}: expected an object"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int | float)
            for value in idf_payload.values()
        ):
            raise IncompatibleClassicMLArtifactError(
                f"incompatible vocabulary idf values at {path}: expected numbers"
            )
        idf = {str(key): float(value) for key, value in idf_payload.items()}
        if any(not math.isfinite(value) for value in idf.values()):
            raise IncompatibleClassicMLArtifactError(
                f"incompatible vocabulary idf values at {path}: values must be finite"
            )
        if provider == "builtin.count" and idf:
            raise IncompatibleClassicMLArtifactError(
                f"incompatible count-vector artifact at {path}: idf must be empty"
            )
        if provider == "builtin.tfidf" and set(idf) != set(vocabulary):
            raise IncompatibleClassicMLArtifactError(
                f"incompatible TF-IDF artifact at {path}: idf keys must match terms"
            )
        payload_options = _validated_persisted_options(
            provider,
            vocab_payload.get("options"),
            path,
            surface="vocabulary payload",
        )
    except ClassicMLArtifactError:
        raise
    except (KeyError, TypeError, ValueError) as e:
        raise IncompatibleClassicMLArtifactError(
            f"malformed vocabulary payload at {path}: {e}"
        ) from e
    if payload_options != metadata_options:
        raise IncompatibleClassicMLArtifactError(
            f"incompatible vocabulary options at {path}: metadata and payload differ"
        )
    vectorizer = {
        "provider": provider,
        "vocabulary": vocabulary,
        "idf": idf,
        "n_features": len(vocabulary),
        "options": payload_options,
    }
    integrity = metadata.get("integrity")
    if integrity is not None:
        if not isinstance(integrity, dict):
            raise IncompatibleClassicMLArtifactError(
                f"incompatible artifact integrity at {path}: expected an object"
            )
        feature_count = integrity.get("feature_count")
        if (
            isinstance(feature_count, bool)
            or not isinstance(feature_count, int)
            or feature_count != len(vocabulary)
        ):
            raise IncompatibleClassicMLArtifactError(
                f"incompatible artifact integrity at {path}: feature_count mismatch"
            )
    return metadata, vectorizer


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
    if isinstance(metadata_options_payload, dict) and "alpha" not in metadata_options_payload:
        legacy_classifier_options = metadata.get("classifier_options")
        if isinstance(legacy_classifier_options, dict) and "alpha" in legacy_classifier_options:
            metadata_options_payload = {
                **metadata_options_payload,
                "alpha": legacy_classifier_options["alpha"],
            }
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


def _validated_persisted_options(
    provider: FeatureProvider | ClassifierProvider,
    options: object,
    path: Path,
    *,
    surface: str,
) -> dict[str, Any]:
    try:
        return validate_persisted_ml_options(provider, options)
    except MLContractError as e:
        raise IncompatibleClassicMLArtifactError(
            f"incompatible {surface} at {path}: {e}"
        ) from e


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
            f"missing artifact metadata at {metadata_path}; run fit/fit_transform or "
            "supply a dbt-ml-native artifact first"
        )
    return _read_artifact_json(metadata_path, path, "metadata")


def _read_artifact_json(
    file_path: Path,
    artifact_path: Path,
    label: str,
) -> dict[str, Any]:
    if not file_path.exists():
        raise MissingClassicMLArtifactError(
            f"missing artifact payload '{file_path.name}' at {artifact_path}; run "
            "fit/fit_transform or supply a dbt-ml-native artifact first"
        )
    if file_path.is_symlink() or not file_path.is_file():
        raise IncompatibleClassicMLArtifactError(
            f"incompatible {label} at {artifact_path}: expected a regular, "
            "non-symlink file"
        )
    try:
        payload = json.loads(file_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as e:
        raise IncompatibleClassicMLArtifactError(
            f"malformed {label} JSON at {artifact_path}: {e}"
        ) from e
    if not isinstance(payload, dict):
        raise IncompatibleClassicMLArtifactError(
            f"malformed {label} JSON at {artifact_path}: expected an object"
        )
    return cast(dict[str, Any], payload)


def _validate_metadata(
    metadata: dict[str, Any],
    path: Path,
    provider: str,
    ml: MLConfig,
    *,
    expected_files: tuple[str, ...],
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
    if metadata.get("mode") not in {"fit", "fit_transform", "load_pretrained"}:
        raise IncompatibleClassicMLArtifactError(
            f"incompatible artifact mode at {path}: expected a fitted or pretrained "
            "artifact, "
            f"found {metadata.get('mode')!r}"
        )
    files = metadata.get("files")
    if files != list(expected_files):
        raise IncompatibleClassicMLArtifactError(
            f"incompatible artifact file contract at {path}: expected "
            f"{list(expected_files)!r}, found {files!r}"
        )
    runtime = metadata.get("runtime")
    required_runtime_fields = ("python", "dbt_ml", "polars", "provider")
    if not isinstance(runtime, dict) or any(
        not isinstance(runtime.get(field), str) or not runtime[field]
        for field in required_runtime_fields
    ):
        raise IncompatibleClassicMLArtifactError(
            f"incompatible artifact runtime contract at {path}: expected non-empty "
            f"fields {list(required_runtime_fields)!r}"
        )
    if runtime["provider"] != provider:
        raise IncompatibleClassicMLArtifactError(
            f"incompatible artifact runtime provider at {path}: expected {provider}, "
            f"found {runtime['provider']!r}"
        )
    if not isinstance(metadata.get("options"), dict):
        raise IncompatibleClassicMLArtifactError(
            f"incompatible artifact options at {path}: expected an object"
        )
    if not isinstance(metadata.get("artifact_files_hash"), str):
        raise IncompatibleClassicMLArtifactError(
            f"incompatible artifact file hash at {path}: expected a string"
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
    try:
        actual_hash = _artifact_files_hash(path, payload_files, vectorizer)
    except ClassicMLArtifactError:
        raise
    except OSError as e:
        raise IncompatibleClassicMLArtifactError(
            f"could not validate artifact payload at {path}: {e}"
        ) from e
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
    h = hashlib.blake2b(digest_size=HASH_DIGEST_SIZE)
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


def _new_artifact_staging_path(artifact_path: Path) -> Path:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        mkdtemp(
            prefix=f".{artifact_path.name}.staging-",
            dir=artifact_path.parent,
        )
    )


def _recover_artifact_publications(
    project: ProjectConfig,
    project_dir: Path,
) -> None:
    registry_dir = project_dir / project.target_path / "artifacts"
    if not registry_dir.exists():
        return
    registry_path = registry_dir / ARTIFACT_REGISTRY_FILENAME
    lock_path = registry_path.with_name(f".{registry_path.name}.lock")
    with _exclusive_file_lock(lock_path):
        _recover_pending_publications(registry_path)


def _artifact_publication(
    *,
    project: ProjectConfig,
    project_dir: Path,
    model: ModelConfig,
    artifact_path: Path,
    staged_path: Path,
    metadata: dict[str, Any],
) -> ClassicMLArtifactPublication:
    registry_dir = project_dir / project.target_path / "artifacts"
    registry_dir.mkdir(parents=True, exist_ok=True)
    registry_path = registry_dir / ARTIFACT_REGISTRY_FILENAME
    entry = {
        "model_name": model.name,
        "artifact_path": _display_path(artifact_path, project_dir),
        "artifact_version": metadata["artifact_version"],
        "provider": metadata["provider"],
        "task": metadata["task"],
        "code_version": metadata["code_version"],
        "config_hash": metadata["config_hash"],
        "artifact_files_hash": metadata["artifact_files_hash"],
        "training_input": metadata["training_input"],
    }
    if "metrics" in metadata:
        entry["metrics"] = metadata["metrics"]
    return ClassicMLArtifactPublication(
        final_path=artifact_path,
        staged_path=staged_path,
        registry_path=registry_path,
        model_name=model.name,
        registry_entry=entry,
    )


def _publish_staged_artifact(publication: ClassicMLArtifactPublication) -> None:
    final_path = publication.final_path
    staged_path = publication.staged_path
    registry_path = publication.registry_path
    if not staged_path.is_dir() or staged_path.is_symlink():
        raise ClassicMLArtifactError(
            f"staged artifact is missing or invalid at {staged_path}"
        )
    if final_path.is_symlink() or (final_path.exists() and not final_path.is_dir()):
        raise ClassicMLArtifactError(
            f"artifact path is not a regular directory: {final_path}"
        )

    lock_path = registry_path.with_name(f".{registry_path.name}.lock")
    with _exclusive_file_lock(lock_path):
        _recover_pending_publications(registry_path)
        registry_before = _read_artifact_registry(registry_path)
        publication_id = uuid4().hex
        backup_path = final_path.with_name(
            f".{final_path.name}.backup-{publication_id}"
        )
        journal_path = registry_path.with_name(
            f".artifact-publication-{publication_id}.json"
        )
        journal: dict[str, Any] = {
            "publication_id": publication_id,
            "model_name": publication.model_name,
            "artifact_version": publication.registry_entry["artifact_version"],
            "final_path": str(final_path),
            "staged_path": str(staged_path),
            "backup_path": str(backup_path),
            "prior_artifact_exists": final_path.exists(),
            "registry_before": registry_before,
        }
        _atomic_write_json(journal_path, journal)

        try:
            if final_path.exists():
                os.replace(final_path, backup_path)
            os.replace(staged_path, final_path)

            registry = deepcopy(registry_before)
            artifacts = registry.setdefault("artifacts", {})
            if not isinstance(artifacts, dict):
                raise ClassicMLArtifactError(
                    f"malformed artifact registry at {registry_path}: 'artifacts' must be an object"
                )
            artifacts[publication.model_name] = publication.registry_entry
            _publish_registry(registry_path, registry)
        except BaseException as error:
            try:
                _rollback_publication(journal_path, journal, registry_path)
            except BaseException as rollback_error:
                error.add_note(
                    f"Failed to roll back artifact publication: {rollback_error}"
                )
            raise
        else:
            _cleanup_committed_publication(journal_path, journal)


def _publish_registry(path: Path, registry: dict[str, Any]) -> None:
    _atomic_write_json(path, registry)


def _recover_pending_publications(registry_path: Path) -> None:
    for journal_path in sorted(
        registry_path.parent.glob(".artifact-publication-*.json")
    ):
        journal = _read_json_object(journal_path, "artifact publication journal")
        model_name = journal.get("model_name")
        artifact_version = journal.get("artifact_version")
        final_path = Path(str(journal.get("final_path", "")))
        registry = _read_artifact_registry(registry_path)
        artifacts = registry.get("artifacts")
        entry = artifacts.get(model_name) if isinstance(artifacts, dict) else None
        committed = (
            isinstance(entry, dict)
            and entry.get("artifact_version") == artifact_version
            and _artifact_version_at(final_path) == artifact_version
        )
        if committed:
            _cleanup_committed_publication(journal_path, journal)
        else:
            _rollback_publication(journal_path, journal, registry_path)


def _rollback_publication(
    journal_path: Path,
    journal: dict[str, Any],
    registry_path: Path,
) -> None:
    final_path, staged_path, backup_path = _validated_journal_paths(
        journal_path, journal
    )
    artifact_version = journal["artifact_version"]
    prior_exists = bool(journal["prior_artifact_exists"])
    registry_before = journal.get("registry_before")
    if not isinstance(registry_before, dict):
        raise ClassicMLArtifactError(
            f"malformed artifact publication journal at {journal_path}"
        )

    if backup_path.exists():
        _remove_path(final_path)
        os.replace(backup_path, final_path)
    elif not prior_exists and _artifact_version_at(final_path) == artifact_version:
        _remove_path(final_path)

    _atomic_write_json(registry_path, registry_before)
    _remove_path(staged_path)
    journal_path.unlink(missing_ok=True)


def _cleanup_committed_publication(
    journal_path: Path,
    journal: dict[str, Any],
) -> None:
    try:
        _, staged_path, backup_path = _validated_journal_paths(journal_path, journal)
        _remove_path(backup_path)
        _remove_path(staged_path)
        journal_path.unlink(missing_ok=True)
    except OSError as error:
        raise ClassicMLArtifactError(
            "Committed artifact publication cleanup remains pending at "
            f"{journal_path}; retry before publishing another artifact"
        ) from error


def _validated_journal_paths(
    journal_path: Path,
    journal: dict[str, Any],
) -> tuple[Path, Path, Path]:
    publication_id = journal.get("publication_id")
    final_raw = journal.get("final_path")
    staged_raw = journal.get("staged_path")
    backup_raw = journal.get("backup_path")
    if not all(
        isinstance(value, str) and value
        for value in (publication_id, final_raw, staged_raw, backup_raw)
    ):
        raise ClassicMLArtifactError(
            f"malformed artifact publication journal at {journal_path}"
        )
    assert isinstance(publication_id, str)
    assert isinstance(final_raw, str)
    assert isinstance(staged_raw, str)
    assert isinstance(backup_raw, str)
    final_path = Path(final_raw)
    staged_path = Path(staged_raw)
    backup_path = Path(backup_raw)
    valid = (
        journal_path.name == f".artifact-publication-{publication_id}.json"
        and final_path.is_absolute()
        and staged_path.is_absolute()
        and backup_path.is_absolute()
        and staged_path.parent == final_path.parent
        and backup_path.parent == final_path.parent
        and staged_path.name.startswith(f".{final_path.name}.staging-")
        and backup_path.name == f".{final_path.name}.backup-{publication_id}"
    )
    if not valid:
        raise ClassicMLArtifactError(
            f"malformed artifact publication journal paths at {journal_path}"
        )
    return final_path, staged_path, backup_path


def _artifact_version_at(path: Path) -> str | None:
    try:
        metadata = _read_metadata(path)
    except (ClassicMLArtifactError, OSError, UnicodeError, json.JSONDecodeError):
        return None
    version = metadata.get("artifact_version")
    return version if isinstance(version, str) else None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        module = importlib.import_module("msvcrt" if os.name == "nt" else "fcntl")
        if os.name == "nt":
            module.locking(handle.fileno(), module.LK_LOCK, 1)
        else:
            module.flock(handle.fileno(), module.LOCK_EX)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                module.locking(handle.fileno(), module.LK_UNLCK, 1)
            else:
                module.flock(handle.fileno(), module.LOCK_UN)


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def _read_artifact_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"artifact_schema_version": ARTIFACT_SCHEMA_VERSION, "artifacts": {}}
    registry = _read_json_object(path, "artifact registry")
    registry.setdefault("artifact_schema_version", ARTIFACT_SCHEMA_VERSION)
    artifacts = registry.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ClassicMLArtifactError(
            f"malformed artifact registry at {path}: 'artifacts' must be an object"
        )
    return registry


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as e:
        raise ClassicMLArtifactError(f"malformed {label} at {path}: {e}") from e
    if not isinstance(payload, dict):
        raise ClassicMLArtifactError(f"malformed {label} at {path}: expected an object")
    return cast(dict[str, Any], payload)


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
    return hashlib.blake2b(raw.encode(), digest_size=HASH_DIGEST_SIZE).hexdigest()


# ---------------------------------------------------------------------------
# Unsupervised matrix tasks: clustering + topic modeling (issue #42)
#
# These tasks consume a document-feature matrix (pivoted from a `features`
# model or read from a dense embedding column) instead of raw text. One model
# emits its primary per-document table plus companion tables materialized as
# `<model>__topics` / `<model>__representative_docs`.
# ---------------------------------------------------------------------------

_ML_FEATURE = "Clustering and topic modeling"


def _numpy() -> Any:
    return import_optional_dependency("numpy", extra="ml", feature=_ML_FEATURE)


def _sklearn(module: str) -> Any:
    return import_optional_dependency(module, extra="ml", feature=_ML_FEATURE)


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


@dataclass
class _MatrixDoc:
    row_index: int
    row_id: str
    document_id: Any
    source_path: Any


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
