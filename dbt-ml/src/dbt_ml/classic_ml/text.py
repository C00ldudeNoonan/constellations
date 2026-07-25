"""Text feature providers (issue #190, Workstream B).

Analyzer and vectorizer machinery for the `features` task: tokenization and
n-gram analysis, count/TF-IDF/hashing vectorizer fitting, feature row shaping,
and the feature-artifact reader/writer.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, TypedDict, cast

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
    _write_artifact_payload,
    _write_metadata,
)
from .common import ClassicMLRun, _project_metrics, _source_rows, _training_input
from .contracts import Analyzer, FeatureProvider

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
    value: float | None,
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
        if not isinstance(integrity, dict) or "feature_count" not in integrity:
            raise IncompatibleClassicMLArtifactError(
                f"incompatible hashing artifact integrity at {path}: missing "
                "feature_count"
            )
        feature_count = integrity["feature_count"]
        if feature_count != metadata_options["n_features"]:
            raise IncompatibleClassicMLArtifactError(
                f"incompatible artifact integrity at {path}: feature_count does "
                "not match persisted n_features"
            )
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
    source_df = adapter.read_table(source_name)
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
