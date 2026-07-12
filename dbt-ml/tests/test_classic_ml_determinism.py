"""Classic-ML determinism and feature semantics (issue #122).

Training must be invariant to warehouse row-return order, proportional
min_df/max_df must round with vectorizer conventions, and the hashing sign
bit must be independent of bucket selection. Artifacts built under the old
semantics are rejected, not silently reused.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from dbt_ml.classic_ml import (
    ARTIFACT_SCHEMA_VERSION,
    IncompatibleClassicMLArtifactError,
    _fit_naive_bayes,
    _fit_vectorizer,
    _hashed_feature_rows,
    _read_artifact,
    _select_terms,
    _source_rows,
    _text_options,
    _training_input,
)
from dbt_ml.config.model import MLConfig

DOCS = [
    ("d1", "the rocket exploded on the launch pad", "incident"),
    ("d2", "invoice paid in full", "billing"),
    ("d3", "rocket launch delayed by weather", "incident"),
    ("d4", "billing question about invoice totals", "billing"),
]


def _frame(order: list[int], with_ids: bool = True) -> pl.DataFrame:
    rows = [DOCS[i] for i in order]
    data: dict[str, Any] = {
        "text": [r[1] for r in rows],
        "label": [r[2] for r in rows],
    }
    if with_ids:
        data = {"document_id": [r[0] for r in rows], **data}
    return pl.DataFrame(data)


PERMUTATIONS = [[0, 1, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1]]


# ─── row-order invariance ────────────────────────────────────────────────────


def test_source_rows_canonical_order_by_document_id() -> None:
    results = [
        _source_rows(_frame(order), "text", "label") for order in PERMUTATIONS
    ]
    assert results[0] == results[1] == results[2]
    assert [r["row_id"] for r in results[0]] == ["d1", "d2", "d3", "d4"]
    assert [r["row_index"] for r in results[0]] == [0, 1, 2, 3]


def test_source_rows_canonical_order_without_identifier() -> None:
    results = [
        _source_rows(_frame(order, with_ids=False), "text", "label")
        for order in PERMUTATIONS
    ]
    assert results[0] == results[1] == results[2]


def test_training_hash_invariant_under_permutation() -> None:
    hashes = {
        _training_input(
            ["ref('tickets')"], _source_rows(_frame(order), "text", "label")
        )["content_hash"]
        for order in PERMUTATIONS
    }
    assert len(hashes) == 1


@pytest.mark.parametrize("provider", ["builtin.count", "builtin.tfidf"])
def test_vectorizer_payload_invariant_under_permutation(provider: str) -> None:
    options = _text_options({})
    payloads = [
        _fit_vectorizer(_source_rows(_frame(order), "text"), provider, options)
        for order in PERMUTATIONS
    ]
    assert payloads[0] == payloads[1] == payloads[2]


def test_naive_bayes_model_invariant_under_permutation() -> None:
    options = _text_options({})
    models = [
        _fit_naive_bayes(
            _source_rows(_frame(order), "text", "label"),
            "builtin.naive_bayes",
            options,
            {},
        )
        for order in PERMUTATIONS
    ]
    assert models[0] == models[1] == models[2]


def test_hashed_features_map_to_same_rows_under_permutation() -> None:
    options = _text_options({"n_features": 64})
    vectorizer = {
        "provider": "builtin.hashing",
        "vocabulary": [],
        "idf": {},
        "n_features": 64,
        "options": dict(options, stop_words=[], ngram_range=[1, 1]),
    }

    def rows_by_id(order: list[int]) -> dict[str, list[tuple[str, float]]]:
        rows = _source_rows(_frame(order), "text")
        tokens = [row["text"].split() for row in rows]
        features = _hashed_feature_rows(rows, tokens, vectorizer, "tickets")
        out: dict[str, list[tuple[str, float]]] = {}
        for feature in features:
            out.setdefault(feature["row_id"], []).append(
                (feature["term"], feature["value"])
            )
        return out

    assert rows_by_id(PERMUTATIONS[0]) == rows_by_id(PERMUTATIONS[1])


# ─── proportional min_df / max_df boundaries ─────────────────────────────────


def _freq(**counts: int) -> Counter[str]:
    return Counter(counts)


def test_fractional_min_df_uses_ceiling() -> None:
    # 0.5 of 3 docs -> threshold 2: a term in 1 doc is out, 2 docs is in
    options = _text_options({"min_df": 0.5})
    terms = _select_terms(_freq(rare=1, mid=2, common=3), 3, options)
    assert terms == ["common", "mid"]


def test_fractional_max_df_uses_floor() -> None:
    # 0.5 of 3 docs -> threshold 1: only terms in at most 1 doc survive
    options = _text_options({"max_df": 0.5})
    terms = _select_terms(_freq(rare=1, mid=2, common=3), 3, options)
    assert terms == ["rare"]


def test_fractional_bounds_exact_fraction_is_inclusive() -> None:
    # 0.5 of 4 docs -> min 2 and max 2 both admit a df of exactly 2
    freq = _freq(rare=1, half=2, common=4)
    assert _select_terms(freq, 4, _text_options({"min_df": 0.5})) == [
        "common",
        "half",
    ]
    assert _select_terms(freq, 4, _text_options({"max_df": 0.5})) == [
        "half",
        "rare",
    ]


def test_integer_thresholds_unchanged() -> None:
    freq = _freq(rare=1, mid=2, common=3)
    assert _select_terms(freq, 3, _text_options({"min_df": 2})) == ["common", "mid"]
    assert _select_terms(freq, 3, _text_options({"max_df": 2})) == ["mid", "rare"]


def test_empty_corpus_selects_nothing() -> None:
    assert _select_terms(Counter(), 0, _text_options({"min_df": 0.5})) == []


# ─── hashing sign independence ───────────────────────────────────────────────


def _signs_and_buckets(n_features: int, tokens: list[str]) -> list[tuple[int, int]]:
    options = _text_options({"n_features": n_features})
    vectorizer = {
        "provider": "builtin.hashing",
        "vocabulary": [],
        "idf": {},
        "n_features": n_features,
        "options": dict(options, stop_words=[], ngram_range=[1, 1]),
    }
    pairs: list[tuple[int, int]] = []
    for token in tokens:
        rows = [{"row_index": 0, "row_id": "r", "text": token}]
        features = _hashed_feature_rows(rows, [[token]], vectorizer, "src")
        (feature,) = features
        pairs.append((feature["hash_bucket"], 1 if feature["value"] > 0 else -1))
    return pairs


def test_hashing_sign_independent_of_bucket_parity() -> None:
    tokens = [f"token{i}" for i in range(400)]
    pairs = _signs_and_buckets(64, tokens)  # even n_features
    # with sign == f(bucket parity), every bucket has exactly one sign;
    # an independent sign bit puts both signs in some buckets
    by_parity: dict[int, set[int]] = {0: set(), 1: set()}
    for bucket, sign in pairs:
        by_parity[bucket % 2].add(sign)
    assert by_parity[0] == {1, -1}
    assert by_parity[1] == {1, -1}


def test_hashing_is_deterministic() -> None:
    tokens = [f"token{i}" for i in range(50)]
    assert _signs_and_buckets(64, tokens) == _signs_and_buckets(64, tokens)


def test_hashing_collisions_can_cancel() -> None:
    # alternate_sign exists so colliding tokens with opposite signs cancel
    # rather than compound; find such a pair in a tiny bucket space and
    # check the summed value.
    options = _text_options({"n_features": 2})
    vectorizer = {
        "provider": "builtin.hashing",
        "vocabulary": [],
        "idf": {},
        "n_features": 2,
        "options": dict(options, stop_words=[], ngram_range=[1, 1]),
    }
    tokens = [f"t{i}" for i in range(64)]
    pairs = dict(zip(tokens, _signs_and_buckets(2, tokens), strict=True))
    opposite = [
        (a, b)
        for a in tokens
        for b in tokens
        if a < b and pairs[a][0] == pairs[b][0] and pairs[a][1] == -pairs[b][1]
    ]
    assert opposite, "expected opposite-sign collisions in a 2-bucket space"
    a, b = opposite[0]
    rows = [{"row_index": 0, "row_id": "r", "text": f"{a} {b}"}]
    (feature,) = _hashed_feature_rows(rows, [[a, b]], vectorizer, "src")
    assert feature["value"] == 0.0


# ─── artifact compatibility ──────────────────────────────────────────────────


def test_v1_artifacts_rejected_with_refit_hint(tmp_path: Path) -> None:
    assert ARTIFACT_SCHEMA_VERSION == 2
    (tmp_path / "metadata.json").write_text(
        json.dumps({"artifact_schema_version": 1, "artifact_type": "classic_ml"})
    )
    ml = MLConfig(task="features", mode="predict", text_field="text")
    with pytest.raises(IncompatibleClassicMLArtifactError, match="fit_transform"):
        _read_artifact(tmp_path, "builtin.tfidf", ml)
