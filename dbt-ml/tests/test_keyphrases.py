"""Deterministic keyphrase extraction from the NLP token child table (issue #219).

Scores and ranks are hand-computed on small in-memory token frames — no spaCy,
no snapshots, mirroring the document-features and tone tests.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from dbt_ml.adapters import parse_warehouse_config
from dbt_ml.text.keyphrases import KeyphraseOptions
from dbt_ml.text.transforms import extract_keyphrases as kp
from dbt_ml.transforms import TransformContext

_IDENTITY = {
    "nlp_provider": "spacy",
    "nlp_provider_version": "3.8.0",
    "nlp_model": "en_core_web_sm",
    "nlp_model_version": "3.8.0",
    "nlp_language": "en",
}


def _identity_columns(height: int) -> dict[str, list[str]]:
    return {key: [value] * height for key, value in _IDENTITY.items()}


# d1: "the growth is strong" — "growth" and "strong" are content unigrams;
#     "growth is" is invalid (boundary stop), "is strong" is invalid (boundary stop).
#     Bigram "growth strong"? — NOT consecutive: gap via "is". Wait, consecutive means
#     adjacent token indices. "the"(0) "growth"(1) "is"(2) "strong"(3). So bigrams are:
#     (0,1), (1,2), (2,3). "growth is" — "is" is stop → last boundary invalid.
#     "is strong" — "is" is stop → first boundary invalid.
#     Only valid bigram with stop_pos defaults: none (all span a stop at a boundary).
#     Valid unigrams: growth, strong (is/the are stop words).
#
# d2: "rate of return" — "rate"(0) "of"(1) "return"(2). "of" is stop but not in stop_pos
#     (stop_pos filters POS tags, not stop-word flag). However "of" has is_stop=True.
#     Bigram "rate of" → last token "of" is stop → invalid.
#     Bigram "of return" → first token "of" is stop → invalid.
#     Trigram "rate of return" → first "rate" OK, last "return" OK → valid 3-gram!
#     Valid unigrams: rate, return (of is stop).
#     Valid trigrams: "rate of return".
#
# d3: "strong growth" — repeated order, appears twice via d3a and d3b below.
_TOKENS = pl.DataFrame(
    {
        "document_id": [
            "d1", "d1", "d1", "d1",
            "d2", "d2", "d2",
            "d3", "d3",
        ],
        "token_index": [0, 1, 2, 3, 0, 1, 2, 0, 1],
        "sentence_index": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        "lemma": [
            "the", "growth", "be", "strong",
            "rate", "of", "return",
            "strong", "growth",
        ],
        "token_text": [
            "The", "growth", "is", "strong",
            "rate", "of", "return",
            "Strong", "growth",
        ],
        "pos": [
            "DET", "NOUN", "AUX", "ADJ",
            "NOUN", "ADP", "NOUN",
            "ADJ", "NOUN",
        ],
        "is_stop": [True, False, True, False, False, True, False, False, False],
        "is_alpha": [True, True, True, True, True, True, True, True, True],
        **_identity_columns(9),
    }
)


def _ctx(options: dict[str, object]) -> TransformContext:
    return TransformContext(
        project_dir=Path("."),
        profile_name="test",
        target_name="dev",
        warehouse=parse_warehouse_config(
            {"type": "duckdb", "path": "./test.duckdb", "schema": "main"}
        ),
        llm=None,
        options=options,
    )


def _run(
    options: dict[str, object] | None = None,
    *,
    tokens: pl.DataFrame = _TOKENS,
) -> pl.DataFrame:
    merged: dict[str, object] = {"tokens": "tok", **(options or {})}
    return kp.run({"tok": tokens}, _ctx(merged)).sort("document_id")


def _row(frame: pl.DataFrame, document_id: str) -> dict[str, object]:
    rows = frame.filter(pl.col("document_id") == document_id).to_dicts()
    assert rows, f"No row for document_id={document_id!r}"
    return rows[0]


def _rows(frame: pl.DataFrame, document_id: str) -> list[dict[str, object]]:
    return frame.filter(pl.col("document_id") == document_id).to_dicts()


# ── scoring and ranking ───────────────────────────────────────────────────────


def test_unigram_scores_and_ranks_are_hand_computed() -> None:
    # d1 candidates (unigrams): "growth", "strong" — each appears once.
    # total candidates = 2, so score = 1/2 = 0.5 for both.
    # Tie-break: "growth" < "strong" alphabetically → growth rank=1, strong rank=2.
    result = _run({"max_phrase_length": 1})
    rows = _rows(result, "d1")
    assert len(rows) == 2
    top = rows[0]
    assert top["phrase_lemma"] == "growth"
    assert top["rank"] == 1
    assert top["score"] == pytest.approx(0.5)
    second = rows[1]
    assert second["phrase_lemma"] == "strong"
    assert second["rank"] == 2
    assert second["score"] == pytest.approx(0.5)


def test_trigram_spanning_interior_stop_word_is_valid() -> None:
    # d2: "rate of return" — interior "of" is stop but boundary tokens are content words.
    result = _run()
    d2_rows = _rows(result, "d2")
    lemmas = {r["phrase_lemma"] for r in d2_rows}
    assert "rate of return" in lemmas


def test_top_k_limits_output() -> None:
    result = _run({"top_k": 1})
    for doc_id in _TOKENS["document_id"].unique().to_list():
        rows = _rows(result, doc_id)
        assert len(rows) <= 1


def test_min_score_filters_low_frequency_phrases() -> None:
    # With a high min_score, only phrases scoring >= threshold survive.
    result = _run({"min_score": 0.9})
    # All scores should be >= 0.9.
    assert (result["score"] >= 0.9).all()


def test_rank_is_one_indexed() -> None:
    result = _run()
    min_rank = result["rank"].min()
    assert min_rank == 1


def test_rank_is_contiguous_within_document() -> None:
    result = _run()
    for doc_id in result["document_id"].unique().to_list():
        ranks = sorted(_rows(result, doc_id)[i]["rank"] for i in range(len(_rows(result, doc_id))))
        assert ranks == list(range(1, len(ranks) + 1))


def test_score_is_between_zero_and_one() -> None:
    result = _run()
    assert (result["score"] >= 0.0).all()
    assert (result["score"] <= 1.0).all()


def test_output_is_deterministic() -> None:
    assert _run().equals(_run())


def test_phrase_id_is_stable() -> None:
    r1 = _run()
    r2 = _run()
    assert r1["phrase_id"].equals(r2["phrase_id"])


def test_phrase_ids_are_unique_within_run() -> None:
    result = _run()
    assert result["phrase_id"].n_unique() == result.height


# ── phrase boundaries ─────────────────────────────────────────────────────────


def test_stop_word_boundary_tokens_are_excluded_from_unigrams() -> None:
    # "the" and "be" are stop words; they must not appear as unigrams.
    result = _run({"max_phrase_length": 1})
    lemmas = result["phrase_lemma"].to_list()
    assert "the" not in lemmas
    assert "be" not in lemmas


def test_stop_pos_excludes_boundary_token_by_pos_tag() -> None:
    # "DET" is not in DEFAULT_STOP_POS, so without stop_pos filtering "the" (POS=DET,
    # is_stop=True) is excluded by the is_stop check. With stop_pos=("NOUN",) and
    # is_stop disabled on "the", "the" would still be excluded as a boundary NOUN.
    # Here we verify that stop_pos=("ADJ",) filters "strong" from being a phrase start.
    tokens = _TOKENS.with_columns(pl.col("is_stop").cast(pl.Boolean()) & False)
    result = _run(
        {"max_phrase_length": 1, "stop_pos": ["ADJ"]},
        tokens=tokens,
    )
    d1_lemmas = {r["phrase_lemma"] for r in _rows(result, "d1")}
    assert "strong" not in d1_lemmas


def test_min_phrase_length_filters_unigrams() -> None:
    result = _run({"min_phrase_length": 2})
    assert (result["phrase_length"] >= 2).all()


def test_max_phrase_length_limits_ngram_size() -> None:
    result = _run({"max_phrase_length": 1})
    assert (result["phrase_length"] == 1).all()


# ── phrase text ───────────────────────────────────────────────────────────────


def test_phrase_text_absent_by_default() -> None:
    result = _run()
    assert "phrase_text" not in result.columns


def test_phrase_text_opt_in() -> None:
    result = _run({"include_phrase_text": True, "max_phrase_length": 1})
    assert "phrase_text" in result.columns


def test_phrase_text_value_for_unigram() -> None:
    result = _run({"include_phrase_text": True, "max_phrase_length": 1})
    d1_rows = {r["phrase_lemma"]: r["phrase_text"] for r in _rows(result, "d1")}
    assert d1_rows["growth"] == "growth"
    assert d1_rows["strong"] == "strong"


def test_phrase_text_value_for_trigram() -> None:
    result = _run({"include_phrase_text": True})
    d2_rows = {r["phrase_lemma"]: r["phrase_text"] for r in _rows(result, "d2")}
    assert d2_rows.get("rate of return") == "rate of return"


# ── NLP identity ─────────────────────────────────────────────────────────────


def test_nlp_identity_columns_present() -> None:
    result = _run()
    identity_cols = (
        "nlp_provider", "nlp_provider_version", "nlp_model", "nlp_model_version", "nlp_language"
    )
    for col in identity_cols:
        assert col in result.columns
        assert result[col].null_count() == 0


def test_extractor_and_version_are_set() -> None:
    result = _run()
    assert (result["extractor"] == "ngram_freq").all()
    assert (result["extractor_version"] == "1").all()


def test_disagreeing_nlp_identity_is_rejected() -> None:
    tokens = _TOKENS.with_columns(
        pl.when((pl.col("document_id") == "d1") & (pl.col("token_index") == 0))
        .then(pl.lit("en_core_web_lg"))
        .otherwise(pl.col("nlp_model"))
        .alias("nlp_model")
    )
    with pytest.raises(ValueError, match="disagree"):
        _run(tokens=tokens)


# ── language validation ───────────────────────────────────────────────────────


def test_language_mismatch_fails_actionably() -> None:
    tokens = _TOKENS.with_columns(
        pl.when(pl.col("document_id") == "d2")
        .then(pl.lit("fr"))
        .otherwise(pl.col("nlp_language"))
        .alias("nlp_language")
    )
    with pytest.raises(ValueError, match="language"):
        _run(tokens=tokens)


def test_null_sentence_index_rejected_for_multi_token() -> None:
    # spaCy can emit null sentence_index when the sentencizer is disabled.
    # Multi-token extraction would then group all null-sentence tokens together
    # and form phrases across unknown boundaries — reject it early.
    tokens = _TOKENS.with_columns(
        pl.when(pl.col("document_id") == "d1")
        .then(None)
        .otherwise(pl.col("sentence_index"))
        .cast(pl.Int64())
        .alias("sentence_index")
    )
    with pytest.raises(ValueError, match="sentence"):
        _run({"max_phrase_length": 2}, tokens=tokens)


def test_null_sentence_index_allowed_for_unigrams() -> None:
    # Unigram extraction does not use sentence_index for phrase construction.
    tokens = _TOKENS.with_columns(pl.lit(None).cast(pl.Int64()).alias("sentence_index"))
    result = _run({"max_phrase_length": 1}, tokens=tokens)
    # sentence_index column is still null in the output, but no error raised.
    assert not result.is_empty()


# ── empty and edge cases ──────────────────────────────────────────────────────


def test_empty_tokens_returns_typed_empty_frame() -> None:
    empty = _TOKENS.head(0)
    result = _run(tokens=empty)
    assert result.is_empty()
    assert "phrase_id" in result.columns
    assert "rank" in result.columns
    assert "extractor_version" in result.columns


def test_all_stop_words_returns_empty() -> None:
    tokens = _TOKENS.with_columns(pl.lit(True).alias("is_stop"))
    result = _run(tokens=tokens)
    assert result.is_empty()


def test_empty_output_has_correct_schema() -> None:
    empty = _TOKENS.head(0)
    result = _run(tokens=empty)
    assert result.schema["rank"] == pl.Int64()
    assert result.schema["score"] == pl.Float64()
    assert result.schema["phrase_id"] == pl.String()


def test_empty_output_with_phrase_text_has_correct_schema() -> None:
    empty = _TOKENS.head(0)
    result = _run({"include_phrase_text": True}, tokens=empty)
    assert "phrase_text" in result.columns
    assert result.schema["phrase_text"] == pl.String()


def test_missing_token_text_field_raises_when_phrase_text_enabled() -> None:
    tokens = _TOKENS.drop("token_text")
    with pytest.raises(ValueError, match="missing"):
        _run({"include_phrase_text": True}, tokens=tokens)


# ── declared_dependencies ─────────────────────────────────────────────────────


def test_declared_dependencies_is_tokens_only() -> None:
    assert kp.declared_dependencies({"tokens": "my_tokens"}) == ("my_tokens",)


# ── options validation ────────────────────────────────────────────────────────


def test_max_less_than_min_is_rejected() -> None:
    with pytest.raises(ValidationError, match="max_phrase_length"):
        KeyphraseOptions(tokens="tok", min_phrase_length=3, max_phrase_length=2)


def test_empty_tokens_name_is_rejected() -> None:
    with pytest.raises(ValidationError):
        KeyphraseOptions(tokens="")


def test_duplicate_stop_pos_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unique"):
        KeyphraseOptions(tokens="tok", stop_pos=("PUNCT", "PUNCT"))


def test_top_k_zero_is_rejected() -> None:
    with pytest.raises(ValidationError):
        KeyphraseOptions(tokens="tok", top_k=0)


def test_min_score_above_one_is_rejected() -> None:
    with pytest.raises(ValidationError):
        KeyphraseOptions(tokens="tok", min_score=1.1)


# ── example project ───────────────────────────────────────────────────────────


def test_economic_nlp_example_compiles_with_extract_keyphrases() -> None:
    from dbt_ml.compiler import validate_project_contract
    from dbt_ml.config import load_project

    project_dir = Path(__file__).resolve().parents[1] / "examples" / "economic_nlp"
    project, sources, models = load_project(project_dir)
    dag = validate_project_contract(project, sources, models, project_dir)

    order = dag.execution_order()
    assert order.index("document_keyphrases") > order.index("document_tokens")
