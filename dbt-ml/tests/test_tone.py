"""Deterministic document tone/sentiment transform (issue #216).

Scores are hand-computed on small in-memory token + lexicon frames — no spaCy,
no snapshots, mirroring the document-features tests.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from dbt_ml.adapters import parse_warehouse_config
from dbt_ml.text.tone import ToneOptions, tone_lexicon_fingerprint
from dbt_ml.text.transforms import document_tone as tone
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


# d1: "the growth is strong" — two positive terms.
# d2: "not weak" — one negative term negated by a preceding negator.
# d3: "quiet" — one token, no lexicon match.
_TOKENS = pl.DataFrame(
    {
        "document_id": ["d1", "d1", "d1", "d1", "d2", "d2", "d3"],
        "token_index": [0, 1, 2, 3, 0, 1, 0],
        "sentence_index": [0, 0, 0, 0, 0, 0, 0],
        "lemma": ["the", "growth", "be", "strong", "not", "weak", "quiet"],
        "publisher": ["Fed"] * 4 + ["SEC"] * 2 + ["BEA"],
        **_identity_columns(7),
    }
)

_LEXICON = pl.DataFrame(
    {
        "term": ["growth", "strong", "weak", "decline", "uncertain"],
        "category": ["positive", "positive", "negative", "negative", "uncertainty"],
        "weight": [1.0, 1.0, 1.0, 1.0, 1.0],
    }
)

# A documents spine that includes d4, which produces no tokens.
_DOCUMENTS = pl.DataFrame(
    {
        "document_id": ["d1", "d2", "d3", "d4"],
        "publisher": ["Fed", "SEC", "BEA", "Treasury"],
        "published_at": ["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"],
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
    lexicon: pl.DataFrame = _LEXICON,
    documents: pl.DataFrame | None = None,
) -> pl.DataFrame:
    merged: dict[str, object] = {
        "tokens": "tok",
        "lexicon": "lex",
        "emit": ["positive", "negative", "uncertainty"],
        **(options or {}),
    }
    deps = {"tok": tokens, "lex": lexicon}
    if documents is not None:
        merged.setdefault("documents", "doc")
        deps["doc"] = documents
    return tone.run(deps, _ctx(merged)).sort("document_id")


def _row(frame: pl.DataFrame, document_id: str) -> dict[str, object]:
    return frame.filter(pl.col("document_id") == document_id).to_dicts()[0]


# ── scoring ──────────────────────────────────────────────────────────────────


def test_polarity_scores_and_hits_are_hand_computed() -> None:
    result = _run()
    d1 = _row(result, "d1")
    assert d1["positive_hits"] == 2
    assert d1["positive_score"] == pytest.approx(2 / 4)  # two hits over four tokens
    assert d1["negative_hits"] == 0
    assert d1["negative_score"] == pytest.approx(0.0)
    assert d1["uncertainty_hits"] == 0
    assert d1["token_count"] == 4
    assert d1["matched_token_count"] == 2
    assert d1["coverage"] == pytest.approx(2 / 4)
    assert d1["status"] == "scored"


def test_negation_flips_a_matched_term() -> None:
    d2 = _row(_run(), "d2")
    # "weak" is negated by the preceding "not", so its negative contribution is
    # negative, but it is still counted as a hit.
    assert d2["negative_hits"] == 1
    assert d2["negative_score"] == pytest.approx(-1 / 2)


def test_negation_disabled_keeps_positive_contribution() -> None:
    d2 = _row(_run({"negation": False}), "d2")
    assert d2["negative_hits"] == 1
    assert d2["negative_score"] == pytest.approx(1 / 2)


def test_domain_signal_is_separate_from_polarity() -> None:
    tokens = pl.DataFrame(
        {
            "document_id": ["d9", "d9"],
            "token_index": [0, 1],
            "sentence_index": [0, 0],
            "lemma": ["strong", "uncertain"],
            "publisher": ["Fed", "Fed"],
            **_identity_columns(2),
        }
    )
    d9 = _row(_run(tokens=tokens), "d9")
    assert d9["positive_hits"] == 1
    assert d9["uncertainty_hits"] == 1
    assert d9["positive_score"] == pytest.approx(1 / 2)
    assert d9["uncertainty_score"] == pytest.approx(1 / 2)


def test_unmatched_document_scores_zero_with_null_free_counts() -> None:
    d3 = _row(_run(), "d3")
    assert d3["positive_hits"] == 0
    assert d3["matched_token_count"] == 0
    assert d3["coverage"] == pytest.approx(0.0)
    assert d3["positive_score"] == pytest.approx(0.0)
    assert d3["status"] == "scored"


def test_weights_are_applied() -> None:
    lexicon = _LEXICON.with_columns(
        pl.when(pl.col("term") == "growth").then(2.0).otherwise(pl.col("weight")).alias("weight")
    )
    d1 = _row(_run(lexicon=lexicon), "d1")
    assert d1["positive_hits"] == 2
    assert d1["positive_score"] == pytest.approx((2.0 + 1.0) / 4)


def test_insufficient_text_nulls_scores() -> None:
    result = _run({"min_tokens": 3})
    d2 = _row(result, "d2")  # only two tokens
    assert d2["status"] == "insufficient_text"
    assert d2["negative_score"] is None
    assert d2["coverage"] is None
    # counts are still reported
    assert d2["token_count"] == 2
    assert d2["negative_hits"] == 1


def test_include_fields_and_identity_pass_through() -> None:
    d1 = _row(_run({"include_fields": ["publisher"]}, documents=_DOCUMENTS), "d1")
    assert d1["publisher"] == "Fed"
    assert d1["nlp_model"] == "en_core_web_sm"
    assert d1["scorer"] == "lexicon"
    assert d1["scorer_version"] == "1"
    assert isinstance(d1["lexicon_version"], str) and d1["lexicon_version"]


def test_documents_spine_retains_zero_token_documents() -> None:
    # d4 produces no tokens; with a documents spine it still gets a row.
    result = _run(documents=_DOCUMENTS)
    d4 = _row(result, "d4")
    assert d4["token_count"] == 0
    assert d4["matched_token_count"] == 0
    assert d4["status"] == "insufficient_text"
    assert d4["positive_score"] is None
    assert d4["coverage"] is None
    # Without a spine, d4 is absent entirely.
    assert "d4" not in _run()["document_id"].to_list()


def test_include_fields_requires_documents_dependency() -> None:
    with pytest.raises(ValidationError, match="documents"):
        ToneOptions(tokens="tok", lexicon="lex", emit=("positive",), include_fields=("publisher",))


def test_empty_output_preserves_passthrough_dtype() -> None:
    documents = pl.DataFrame(
        schema={
            "document_id": pl.String(),
            "publisher": pl.String(),
            "published_at": pl.Date(),
        }
    )
    result = _run({"include_fields": ["publisher", "published_at"]}, documents=documents)
    assert result.is_empty()
    # A date passthrough stays a date even on an empty run.
    assert result.schema["published_at"] == pl.Date()


def test_non_finite_lexicon_weight_is_rejected() -> None:
    lexicon = pl.DataFrame(
        {"term": ["growth"], "category": ["positive"], "weight": [float("inf")]}
    )
    with pytest.raises(ValueError, match="finite"):
        _run(lexicon=lexicon)


def test_output_is_deterministic() -> None:
    assert _run().equals(_run())


# ── lexicon version ──────────────────────────────────────────────────────────


def test_lexicon_version_changes_when_lexicon_changes() -> None:
    baseline = _row(_run(), "d1")["lexicon_version"]
    edited = _LEXICON.with_columns(
        pl.when(pl.col("term") == "growth").then(3.0).otherwise(pl.col("weight")).alias("weight")
    )
    changed = _row(_run(lexicon=edited), "d1")["lexicon_version"]
    assert baseline != changed


def test_lexicon_fingerprint_is_order_and_duplicate_insensitive() -> None:
    rows = [
        {"term": "growth", "category": "positive", "weight": 1.0},
        {"term": "weak", "category": "negative", "weight": 1.0},
    ]
    reordered = [*reversed(rows), rows[0]]  # reordered + a duplicate
    assert tone_lexicon_fingerprint(rows) == tone_lexicon_fingerprint(reordered)


# ── empties and failures ─────────────────────────────────────────────────────


def test_empty_tokens_returns_typed_empty_frame() -> None:
    empty = _TOKENS.head(0)
    result = _run(tokens=empty)
    assert result.is_empty()
    assert "positive_score" in result.columns
    assert "lexicon_version" in result.columns


def test_language_mismatch_fails_actionably() -> None:
    tokens = _TOKENS.with_columns(
        pl.when(pl.col("document_id") == "d2")
        .then(pl.lit("fr"))
        .otherwise(pl.col("nlp_language"))
        .alias("nlp_language")
    )
    with pytest.raises(ValueError, match="language"):
        _run(tokens=tokens)


def test_conflicting_lexicon_weights_are_rejected() -> None:
    lexicon = pl.DataFrame(
        {
            "term": ["growth", "growth"],
            "category": ["positive", "positive"],
            "weight": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="conflicting weights"):
        _run(lexicon=lexicon)


def test_disagreeing_identity_is_rejected() -> None:
    tokens = _TOKENS.with_columns(
        pl.when((pl.col("document_id") == "d1") & (pl.col("token_index") == 0))
        .then(pl.lit("en_core_web_lg"))
        .otherwise(pl.col("nlp_model"))
        .alias("nlp_model")
    )
    with pytest.raises(ValueError, match="disagree"):
        _run(tokens=tokens)


# ── options ──────────────────────────────────────────────────────────────────


def test_declared_dependencies_are_tokens_and_lexicon() -> None:
    options = {"tokens": "tok", "lexicon": "lex", "emit": ["positive"]}
    assert tone.declared_dependencies(options) == ("tok", "lex")


def test_empty_emit_is_rejected() -> None:
    with pytest.raises(ValidationError, match="emit"):
        ToneOptions(tokens="tok", lexicon="lex", emit=())


def test_tokens_and_lexicon_must_differ() -> None:
    with pytest.raises(ValidationError, match="different"):
        ToneOptions(tokens="same", lexicon="same", emit=("positive",))


def test_emit_column_collision_with_reserved_is_rejected() -> None:
    with pytest.raises(ValidationError, match=r"collides|reserved|token"):
        ToneOptions(tokens="tok", lexicon="lex", emit=("token",), include_fields=("token_count",))


def test_economic_nlp_example_compiles_with_document_tone() -> None:
    from dbt_ml.compiler import validate_project_contract
    from dbt_ml.config import load_project

    project_dir = Path(__file__).resolve().parents[1] / "examples" / "economic_nlp"
    project, sources, models = load_project(project_dir)

    dag = validate_project_contract(project, sources, models, project_dir)

    order = dag.execution_order()
    assert order.index("document_tone") > order.index("document_tokens")
    assert order.index("document_tone") > order.index("tone_lexicon")

