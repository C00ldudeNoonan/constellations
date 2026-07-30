"""Vectorized deterministic keyphrase extraction over the NLP token child table.

N-gram candidates are formed from contiguous lemma sequences within each
sentence. Boundary tokens (first and last of a candidate) must not be stop
words and must not carry a POS tag in the configured ``stop_pos`` set. Interior
tokens are unrestricted so that phrases like "rate of return" are valid 3-grams.

Score is normalized TF: occurrence count / total candidate n-gram count in the
document. Ranking is score desc, phrase_lemma asc — fully deterministic. Phrase
IDs are stable hashes of (document_id, phrase_lemma).
"""
from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any

import polars as pl

from ...hashing import canonical_fingerprint
from ...transforms import IncrementalContract, TransformContext
from ..keyphrases import (
    KEYPHRASE_DOMAIN,
    KEYPHRASE_EXTRACTOR_NAME,
    KEYPHRASE_EXTRACTOR_VERSION,
    KeyphraseOptions,
)

TOKEN_IDENTITY_COLUMNS: tuple[str, ...] = (
    "nlp_provider",
    "nlp_provider_version",
    "nlp_model",
    "nlp_model_version",
    "nlp_language",
)

_COUNT_DTYPE = pl.Int64()
_SCORE_DTYPE = pl.Float64()


def validate_keyphrase_options(options: Mapping[str, Any]) -> None:
    KeyphraseOptions.model_validate(options)


def declared_keyphrase_dependencies(options: Mapping[str, Any]) -> tuple[str, ...]:
    return KeyphraseOptions.model_validate(options).declared_dependencies()


def keyphrase_incremental_contract(options: Mapping[str, Any]) -> IncrementalContract:
    parsed = KeyphraseOptions.model_validate(options)
    return IncrementalContract(
        parent_key="document_id",
        child_key="phrase_id",
        parent_source_key=parsed.document_id_field,
    )


def run_keyphrases(deps: dict[str, pl.DataFrame], ctx: TransformContext) -> pl.DataFrame:
    options = KeyphraseOptions.model_validate(ctx.options)
    tokens = _get_tokens(deps, options)
    _require_columns(tokens, _required_token_columns(options), options.tokens)
    if tokens.is_empty():
        return _empty_output(options)

    _reject_unsupported_language(tokens, options)
    if options.max_phrase_length > 1:
        _reject_null_sentence_index(tokens, options)

    candidates = _build_candidates(tokens, options)
    if candidates.is_empty():
        return _empty_output(options)

    scored = _score_candidates(candidates, options)
    ranked = _rank_and_filter(scored, options)
    if ranked.is_empty():
        return _empty_output(options)

    identity = _document_identity(tokens, options)
    result = ranked.join(identity, on="document_id", how="left")
    return _finalize(result, options)


# ── token access ─────────────────────────────────────────────────────────────


def _get_tokens(deps: dict[str, pl.DataFrame], options: KeyphraseOptions) -> pl.DataFrame:
    expected = set(options.declared_dependencies())
    if set(deps) != expected:
        raise ValueError(
            f"Keyphrase extraction expects dependencies named by the `tokens` option "
            f"({sorted(expected)}); got: {sorted(deps)}"
        )
    return deps[options.tokens]


def _required_token_columns(options: KeyphraseOptions) -> tuple[str, ...]:
    cols = [
        options.document_id_field,
        options.token_index_field,
        options.sentence_index_field,
        options.lemma_field,
        options.is_stop_field,
        options.pos_field,
        options.language_field,
        *TOKEN_IDENTITY_COLUMNS,
    ]
    if options.include_phrase_text:
        cols.append(options.text_field)
    return tuple(dict.fromkeys(cols))


def _require_columns(
    frame: pl.DataFrame, required: tuple[str, ...], model_name: str
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(
            f"Tokens model '{model_name}' is missing configured columns "
            f"{missing}; got: {sorted(frame.columns)}"
        )


def _reject_null_sentence_index(
    tokens: pl.DataFrame, options: KeyphraseOptions
) -> None:
    null_count = tokens[options.sentence_index_field].null_count()
    if null_count:
        raise ValueError(
            f"Tokens model '{options.tokens}' column '{options.sentence_index_field}' "
            f"contains {null_count} null value(s). Multi-token keyphrase extraction "
            f"(max_phrase_length > 1) requires sentence boundaries — rebuild the token "
            "table with a spaCy pipeline that includes the sentencizer or dependency "
            "parser, or set max_phrase_length: 1 to extract unigrams only."
        )


def _reject_unsupported_language(
    tokens: pl.DataFrame, options: KeyphraseOptions
) -> None:
    languages = tokens[options.language_field].drop_nulls().unique().to_list()
    unsupported = sorted(str(v) for v in languages if str(v) != options.language)
    if unsupported:
        raise ValueError(
            f"Keyphrase extraction is configured for language '{options.language}' "
            f"but the tokens carry language(s) {unsupported}. Configure a matching "
            "`language`, or split the corpus by language."
        )


# ── candidate n-gram construction ─────────────────────────────────────────────


def _build_candidates(
    tokens: pl.DataFrame, options: KeyphraseOptions
) -> pl.DataFrame:
    frame = (
        tokens
        .with_columns(
            pl.col(options.document_id_field).cast(pl.String()).alias("document_id"),
            pl.col(options.token_index_field).cast(_COUNT_DTYPE).alias("_tidx"),
            pl.col(options.sentence_index_field).cast(_COUNT_DTYPE).alias("_sidx"),
            pl.col(options.lemma_field).cast(pl.String()).alias("_lemma"),
            pl.col(options.is_stop_field).cast(pl.Boolean()).alias("_is_stop"),
            pl.col(options.pos_field).cast(pl.String()).alias("_pos"),
        )
        .sort(["document_id", "_sidx", "_tidx"])
    )
    if options.include_phrase_text:
        frame = frame.with_columns(
            pl.col(options.text_field).cast(pl.String()).alias("_text")
        )

    stop_pos = frozenset(options.stop_pos)
    parts: list[pl.DataFrame] = []
    for n in range(options.min_phrase_length, options.max_phrase_length + 1):
        part = _extract_ngrams(frame, n, stop_pos, options.include_phrase_text)
        if not part.is_empty():
            parts.append(part)

    if not parts:
        return pl.DataFrame(
            schema={
                "document_id": pl.String(),
                "phrase_lemma": pl.String(),
                "token_start": _COUNT_DTYPE,
                "token_end": _COUNT_DTYPE,
                "phrase_length": _COUNT_DTYPE,
                "sentence_index": _COUNT_DTYPE,
                **({"phrase_text": pl.String()} if options.include_phrase_text else {}),
            }
        )
    return pl.concat(parts)


def _extract_ngrams(
    frame: pl.DataFrame,
    n: int,
    stop_pos: frozenset[str],
    include_text: bool,
) -> pl.DataFrame:
    """All valid n-grams of exactly n consecutive tokens within a sentence."""
    group = ["document_id", "_sidx"]
    stop_pos_list = sorted(stop_pos)

    if n == 1:
        valid = frame.filter(~pl.col("_is_stop") & ~pl.col("_pos").is_in(stop_pos_list))
        selects: list[pl.Expr] = [
            pl.col("document_id"),
            pl.col("_lemma").alias("phrase_lemma"),
            pl.col("_tidx").alias("token_start"),
            pl.col("_tidx").alias("token_end"),
            pl.lit(1).cast(_COUNT_DTYPE).alias("phrase_length"),
            pl.col("_sidx").alias("sentence_index"),
        ]
        if include_text:
            selects.append(pl.col("_text").alias("phrase_text"))
        return valid.select(selects)

    # Add shifted columns for positions 1..n-1 within each (document, sentence).
    f = frame.clone()
    for i in range(1, n):
        shifts: list[pl.Expr] = [
            pl.col("_lemma").shift(-i).over(group).alias(f"_lemma_{i}"),
            pl.col("_tidx").shift(-i).over(group).alias(f"_tidx_{i}"),
            pl.col("_is_stop").shift(-i).over(group).alias(f"_stop_{i}"),
            pl.col("_pos").shift(-i).over(group).alias(f"_pos_{i}"),
        ]
        if include_text:
            shifts.append(pl.col("_text").shift(-i).over(group).alias(f"_text_{i}"))
        f = f.with_columns(shifts)

    # Consecutive token indices and no null shift overflow.
    consecutive = functools.reduce(
        lambda a, b: a & b,
        (pl.col(f"_tidx_{i}") == (pl.col("_tidx") + i) for i in range(1, n)),
    )
    not_null = functools.reduce(
        lambda a, b: a & b,
        (pl.col(f"_lemma_{i}").is_not_null() for i in range(1, n)),
    )
    # Boundary condition: first and last tokens must not be stop words or stop POS.
    first_ok = ~pl.col("_is_stop") & ~pl.col("_pos").is_in(stop_pos_list)
    last_ok = ~pl.col(f"_stop_{n - 1}") & ~pl.col(f"_pos_{n - 1}").is_in(stop_pos_list)
    f = f.filter(consecutive & not_null & first_ok & last_ok)

    lemma_cols = ["_lemma"] + [f"_lemma_{i}" for i in range(1, n)]
    f = f.with_columns(
        pl.concat_str(lemma_cols, separator=" ").alias("phrase_lemma"),
    )

    selects = [
        pl.col("document_id"),
        pl.col("phrase_lemma"),
        pl.col("_tidx").alias("token_start"),
        pl.col(f"_tidx_{n - 1}").alias("token_end"),
        pl.lit(n).cast(_COUNT_DTYPE).alias("phrase_length"),
        pl.col("_sidx").alias("sentence_index"),
    ]
    if include_text:
        text_cols = ["_text"] + [f"_text_{i}" for i in range(1, n)]
        f = f.with_columns(pl.concat_str(text_cols, separator=" ").alias("phrase_text"))
        selects.append(pl.col("phrase_text"))
    return f.select(selects)


# ── scoring and ranking ───────────────────────────────────────────────────────


def _score_candidates(
    candidates: pl.DataFrame, options: KeyphraseOptions
) -> pl.DataFrame:
    """Occurrence count per (document_id, phrase_lemma), normalized by the total
    number of candidate n-grams in the document."""
    aggs: list[pl.Expr] = [
        pl.len().cast(_COUNT_DTYPE).alias("_count"),
        pl.col("token_start").first(),
        pl.col("token_end").first(),
        pl.col("phrase_length").first(),
        pl.col("sentence_index").first(),
    ]
    if options.include_phrase_text:
        aggs.append(pl.col("phrase_text").first())

    counts = candidates.group_by(["document_id", "phrase_lemma"]).agg(aggs)

    totals = candidates.group_by("document_id").agg(
        pl.len().cast(_COUNT_DTYPE).alias("_total")
    )
    scored = counts.join(totals, on="document_id", how="left").with_columns(
        (pl.col("_count") / pl.col("_total")).cast(_SCORE_DTYPE).alias("score")
    )
    return scored.drop(["_count", "_total"])


def _rank_and_filter(
    scored: pl.DataFrame, options: KeyphraseOptions
) -> pl.DataFrame:
    filtered = scored.filter(pl.col("score") >= options.min_score)
    if filtered.is_empty():
        return filtered

    # Sort globally: primary by (document_id, score DESC, phrase_lemma ASC),
    # then assign 1-based row indices within each document group.
    ranked = filtered.sort(
        ["document_id", "score", "phrase_lemma"], descending=[False, True, False]
    ).with_columns(
        (pl.int_range(pl.len(), dtype=_COUNT_DTYPE).over("document_id") + 1).alias("rank")
    )
    return ranked.filter(pl.col("rank") <= options.top_k)


# ── NLP identity ─────────────────────────────────────────────────────────────


def _document_identity(tokens: pl.DataFrame, options: KeyphraseOptions) -> pl.DataFrame:
    frame = tokens.with_columns(
        pl.col(options.document_id_field).cast(pl.String()).alias("document_id")
    )
    return _single_valued(frame, TOKEN_IDENTITY_COLUMNS, options.tokens)


def _single_valued(
    frame: pl.DataFrame, columns: tuple[str, ...], model_name: str
) -> pl.DataFrame:
    grouped = frame.group_by("document_id").agg(
        [pl.col(c).first() for c in columns]
        + [pl.col(c).n_unique().alias(f"__n_{c}") for c in columns]
    )
    inconsistent = grouped.filter(
        pl.any_horizontal([pl.col(f"__n_{c}") > 1 for c in columns])
    )
    if not inconsistent.is_empty():
        documents = sorted(inconsistent["document_id"].to_list())[:5]
        raise ValueError(
            f"Model '{model_name}' has documents whose token rows disagree on "
            f"NLP identity {list(columns)}: {documents}. Rebuild the token table "
            "so each document carries one value."
        )
    return grouped.select("document_id", *columns)


# ── finalization and output ordering ─────────────────────────────────────────


def _finalize(result: pl.DataFrame, options: KeyphraseOptions) -> pl.DataFrame:
    doc_ids = result["document_id"].to_list()
    lemmas = result["phrase_lemma"].to_list()
    phrase_ids = [
        canonical_fingerprint((doc_id, lemma), domain=KEYPHRASE_DOMAIN)
        for doc_id, lemma in zip(doc_ids, lemmas, strict=True)
    ]
    result = result.with_columns(
        pl.Series("phrase_id", phrase_ids, dtype=pl.String()),
        pl.lit(KEYPHRASE_EXTRACTOR_NAME).alias("extractor"),
        pl.lit(KEYPHRASE_EXTRACTOR_VERSION).alias("extractor_version"),
    )
    return result.select(_output_columns(options)).sort(["document_id", "rank"])


def _output_columns(options: KeyphraseOptions) -> list[str]:
    cols = [
        "phrase_id",
        "document_id",
        "rank",
        "score",
        "phrase_lemma",
        "phrase_length",
        "token_start",
        "token_end",
        "sentence_index",
    ]
    if options.include_phrase_text:
        cols.append("phrase_text")
    cols.extend(TOKEN_IDENTITY_COLUMNS)
    cols.extend(["extractor", "extractor_version"])
    return list(dict.fromkeys(cols))


def _empty_output(options: KeyphraseOptions) -> pl.DataFrame:
    schema: dict[str, pl.DataType] = {
        "phrase_id": pl.String(),
        "document_id": pl.String(),
        "rank": _COUNT_DTYPE,
        "score": _SCORE_DTYPE,
        "phrase_lemma": pl.String(),
        "phrase_length": _COUNT_DTYPE,
        "token_start": _COUNT_DTYPE,
        "token_end": _COUNT_DTYPE,
        "sentence_index": _COUNT_DTYPE,
    }
    if options.include_phrase_text:
        schema["phrase_text"] = pl.String()
    for col in TOKEN_IDENTITY_COLUMNS:
        schema[col] = pl.String()
    schema["extractor"] = pl.String()
    schema["extractor_version"] = pl.String()
    return pl.DataFrame(schema=schema)
