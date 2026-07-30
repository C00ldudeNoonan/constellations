"""Vectorized document-grain tone scoring over the token child table.

Tokens are matched (case-insensitively on the configured field) against the
operator-owned tone lexicon; per-category scores are polars group-by
aggregations, so a large corpus costs one pass rather than per-document Python.
Negation, when enabled, flips a matched term preceded by a negator within a
bounded same-sentence window.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import polars as pl

from ...transforms import TransformContext
from ..tone import (
    TONE_SCORER_VERSION,
    ToneOptions,
    category_hits_column,
    category_score_column,
    tone_lexicon_fingerprint,
)

# The spaCy identity carried on every token row; recorded per document as tone
# provenance and required so scores name the tokenization that produced them.
TOKEN_IDENTITY_COLUMNS: tuple[str, ...] = (
    "nlp_provider",
    "nlp_provider_version",
    "nlp_model",
    "nlp_model_version",
    "nlp_language",
)

_COUNT_DTYPE = pl.Int64()
_SCORE_DTYPE = pl.Float64()
_MATCH_KEY = "__match_key"
_TOKEN_KEY = "__token_index"
_IS_NEGATOR = "__is_negator"
_NEGATED = "__negated"
_SIGN = "__sign"
_WEIGHT = "__weight"
_CATEGORY = "__category"


def validate_tone_options(options: Mapping[str, Any]) -> None:
    ToneOptions.model_validate(options)


def declared_tone_dependencies(options: Mapping[str, Any]) -> tuple[str, str]:
    return ToneOptions.model_validate(options).declared_dependencies()


def run_tone(deps: dict[str, pl.DataFrame], ctx: TransformContext) -> pl.DataFrame:
    options = ToneOptions.model_validate(ctx.options)
    tokens, lexicon = _dep_frames(deps, options)
    _require_columns(tokens, _token_columns(options), options.tokens, "tokens")

    lexicon_rows = _lexicon_rows(lexicon, options)
    lexicon_version = tone_lexicon_fingerprint(lexicon_rows)
    lexicon_frame = _lexicon_frame(lexicon_rows)

    universe = _document_universe(tokens, options)
    if universe.is_empty():
        return _empty_output(options, lexicon_version)

    _reject_unsupported_language(tokens, options)

    prepared = _prepare_tokens(tokens, options)
    token_counts = prepared.group_by("document_id").agg(
        pl.len().alias("token_count").cast(_COUNT_DTYPE)
    )
    matched = _matched_tokens(prepared, lexicon_frame)

    combined = universe.join(token_counts, on="document_id", how="left")
    combined = combined.join(_category_scores(matched, options), on="document_id", how="left")
    combined = combined.join(_coverage(matched), on="document_id", how="left")
    combined = combined.join(_document_identity(prepared, options), on="document_id", how="left")
    metadata = _document_metadata(prepared, options)
    if metadata is not None:
        combined = combined.join(metadata, on="document_id", how="left")

    combined = _finalize(combined, options, lexicon_version)
    return combined.select(_output_order(options)).sort("document_id")


def _dep_frames(
    deps: dict[str, pl.DataFrame], options: ToneOptions
) -> tuple[pl.DataFrame, pl.DataFrame]:
    expected = {options.tokens, options.lexicon}
    if set(deps) != expected:
        raise ValueError(
            "Tone scoring expects dependencies named by the `tokens` and `lexicon` "
            f"options ({sorted(expected)}); got: {sorted(deps)}"
        )
    return deps[options.tokens], deps[options.lexicon]


def _token_columns(options: ToneOptions) -> tuple[str, ...]:
    columns = [
        options.document_id_field,
        options.match_field,
        options.language_field,
        options.token_index_field,
        *TOKEN_IDENTITY_COLUMNS,
        *options.include_fields,
    ]
    if options.negation:
        columns.append(options.sentence_index_field)
    return tuple(dict.fromkeys(columns))


def _require_columns(
    frame: pl.DataFrame, required: tuple[str, ...], model_name: str, role: str
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(
            f"{role.capitalize()} model '{model_name}' is missing configured columns "
            f"{missing}; got: {sorted(frame.columns)}"
        )


def _lexicon_rows(lexicon: pl.DataFrame, options: ToneOptions) -> list[dict[str, Any]]:
    _require_columns(
        lexicon,
        (options.lexicon_term_field, options.lexicon_category_field),
        options.lexicon,
        "lexicon",
    )
    configured_weight = options.lexicon_weight_field
    weight_column = (
        configured_weight
        if configured_weight is not None and configured_weight in lexicon.columns
        else None
    )

    rows: list[dict[str, Any]] = []
    for position, row in enumerate(lexicon.iter_rows(named=True)):
        term = row[options.lexicon_term_field]
        category = row[options.lexicon_category_field]
        if not isinstance(term, str) or not term.strip():
            raise ValueError(
                f"Lexicon model '{options.lexicon}' column "
                f"'{options.lexicon_term_field}' must contain non-empty strings "
                f"(row {position})"
            )
        if not isinstance(category, str) or not category.strip():
            raise ValueError(
                f"Lexicon model '{options.lexicon}' column "
                f"'{options.lexicon_category_field}' must contain non-empty strings "
                f"(row {position})"
            )
        weight = 1.0
        if weight_column is not None and row[weight_column] is not None:
            try:
                weight = float(row[weight_column])
            except (TypeError, ValueError):
                raise ValueError(
                    f"Lexicon model '{options.lexicon}' column '{weight_column}' must "
                    f"be numeric (row {position})"
                ) from None
        rows.append({"term": term, "category": category, "weight": weight})
    return rows


def _lexicon_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """One row per (normalized term, category) with its weight. A term repeated
    in one category with conflicting weights is an operator error, not a silent
    pick."""
    schema = {_MATCH_KEY: pl.String(), _CATEGORY: pl.String(), _WEIGHT: pl.Float64()}
    if not rows:
        return pl.DataFrame(schema=schema)
    frame = pl.DataFrame(
        {
            _MATCH_KEY: [_normalize(row["term"]) for row in rows],
            _CATEGORY: [row["category"] for row in rows],
            _WEIGHT: [float(row["weight"]) for row in rows],
        },
        schema=schema,
    )
    conflicts = frame.group_by([_MATCH_KEY, _CATEGORY]).agg(
        pl.col(_WEIGHT).n_unique().alias("__n")
    )
    if not conflicts.filter(pl.col("__n") > 1).is_empty():
        raise ValueError(
            "Tone lexicon has conflicting weights for the same term and category"
        )
    return frame.unique([_MATCH_KEY, _CATEGORY], keep="first")


def _normalize(value: str) -> str:
    return value.strip().casefold()


def _document_universe(tokens: pl.DataFrame, options: ToneOptions) -> pl.DataFrame:
    universe = (
        _renamed_id(tokens.select(options.document_id_field), options.document_id_field)
        .with_columns(pl.col("document_id").cast(pl.String()))
        .unique()
    )
    if universe["document_id"].null_count():
        raise ValueError(
            f"Tokens model '{options.tokens}' column '{options.document_id_field}' "
            "contains null values"
        )
    return universe


def _reject_unsupported_language(tokens: pl.DataFrame, options: ToneOptions) -> None:
    languages = tokens[options.language_field].drop_nulls().unique().to_list()
    unsupported = sorted(str(value) for value in languages if str(value) != options.language)
    if unsupported:
        raise ValueError(
            f"Tone scoring is configured for language '{options.language}' but the "
            f"tokens carry language(s) {unsupported}. Configure a matching lexicon "
            "and `language`, or split the corpus by language."
        )


def _prepare_tokens(tokens: pl.DataFrame, options: ToneOptions) -> pl.DataFrame:
    frame = _renamed_id(tokens, options.document_id_field).with_columns(
        pl.col("document_id").cast(pl.String()),
        pl.col(options.token_index_field).alias(_TOKEN_KEY),
        pl.col(options.match_field)
        .cast(pl.String())
        .str.strip_chars()
        .str.to_lowercase()
        .alias(_MATCH_KEY),
    )
    if not options.negation:
        return frame.with_columns(pl.lit(1.0).alias(_SIGN))

    negators = sorted({_normalize(value) for value in options.negators})
    frame = frame.sort(
        ["document_id", options.sentence_index_field, options.token_index_field]
    ).with_columns(pl.col(_MATCH_KEY).is_in(negators).alias(_IS_NEGATOR))
    window_group = ["document_id", options.sentence_index_field]
    negated = pl.any_horizontal(
        [
            pl.col(_IS_NEGATOR).shift(distance).over(window_group)
            for distance in range(1, options.negation_window + 1)
        ]
    ).fill_null(False)
    return frame.with_columns(negated.alias(_NEGATED)).with_columns(
        pl.when(pl.col(_NEGATED)).then(-1.0).otherwise(1.0).alias(_SIGN)
    )


def _matched_tokens(prepared: pl.DataFrame, lexicon_frame: pl.DataFrame) -> pl.DataFrame:
    """One row per (token, matched category), carrying the term weight and the
    negation sign. Empty when nothing matches."""
    matched = prepared.select("document_id", _TOKEN_KEY, _MATCH_KEY, _SIGN).join(
        lexicon_frame, on=_MATCH_KEY, how="inner"
    )
    return matched.drop(_MATCH_KEY)


def _category_scores(matched: pl.DataFrame, options: ToneOptions) -> pl.DataFrame:
    schema: dict[str, pl.DataType] = {"document_id": pl.String()}
    for category in options.emit:
        schema[category_hits_column(category)] = _COUNT_DTYPE
        schema[f"__raw_{category_score_column(category)}"] = _SCORE_DTYPE
    if matched.is_empty():
        return pl.DataFrame(schema=schema)

    aggregations: list[pl.Expr] = []
    for category in options.emit:
        selector = pl.col(_CATEGORY) == category
        aggregations.append(
            selector.sum().alias(category_hits_column(category)).cast(_COUNT_DTYPE)
        )
        aggregations.append(
            pl.when(selector)
            .then(pl.col(_WEIGHT) * pl.col(_SIGN))
            .otherwise(0.0)
            .sum()
            .alias(f"__raw_{category_score_column(category)}")
            .cast(_SCORE_DTYPE)
        )
    return matched.group_by("document_id").agg(aggregations)


def _coverage(matched: pl.DataFrame) -> pl.DataFrame:
    if matched.is_empty():
        return pl.DataFrame(
            schema={"document_id": pl.String(), "matched_token_count": _COUNT_DTYPE}
        )
    return matched.group_by("document_id").agg(
        pl.col(_TOKEN_KEY).n_unique().alias("matched_token_count").cast(_COUNT_DTYPE)
    )


def _document_identity(prepared: pl.DataFrame, options: ToneOptions) -> pl.DataFrame:
    _require_columns(prepared, TOKEN_IDENTITY_COLUMNS, options.tokens, "tokens")
    return _single_valued(prepared, TOKEN_IDENTITY_COLUMNS, options.tokens, "NLP identity")


def _document_metadata(prepared: pl.DataFrame, options: ToneOptions) -> pl.DataFrame | None:
    if not options.include_fields:
        return None
    return _single_valued(prepared, options.include_fields, options.tokens, "include_fields")


def _single_valued(
    frame: pl.DataFrame, columns: tuple[str, ...], model_name: str, role: str
) -> pl.DataFrame:
    """One row per document, rejecting documents whose token rows disagree on
    these columns — a single value per document would otherwise be a fiction."""
    grouped = frame.group_by("document_id").agg(
        [pl.col(column).first().alias(column) for column in columns]
        + [pl.col(column).n_unique().alias(f"__n_{column}") for column in columns]
    )
    inconsistent = grouped.filter(
        pl.any_horizontal([pl.col(f"__n_{column}") > 1 for column in columns])
    )
    if not inconsistent.is_empty():
        documents = sorted(inconsistent["document_id"].to_list())[:5]
        raise ValueError(
            f"Model '{model_name}' has documents whose token rows disagree on "
            f"{role} {list(columns)}: {documents}. Rebuild the token table so each "
            "document carries one value."
        )
    return grouped.select("document_id", *columns)


def _finalize(
    frame: pl.DataFrame, options: ToneOptions, lexicon_version: str
) -> pl.DataFrame:
    frame = frame.with_columns(
        pl.col("token_count").fill_null(0).cast(_COUNT_DTYPE),
        pl.col("matched_token_count").fill_null(0).cast(_COUNT_DTYPE),
    )
    frame = frame.with_columns(
        [
            pl.col(category_hits_column(category)).fill_null(0).cast(_COUNT_DTYPE)
            for category in options.emit
        ]
    )

    # Below the minimum, scores are null (not a misleading 0); scores and coverage
    # divide by token_count and are null when it is 0.
    sufficient = pl.col("token_count") >= max(options.min_tokens, 1)
    frame = frame.with_columns(
        pl.when(sufficient)
        .then(pl.lit("scored"))
        .otherwise(pl.lit("insufficient_text"))
        .alias("status"),
        # Coverage and per-category scores are ratios over token_count; below the
        # minimum they are null (not a misleading 0), leaving only raw counts.
        pl.when(sufficient)
        .then(pl.col("matched_token_count") / pl.col("token_count"))
        .otherwise(None)
        .cast(_SCORE_DTYPE)
        .alias("coverage"),
    )
    score_expressions: list[pl.Expr] = []
    for category in options.emit:
        raw = pl.col(f"__raw_{category_score_column(category)}").fill_null(0.0)
        score_expressions.append(
            pl.when(sufficient)
            .then(raw / pl.col("token_count"))
            .otherwise(None)
            .cast(_SCORE_DTYPE)
            .alias(category_score_column(category))
        )
    frame = frame.with_columns(score_expressions)
    return frame.with_columns(
        pl.lit("lexicon").alias("scorer"),
        pl.lit(TONE_SCORER_VERSION).alias("scorer_version"),
        pl.lit(lexicon_version).alias("lexicon_version"),
    )


def _output_order(options: ToneOptions) -> list[str]:
    columns = ["document_id"]
    for category in options.emit:
        columns.append(category_score_column(category))
        columns.append(category_hits_column(category))
    columns.extend(("token_count", "matched_token_count", "coverage", "status"))
    columns.extend(options.include_fields)
    columns.extend(TOKEN_IDENTITY_COLUMNS)
    columns.extend(("scorer", "scorer_version", "lexicon_version"))
    return list(dict.fromkeys(columns))


def _empty_output(options: ToneOptions, lexicon_version: str) -> pl.DataFrame:
    schema: dict[str, pl.DataType] = {"document_id": pl.String()}
    for category in options.emit:
        schema[category_score_column(category)] = _SCORE_DTYPE
        schema[category_hits_column(category)] = _COUNT_DTYPE
    schema["token_count"] = _COUNT_DTYPE
    schema["matched_token_count"] = _COUNT_DTYPE
    schema["coverage"] = _SCORE_DTYPE
    schema["status"] = pl.String()
    for field in options.include_fields:
        schema[field] = pl.String()
    for column in TOKEN_IDENTITY_COLUMNS:
        schema[column] = pl.String()
    schema["scorer"] = pl.String()
    schema["scorer_version"] = pl.String()
    schema["lexicon_version"] = pl.String()
    return pl.DataFrame(schema=schema)


def _renamed_id(frame: pl.DataFrame, id_field: str) -> pl.DataFrame:
    if id_field == "document_id":
        return frame
    return frame.rename({id_field: "document_id"})
