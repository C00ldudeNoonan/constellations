"""Vectorized document-grain aggregation over the NLP child tables.

Unlike the row-constructing NLP transforms, this module never iterates rows: all
rollups compile to polars group-by expressions so a large corpus costs one pass
per child table rather than per-document Python.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import polars as pl

from ...transforms import TransformContext
from ..features import (
    DocumentFeatureOptions,
    entity_label_column,
    link_namespace_column,
    link_status_column,
    pos_count_column,
    pos_ratio_column,
)

IDENTITY_COLUMNS: tuple[str, ...] = (
    "nlp_provider",
    "nlp_provider_version",
    "nlp_model",
    "nlp_model_version",
    "nlp_language",
)

# Link-table identity is named for the resolver, and is prefixed on output so it
# cannot be confused with the spaCy identity carried by the token table.
LINK_IDENTITY_COLUMNS: tuple[str, str] = ("resolver", "resolver_version")
LINK_IDENTITY_EXTRA = "alias_set_version"

_COUNT_DTYPE = pl.Int64()
_RATIO_DTYPE = pl.Float64()

_STOP_COUNT = "__stop_count"
_ALPHA_COUNT = "__alpha_count"
_UNIQUE_LEMMA = "__unique_lemma_count"
_SENTENCE_DISTINCT = "__sentence_distinct"
_HAS_SENTENCE = "__has_sentence"


def validate_feature_options(options: Mapping[str, Any]) -> None:
    DocumentFeatureOptions.model_validate(options)


def declared_feature_dependencies(options: Mapping[str, Any]) -> tuple[str, ...]:
    return DocumentFeatureOptions.model_validate(options).declared_dependencies()


def run_document_features(
    deps: dict[str, pl.DataFrame],
    ctx: TransformContext,
) -> pl.DataFrame:
    options = DocumentFeatureOptions.model_validate(ctx.options)
    frames = _dep_frames(deps, options)
    tokens = frames[options.tokens]

    _require_columns(tokens, _token_columns(options), options.tokens, "tokens")
    identity = _document_identity(tokens, options, options.tokens)

    universe, metadata = _document_universe(frames, options)
    features = _token_features(tokens, options)

    if options.entities is not None:
        entities = frames[options.entities]
        _require_columns(
            entities, _entity_columns(options), options.entities, "entities"
        )
        _reconcile_identity(identity, entities, options, options.entities)
        features = features.join(
            _entity_features(entities, options), on="document_id", how="full",
            coalesce=True,
        )

    link_identity: pl.DataFrame | None = None
    if options.links is not None:
        links = frames[options.links]
        _require_columns(links, _link_columns(options), options.links, "links")
        link_identity = _link_identity(links, options, options.links)
        features = features.join(
            _link_features(links, options), on="document_id", how="full",
            coalesce=True,
        )

    combined = universe.join(features, on="document_id", how="left")
    combined = combined.join(identity, on="document_id", how="left")
    if link_identity is not None:
        combined = combined.join(link_identity, on="document_id", how="left")
    if metadata is not None:
        combined = combined.join(metadata, on="document_id", how="left")

    combined = _fill_counts(combined, options)
    combined = _add_ratios(combined, options)
    return combined.select(_output_order(options)).sort("document_id")


def _dep_frames(
    deps: dict[str, pl.DataFrame],
    options: DocumentFeatureOptions,
) -> dict[str, pl.DataFrame]:
    expected = set(options.declared_dependencies())
    if set(deps) != expected:
        raise ValueError(
            "Document features expect dependencies named by the `tokens`, "
            f"`entities`, `links`, and `documents` options ({sorted(expected)}); "
            f"got: {sorted(deps)}"
        )
    return deps


def _token_columns(options: DocumentFeatureOptions) -> tuple[str, ...]:
    columns = [options.document_id_field, options.lemma_field]
    if "sentence_count" in options.emit:
        columns.append(options.sentence_index_field)
    if options.pos_counts or options.pos_ratios:
        columns.append(options.pos_field)
    if "stop_ratio" in options.emit:
        columns.append("is_stop")
    if "alpha_ratio" in options.emit:
        columns.append("is_alpha")
    return tuple(dict.fromkeys(columns))


def _entity_columns(options: DocumentFeatureOptions) -> tuple[str, ...]:
    columns = [options.document_id_field]
    if options.entity_label_counts:
        columns.append(options.label_field)
    return tuple(dict.fromkeys(columns))


def _link_columns(options: DocumentFeatureOptions) -> tuple[str, ...]:
    columns = [options.document_id_field]
    if options.link_namespace_counts:
        columns.extend((options.namespace_field, options.canonical_id_field))
    if options.link_status_counts:
        columns.append(options.status_field)
    return tuple(dict.fromkeys(columns))


def _require_columns(
    frame: pl.DataFrame,
    required: tuple[str, ...],
    model_name: str,
    role: str,
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(
            f"{role.capitalize()} model '{model_name}' is missing configured "
            f"columns {missing}; got: {sorted(frame.columns)}"
        )


def _renamed_id(frame: pl.DataFrame, id_field: str) -> pl.DataFrame:
    if id_field == "document_id":
        return frame
    return frame.rename({id_field: "document_id"})


def _document_universe(
    frames: dict[str, pl.DataFrame],
    options: DocumentFeatureOptions,
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """The set of documents that get a row. With a `documents:` dependency the
    parent table defines it, so a document with zero tokens still appears with
    zero counts; without one, only documents present in the token table do."""
    if options.documents is None:
        tokens = frames[options.tokens]
        universe = (
            _renamed_id(tokens.select(options.document_id_field), options.document_id_field)
            .unique()
        )
        return universe.with_columns(pl.col("document_id").cast(pl.String())), None

    documents = frames[options.documents]
    required = (options.documents_id_field, *options.include_fields)
    _require_columns(documents, required, options.documents, "documents")
    selected = documents.select(required)
    selected = _renamed_id(selected, options.documents_id_field)
    selected = selected.with_columns(pl.col("document_id").cast(pl.String()))
    if selected["document_id"].null_count():
        raise ValueError(
            f"Documents model '{options.documents}' column "
            f"'{options.documents_id_field}' contains null values"
        )
    if selected["document_id"].n_unique() != selected.height:
        raise ValueError(
            f"Documents model '{options.documents}' column "
            f"'{options.documents_id_field}' contains duplicate values"
        )
    universe = selected.select("document_id")
    metadata = selected if options.include_fields else None
    return universe, metadata


def _token_features(
    tokens: pl.DataFrame,
    options: DocumentFeatureOptions,
) -> pl.DataFrame:
    frame = _renamed_id(tokens, options.document_id_field).with_columns(
        pl.col("document_id").cast(pl.String())
    )
    aggregations = [pl.len().alias("token_count").cast(_COUNT_DTYPE)]

    if "unique_lemma_count" in options.emit or "lexical_diversity" in options.emit:
        aggregations.append(
            pl.col(options.lemma_field).n_unique().alias(_UNIQUE_LEMMA).cast(
                _COUNT_DTYPE
            )
        )
    if "sentence_count" in options.emit:
        aggregations.extend(
            (
                pl.col(options.sentence_index_field)
                .drop_nulls()
                .n_unique()
                .alias(_SENTENCE_DISTINCT)
                .cast(_COUNT_DTYPE),
                pl.col(options.sentence_index_field)
                .is_not_null()
                .any()
                .alias(_HAS_SENTENCE),
            )
        )
    if "stop_ratio" in options.emit:
        aggregations.append(
            pl.col("is_stop").sum().alias(_STOP_COUNT).cast(_COUNT_DTYPE)
        )
    if "alpha_ratio" in options.emit:
        aggregations.append(
            pl.col("is_alpha").sum().alias(_ALPHA_COUNT).cast(_COUNT_DTYPE)
        )

    # Ratios need their underlying count even when only the ratio is requested.
    for value in dict.fromkeys((*options.pos_counts, *options.pos_ratios)):
        aggregations.append(
            (pl.col(options.pos_field) == value)
            .sum()
            .alias(pos_count_column(value))
            .cast(_COUNT_DTYPE)
        )

    return frame.group_by("document_id").agg(aggregations)


def _entity_features(
    entities: pl.DataFrame,
    options: DocumentFeatureOptions,
) -> pl.DataFrame:
    frame = _renamed_id(entities, options.document_id_field).with_columns(
        pl.col("document_id").cast(pl.String())
    )
    aggregations = [pl.len().alias("entity_count").cast(_COUNT_DTYPE)]
    for value in options.entity_label_counts:
        aggregations.append(
            (pl.col(options.label_field) == value)
            .sum()
            .alias(entity_label_column(value))
            .cast(_COUNT_DTYPE)
        )
    return frame.group_by("document_id").agg(aggregations)


def _link_features(
    links: pl.DataFrame,
    options: DocumentFeatureOptions,
) -> pl.DataFrame:
    frame = _renamed_id(links, options.document_id_field).with_columns(
        pl.col("document_id").cast(pl.String())
    )
    aggregations: list[pl.Expr] = []
    for value in options.link_namespace_counts:
        # Distinct canonical IDs, so an ambiguous mention resolving to two IDs
        # counts both, while an unmatched mention (null ID) counts neither.
        aggregations.append(
            pl.col(options.canonical_id_field)
            .filter(pl.col(options.namespace_field) == value)
            .drop_nulls()
            .n_unique()
            .alias(link_namespace_column(value))
            .cast(_COUNT_DTYPE)
        )
    for value in options.link_status_counts:
        aggregations.append(
            (pl.col(options.status_field) == value)
            .sum()
            .alias(link_status_column(value))
            .cast(_COUNT_DTYPE)
        )
    if not aggregations:
        return frame.select("document_id").unique()
    return frame.group_by("document_id").agg(aggregations)


def _document_identity(
    frame: pl.DataFrame,
    options: DocumentFeatureOptions,
    model_name: str,
) -> pl.DataFrame:
    _require_columns(frame, IDENTITY_COLUMNS, model_name, "tokens")
    return _single_valued_identity(
        frame, options, IDENTITY_COLUMNS, IDENTITY_COLUMNS, model_name
    )


def _link_identity(
    frame: pl.DataFrame,
    options: DocumentFeatureOptions,
    model_name: str,
) -> pl.DataFrame:
    source = (*LINK_IDENTITY_COLUMNS, LINK_IDENTITY_EXTRA)
    _require_columns(frame, source, model_name, "links")
    return _single_valued_identity(
        frame,
        options,
        source,
        tuple(f"link_{column}" for column in source),
        model_name,
    )


def _single_valued_identity(
    frame: pl.DataFrame,
    options: DocumentFeatureOptions,
    source_columns: tuple[str, ...],
    output_columns: tuple[str, ...],
    model_name: str,
) -> pl.DataFrame:
    """One identity row per document, rejecting documents whose child rows
    disagree — emitting a single identity for rows produced by two different
    models would make the reproducibility guarantee false."""
    renamed = _renamed_id(frame, options.document_id_field).with_columns(
        pl.col("document_id").cast(pl.String())
    )
    grouped = renamed.group_by("document_id").agg(
        [pl.col(column).first().alias(column) for column in source_columns]
        + [
            pl.col(column).n_unique().alias(f"__n_{column}")
            for column in source_columns
        ]
    )
    inconsistent = grouped.filter(
        pl.any_horizontal(
            [pl.col(f"__n_{column}") > 1 for column in source_columns]
        )
    )
    if not inconsistent.is_empty():
        documents = sorted(inconsistent["document_id"].to_list())[:5]
        raise ValueError(
            f"Model '{model_name}' has documents whose rows disagree on "
            f"{list(source_columns)}: {documents}. Aggregates cannot claim a "
            "single reproducible identity; rebuild the child table so each "
            "document is produced by one model."
        )
    return grouped.select(
        "document_id",
        *[
            pl.col(source).alias(output)
            for source, output in zip(source_columns, output_columns, strict=True)
        ],
    )


def _reconcile_identity(
    token_identity: pl.DataFrame,
    entities: pl.DataFrame,
    options: DocumentFeatureOptions,
    model_name: str,
) -> None:
    """Entity rows carry the same identity contract as token rows; when both are
    present they must agree per document, or the emitted identity would describe
    only half the row's features."""
    if set(IDENTITY_COLUMNS) - set(entities.columns):
        return
    entity_identity = _single_valued_identity(
        entities, options, IDENTITY_COLUMNS, IDENTITY_COLUMNS, model_name
    )
    comparison = token_identity.join(
        entity_identity, on="document_id", how="inner", suffix="__entity"
    )
    mismatched = comparison.filter(
        pl.any_horizontal(
            [
                pl.col(column) != pl.col(f"{column}__entity")
                for column in IDENTITY_COLUMNS
            ]
        )
    )
    if not mismatched.is_empty():
        documents = sorted(mismatched["document_id"].to_list())[:5]
        raise ValueError(
            f"Entities model '{model_name}' disagrees with the tokens model on "
            f"NLP identity for documents {documents}. Rebuild both child tables "
            "with the same model so document features stay reproducible."
        )


def _fill_counts(
    frame: pl.DataFrame, options: DocumentFeatureOptions
) -> pl.DataFrame:
    """Documents with no child rows are real documents with zero of everything —
    counts fill to 0, while ratios stay null because they are undefined."""
    count_columns = [
        column
        for column, _ in options.output_columns()
        if column.endswith("_count")
    ]
    # `token_count` is always computed because ratios divide by it, even when it
    # is not itself emitted; the internal helpers back the ratio numerators.
    internal = [
        "token_count",
        _STOP_COUNT,
        _ALPHA_COUNT,
        _UNIQUE_LEMMA,
        _SENTENCE_DISTINCT,
    ]
    present = [
        column
        for column in dict.fromkeys([*count_columns, *internal])
        if column in frame.columns and column not in options.include_fields
    ]
    return frame.with_columns(
        [pl.col(column).fill_null(0).cast(_COUNT_DTYPE) for column in present]
    )


def _ratio(numerator: str) -> pl.Expr:
    return (
        pl.when(pl.col("token_count") > 0)
        .then(pl.col(numerator) / pl.col("token_count"))
        .otherwise(None)
        .cast(_RATIO_DTYPE)
    )


def _add_ratios(
    frame: pl.DataFrame, options: DocumentFeatureOptions
) -> pl.DataFrame:
    expressions: list[pl.Expr] = []
    if "sentence_count" in options.emit:
        expressions.append(
            pl.when(pl.col(_HAS_SENTENCE).fill_null(False))
            .then(pl.col(_SENTENCE_DISTINCT))
            .otherwise(None)
            .cast(_COUNT_DTYPE)
            .alias("sentence_count")
        )
    if "unique_lemma_count" in options.emit:
        expressions.append(pl.col(_UNIQUE_LEMMA).alias("unique_lemma_count"))
    if "lexical_diversity" in options.emit:
        expressions.append(_ratio(_UNIQUE_LEMMA).alias("lexical_diversity"))
    if "stop_ratio" in options.emit:
        expressions.append(_ratio(_STOP_COUNT).alias("stop_ratio"))
    if "alpha_ratio" in options.emit:
        expressions.append(_ratio(_ALPHA_COUNT).alias("alpha_ratio"))
    for value in options.pos_ratios:
        expressions.append(
            _ratio(pos_count_column(value)).alias(pos_ratio_column(value))
        )
    if not expressions:
        return frame
    return frame.with_columns(expressions)


def _output_order(options: DocumentFeatureOptions) -> list[str]:
    columns = ["document_id"]
    columns.extend(
        column for column, _ in options.output_columns() if column not in columns
    )
    columns.extend(IDENTITY_COLUMNS)
    if options.links is not None:
        columns.extend(
            f"link_{column}"
            for column in (*LINK_IDENTITY_COLUMNS, LINK_IDENTITY_EXTRA)
        )
    return columns


__all__ = [
    "declared_feature_dependencies",
    "run_document_features",
    "validate_feature_options",
]
