"""Shared execution for built-in NLP child-table transforms."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import polars as pl

from ...hashing import canonical_fingerprint
from ...transforms import TransformContext
from .. import nlp as nlp_core
from ..nlp import NLPEntityOptions, NLPIdentity, NLPTokenOptions
from ._helpers import require_text_column, upstream_df

_IDENTITY_SCHEMA: dict[str, pl.DataType] = {
    "nlp_provider": pl.String(),
    "nlp_provider_version": pl.String(),
    "nlp_model": pl.String(),
    "nlp_model_version": pl.String(),
    "nlp_language": pl.String(),
}

_TOKEN_SCHEMA: dict[str, pl.DataType] = {
    "token_id": pl.String(),
    "document_id": pl.String(),
    "token_index": pl.Int64(),
    "sentence_index": pl.Int64(),
    "start": pl.Int64(),
    "end": pl.Int64(),
    "token_text": pl.String(),
    "lemma": pl.String(),
    "pos": pl.String(),
    "tag": pl.String(),
    "is_stop": pl.Boolean(),
    "is_alpha": pl.Boolean(),
    **_IDENTITY_SCHEMA,
}

_ENTITY_SCHEMA: dict[str, pl.DataType] = {
    "entity_id": pl.String(),
    "document_id": pl.String(),
    "entity_index": pl.Int64(),
    "sentence_index": pl.Int64(),
    "start": pl.Int64(),
    "end": pl.Int64(),
    "label": pl.String(),
    "confidence": pl.Float64(),
    **_IDENTITY_SCHEMA,
}


@dataclass(frozen=True)
class _InputDocument:
    document_id: str
    text: str
    included: dict[str, Any]


def validate_token_options(options: Mapping[str, Any]) -> None:
    NLPTokenOptions.model_validate(options)


def validate_entity_options(options: Mapping[str, Any]) -> None:
    NLPEntityOptions.model_validate(options)


def run_tokens(
    deps: dict[str, pl.DataFrame],
    ctx: TransformContext,
) -> pl.DataFrame:
    options = NLPTokenOptions.model_validate(ctx.options)
    frame = upstream_df(deps)
    documents, schema = _input_documents(frame, options, _TOKEN_SCHEMA)
    if not documents:
        return pl.DataFrame(schema=schema)

    provider = nlp_core.get_nlp_provider(options)
    analyzed = provider.pipe(
        (document.text for document in documents),
        batch_size=options.batch_size,
    )
    identity = _identity_values(provider.identity)

    rows: list[dict[str, Any]] = []
    for source, analyzed_document in _matched_documents(documents, analyzed):
        for token in analyzed_document.tokens:
            if token.is_space and not options.include_space:
                continue
            rows.append(
                {
                    "token_id": canonical_fingerprint(
                        {
                            "document_id": source.document_id,
                            "token_index": token.index,
                            "start": token.start,
                            "end": token.end,
                            "text": token.text,
                        },
                        domain="dbt-ml.nlp-token",
                    ),
                    "document_id": source.document_id,
                    "token_index": token.index,
                    "sentence_index": token.sentence_index,
                    "start": token.start,
                    "end": token.end,
                    "token_text": token.text,
                    "lemma": token.lemma,
                    "pos": token.pos,
                    "tag": token.tag,
                    "is_stop": token.is_stop,
                    "is_alpha": token.is_alpha,
                    **identity,
                    **source.included,
                }
            )
    return pl.DataFrame(rows, schema=schema, strict=False)


def run_entities(
    deps: dict[str, pl.DataFrame],
    ctx: TransformContext,
) -> pl.DataFrame:
    options = NLPEntityOptions.model_validate(ctx.options)
    base_schema = dict(_ENTITY_SCHEMA)
    if options.include_text:
        base_schema["entity_text"] = pl.String()
    frame = upstream_df(deps)
    documents, schema = _input_documents(
        frame,
        options,
        base_schema,
        reserved_fields=set(_ENTITY_SCHEMA) | {"entity_text"},
    )
    if not documents:
        return pl.DataFrame(schema=schema)

    provider = nlp_core.get_nlp_provider(options)
    analyzed = provider.pipe(
        (document.text for document in documents),
        batch_size=options.batch_size,
    )
    identity = _identity_values(provider.identity)

    rows: list[dict[str, Any]] = []
    for source, analyzed_document in _matched_documents(documents, analyzed):
        for entity in analyzed_document.entities:
            row = {
                "entity_id": canonical_fingerprint(
                    {
                        "document_id": source.document_id,
                        "entity_index": entity.index,
                        "start": entity.start,
                        "end": entity.end,
                        "label": entity.label,
                    },
                    domain="dbt-ml.nlp-entity",
                ),
                "document_id": source.document_id,
                "entity_index": entity.index,
                "sentence_index": entity.sentence_index,
                "start": entity.start,
                "end": entity.end,
                "label": entity.label,
                "confidence": entity.confidence,
                **identity,
                **source.included,
            }
            if options.include_text:
                row["entity_text"] = entity.text
            rows.append(row)
    return pl.DataFrame(rows, schema=schema, strict=False)


def _input_documents(
    frame: pl.DataFrame,
    options: NLPTokenOptions | NLPEntityOptions,
    base_schema: Mapping[str, pl.DataType],
    *,
    reserved_fields: set[str] | None = None,
) -> tuple[list[_InputDocument], dict[str, pl.DataType]]:
    require_text_column(frame, options.text_field)
    if options.document_id_field not in frame.columns:
        raise ValueError(
            f"Expected document ID column '{options.document_id_field}' in upstream; "
            f"got: {sorted(frame.columns)}"
        )

    missing_fields = sorted(set(options.include_fields) - set(frame.columns))
    if missing_fields:
        raise ValueError(f"include_fields contains unknown columns: {missing_fields}")
    collisions = sorted(
        set(options.include_fields) & (reserved_fields or set(base_schema))
    )
    if collisions:
        raise ValueError(
            f"include_fields collides with NLP output columns: {collisions}"
        )

    schema = dict(base_schema)
    schema.update({field: frame.schema[field] for field in options.include_fields})
    documents: list[_InputDocument] = []
    seen_ids: set[str] = set()
    for row in frame.iter_rows(named=True):
        raw_id = row[options.document_id_field]
        if raw_id is None or not str(raw_id).strip():
            raise ValueError(
                f"Document ID column '{options.document_id_field}' contains null "
                "or empty values"
            )
        document_id = str(raw_id)
        if document_id in seen_ids:
            raise ValueError(
                f"Document ID column '{options.document_id_field}' contains "
                f"duplicate value '{document_id}'"
            )
        seen_ids.add(document_id)

        raw_text = row[options.text_field]
        if raw_text is not None and not isinstance(raw_text, str):
            raise ValueError(
                f"Text column '{options.text_field}' must contain strings or nulls"
            )
        documents.append(
            _InputDocument(
                document_id=document_id,
                text=raw_text or "",
                included={field: row[field] for field in options.include_fields},
            )
        )
    return documents, schema


def _matched_documents(
    documents: list[_InputDocument],
    analyzed: Iterable[nlp_core.NLPDocument],
) -> Iterable[tuple[_InputDocument, nlp_core.NLPDocument]]:
    analyzed_iterator = iter(analyzed)
    for index, document in enumerate(documents):
        try:
            analyzed_document = next(analyzed_iterator)
        except StopIteration:
            raise ValueError(
                "NLP provider returned a different number of documents than it "
                f"received: expected {len(documents)}, got {index}"
            ) from None
        yield document, analyzed_document

    try:
        next(analyzed_iterator)
    except StopIteration:
        return
    actual_count = len(documents) + 1 + sum(1 for _ in analyzed_iterator)
    raise ValueError(
        "NLP provider returned a different number of documents than it received: "
        f"expected {len(documents)}, got {actual_count}"
    )


def _identity_values(identity: NLPIdentity) -> dict[str, str]:
    return {
        "nlp_provider": identity.provider,
        "nlp_provider_version": identity.provider_version,
        "nlp_model": identity.model,
        "nlp_model_version": identity.model_version,
        "nlp_language": identity.language,
    }
