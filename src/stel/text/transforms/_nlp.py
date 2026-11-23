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
from ..nlp_routing import (
    FALLBACK_ROUTE_KEY,
    NLPLanguageRouting,
    NLPRoute,
    RouteDecision,
    decide_route,
    observed_language,
)
from ._helpers import require_text_column, upstream_df

_IDENTITY_SCHEMA: dict[str, pl.DataType] = {
    "nlp_provider": pl.String(),
    "nlp_provider_version": pl.String(),
    "nlp_model": pl.String(),
    "nlp_model_version": pl.String(),
    "nlp_language": pl.String(),
}

# Added only when `language_routing` is configured, so an unrouted project's
# output schema is unchanged. `nlp_route` is the matched route key (or
# `fallback`); `nlp_input_language` is the language routing acted on, null when
# there was none; `nlp_language_source` says whether it was declared or detected.
_ROUTING_SCHEMA: dict[str, pl.DataType] = {
    "nlp_route": pl.String(),
    "nlp_input_language": pl.String(),
    "nlp_language_source": pl.String(),
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
    declared_language: str | None = None


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
    documents, schema = _input_documents(
        frame, options, _with_routing_columns(_TOKEN_SCHEMA, options)
    )
    if not documents:
        return pl.DataFrame(schema=schema)

    rows: list[dict[str, Any]] = []
    for source, analyzed_document, identity in _analyze(documents, options):
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
    base_schema = _with_routing_columns(_ENTITY_SCHEMA, options)
    if options.include_text:
        base_schema["entity_text"] = pl.String()
    frame = upstream_df(deps)
    documents, schema = _input_documents(
        frame,
        options,
        base_schema,
        reserved_fields=set(base_schema) | {"entity_text"},
    )
    if not documents:
        return pl.DataFrame(schema=schema)

    rows: list[dict[str, Any]] = []
    for source, analyzed_document, identity in _analyze(documents, options):
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
    routing = options.language_routing
    language_field = routing.language_field if routing is not None else None
    if language_field is not None and language_field not in frame.columns:
        raise ValueError(
            f"Expected language column '{language_field}' in upstream; "
            f"got: {sorted(frame.columns)}"
        )
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
        raw_language = row[language_field] if language_field is not None else None
        if raw_language is not None and not isinstance(raw_language, str):
            raise ValueError(
                f"Language column '{language_field}' must contain strings or nulls"
            )
        documents.append(
            _InputDocument(
                document_id=document_id,
                text=raw_text or "",
                included={field: row[field] for field in options.include_fields},
                declared_language=raw_language,
            )
        )
    return documents, schema


def _with_routing_columns(
    base: Mapping[str, pl.DataType],
    options: NLPTokenOptions | NLPEntityOptions,
) -> dict[str, pl.DataType]:
    schema = dict(base)
    if options.language_routing is not None:
        schema.update(_ROUTING_SCHEMA)
    return schema


def _analyze(
    documents: list[_InputDocument],
    options: NLPTokenOptions | NLPEntityOptions,
) -> Iterable[tuple[_InputDocument, nlp_core.NLPDocument, dict[str, str | None]]]:
    """Each analyzed document with the identity columns to stamp on its rows.

    Without routing this is one provider over the whole input. With routing,
    every document is resolved to a route first (a `fail` policy raises before
    any model loads), then each distinct route is loaded once and runs over its
    own documents."""
    routing = options.language_routing
    if routing is None:
        provider = nlp_core.get_nlp_provider(options)
        analyzed = provider.pipe(
            (document.text for document in documents),
            batch_size=options.batch_size,
        )
        identity: dict[str, str | None] = dict(_identity_values(provider.identity))
        return [
            (source, result, identity)
            for source, result in _matched_documents(documents, analyzed)
        ]
    return _analyze_routed(documents, routing, options.batch_size)


def _analyze_routed(
    documents: list[_InputDocument],
    routing: NLPLanguageRouting,
    batch_size: int,
) -> list[tuple[_InputDocument, nlp_core.NLPDocument, dict[str, str | None]]]:
    decisions: list[RouteDecision] = []
    for document in documents:
        observed = observed_language(
            routing, declared=document.declared_language, text=document.text
        )
        decisions.append(
            decide_route(routing, document_id=document.document_id, observed=observed)
        )

    by_route: dict[str, list[int]] = {}
    for index, decision in enumerate(decisions):
        if decision.route_key is not None:
            by_route.setdefault(decision.route_key, []).append(index)

    providers: dict[str, nlp_core.NLPProvider] = {}
    for key in by_route:
        route, language = _route_and_language(routing, key)
        providers[key] = nlp_core.load_nlp_route(route, language)

    results: dict[int, nlp_core.NLPDocument] = {}
    for key, indexes in by_route.items():
        group = [documents[index] for index in indexes]
        analyzed = providers[key].pipe(
            (document.text for document in group), batch_size=batch_size
        )
        for index, (_, result) in zip(
            indexes, _matched_documents(group, analyzed), strict=True
        ):
            results[index] = result

    output: list[tuple[_InputDocument, nlp_core.NLPDocument, dict[str, str | None]]] = []
    for index, document in enumerate(documents):
        decision = decisions[index]
        if decision.route_key is None:
            continue
        identity: dict[str, str | None] = dict(
            _identity_values(providers[decision.route_key].identity)
        )
        identity["nlp_route"] = decision.route_key
        identity["nlp_input_language"] = decision.observed_language
        identity["nlp_language_source"] = decision.language_source
        output.append((document, results[index], identity))
    return output


def _route_and_language(routing: NLPLanguageRouting, key: str) -> tuple[NLPRoute, str]:
    if key == FALLBACK_ROUTE_KEY:
        fallback = routing.fallback
        if fallback is None or fallback.language is None:
            raise ValueError("fallback route is not configured")
        return fallback, fallback.language
    return routing.routes[key], routing.route_language(key)


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
