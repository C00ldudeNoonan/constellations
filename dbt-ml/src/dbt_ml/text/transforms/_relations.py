"""Shared execution for the built-in relation-extraction child-table transform.

The driver owns mention identity, column validation, the evidence-text opt-in,
row shaping, and the stable ``relation_id``. The pairing itself is delegated to
the extractor selected by the ``extractor`` option; see ``dbt_ml.text.relations``.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import polars as pl

from ...transforms import IncrementalContract, TransformContext
from ..relations import (
    Mention,
    RelationExtractor,
    _order_key,
    get_relation_extractor,
    parse_relation_options,
    relation_id,
)

_RELATION_SCHEMA: dict[str, pl.DataType] = {
    "relation_id": pl.String(),
    "document_id": pl.String(),
    "subject_mention_id": pl.String(),
    "object_mention_id": pl.String(),
    "relation_type": pl.String(),
    # False for symmetric co-occurrence; a typed extractor sets True.
    "directed": pl.Boolean(),
    # The method category ("co_occurrence" / "rule" / "model_assertion").
    "method": pl.String(),
    "status": pl.String(),
    # Reserved for score-producing extractors; co-occurrence emits null rather
    # than implying a calibrated confidence.
    "confidence": pl.Float64(),
    "sentence_index": pl.Int64(),
    "subject_start": pl.Int64(),
    "subject_end": pl.Int64(),
    "object_start": pl.Int64(),
    "object_end": pl.Int64(),
    "subject_label": pl.String(),
    "object_label": pl.String(),
    "extractor": pl.String(),
    "extractor_version": pl.String(),
}


def validate_relation_options(options: Mapping[str, Any]) -> None:
    parse_relation_options(options)


def declared_relation_dependencies(options: Mapping[str, Any]) -> tuple[str, ...]:
    return (parse_relation_options(options).mentions,)


def declared_relation_incremental_contract(
    options: Mapping[str, Any],
) -> IncrementalContract:
    """One relation child table per document; re-analyzing a changed document
    replaces exactly its relation rows (issue #218). Child rows are keyed by
    ``relation_id``."""
    parsed = parse_relation_options(options)
    return IncrementalContract(
        parent_key="document_id",
        child_key="relation_id",
        parent_source=parsed.mentions,
        parent_source_key=parsed.document_id_field,
    )


def run_relations(
    deps: dict[str, pl.DataFrame],
    ctx: TransformContext,
) -> pl.DataFrame:
    options = parse_relation_options(ctx.options)
    extractor = get_relation_extractor(options.extractor)
    frame = _mentions_frame(deps, options)
    schema = _output_schema(options)
    grouped = _grouped_mentions(frame, options, extractor)
    if not grouped:
        return pl.DataFrame(schema=schema)

    rows: list[dict[str, Any]] = []
    for document_id in sorted(grouped):
        for relation in extractor.extract(grouped[document_id], options):
            rows.append(_relation_row(document_id, relation, options, extractor))
    return pl.DataFrame(rows, schema=schema, strict=False)


def _mentions_frame(
    deps: dict[str, pl.DataFrame], options: Any
) -> pl.DataFrame:
    expected = {options.mentions}
    if set(deps) != expected:
        raise ValueError(
            "Relation extraction expects the dependency named by the `mentions` "
            f"option ({sorted(expected)}); got: {sorted(deps)}"
        )
    return deps[options.mentions]


def _required_columns(options: Any, extractor: RelationExtractor) -> list[str]:
    required = [
        options.mention_id_field,
        options.document_id_field,
        options.sentence_index_field,
        options.start_field,
        options.end_field,
        *extractor.required_mention_columns(options),
    ]
    if options.label_field is not None:
        required.append(options.label_field)
    if options.include_mention_text or extractor.text_required():
        required.append(options.mention_text_field)
    return list(dict.fromkeys(required))


def _output_schema(options: Any) -> dict[str, pl.DataType]:
    schema = dict(_RELATION_SCHEMA)
    if options.include_mention_text:
        schema["subject_text"] = pl.String()
        schema["object_text"] = pl.String()
    return schema


def _grouped_mentions(
    frame: pl.DataFrame,
    options: Any,
    extractor: RelationExtractor,
) -> dict[str, list[Mention]]:
    required = _required_columns(options, extractor)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        text_hint = (
            " If mention text is absent, the upstream NLP transform may need "
            "`include_text: true`."
            if options.mention_text_field in missing
            and (options.include_mention_text or extractor.text_required())
            else ""
        )
        raise ValueError(
            f"Mentions model '{options.mentions}' is missing configured columns "
            f"{missing}; got: {sorted(frame.columns)}.{text_hint}"
        )

    want_text = options.include_mention_text or extractor.text_required()
    label_allow = set(options.labels)
    grouped: dict[str, list[Mention]] = {}
    seen_ids: set[str] = set()
    for row in frame.iter_rows(named=True):
        mention_id = _required_id(row, options.mention_id_field, "Mention ID")
        if mention_id in seen_ids:
            raise ValueError(
                f"Mention ID column '{options.mention_id_field}' contains duplicate "
                f"value '{mention_id}'"
            )
        seen_ids.add(mention_id)
        document_id = _required_id(row, options.document_id_field, "Document ID")
        label = (
            _optional_str(row, options.label_field, "Label")
            if options.label_field is not None
            else None
        )
        if label_allow and label not in label_allow:
            continue
        mention = Mention(
            mention_id=mention_id,
            sentence_index=_optional_int(row, options.sentence_index_field),
            start=_required_int(row, options.start_field),
            end=_required_int(row, options.end_field),
            label=label,
            text=(
                _optional_str(row, options.mention_text_field, "Mention text")
                if want_text
                else None
            ),
        )
        grouped.setdefault(document_id, []).append(mention)

    for mentions in grouped.values():
        mentions.sort(key=_order_key)
    return grouped


def _relation_row(
    document_id: str,
    relation: Any,
    options: Any,
    extractor: RelationExtractor,
) -> dict[str, Any]:
    subject = relation.subject
    obj = relation.object
    row: dict[str, Any] = {
        "relation_id": relation_id(
            document_id=document_id,
            subject_mention_id=subject.mention_id,
            object_mention_id=obj.mention_id,
            relation_type=relation.relation_type,
            method=extractor.method,
        ),
        "document_id": document_id,
        "subject_mention_id": subject.mention_id,
        "object_mention_id": obj.mention_id,
        "relation_type": relation.relation_type,
        "directed": relation.directed,
        "method": extractor.method,
        "status": relation.status,
        "confidence": relation.confidence,
        "sentence_index": relation.sentence_index,
        "subject_start": subject.start,
        "subject_end": subject.end,
        "object_start": obj.start,
        "object_end": obj.end,
        "subject_label": subject.label,
        "object_label": obj.label,
        "extractor": extractor.name,
        "extractor_version": extractor.version,
    }
    if options.include_mention_text:
        row["subject_text"] = subject.text
        row["object_text"] = obj.text
    return row


def _required_id(row: Mapping[str, Any], field_name: str, description: str) -> str:
    raw = row[field_name]
    if raw is None or not str(raw).strip():
        raise ValueError(
            f"{description} column '{field_name}' contains null or empty values"
        )
    return str(raw)


def _required_int(row: Mapping[str, Any], field_name: str) -> int:
    raw = row[field_name]
    if raw is None or isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(
            f"Column '{field_name}' must contain non-null integer offsets"
        )
    return raw


def _optional_int(row: Mapping[str, Any], field_name: str) -> int | None:
    raw = row[field_name]
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"Column '{field_name}' must contain integers or nulls")
    return raw


def _optional_str(
    row: Mapping[str, Any], field_name: str, description: str
) -> str | None:
    raw = row[field_name]
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError(f"{description} column '{field_name}' must contain strings or nulls")
    return raw
