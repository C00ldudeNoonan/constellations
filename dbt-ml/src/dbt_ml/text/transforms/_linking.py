"""Shared execution for the built-in entity-linking child-table transform.

The driver here owns mention identity, the passthrough/include projection,
ambiguity policy, and row shaping. The per-resolver matching (alias-table text
matching or vector similarity) is delegated to the resolver selected by the
``resolver`` option; see ``dbt_ml.text.linking``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from ...transforms import IncrementalContract, TransformContext
from ..linking import (
    EntityResolver,
    LinkStatus,
    entity_link_id,
    get_resolver,
    parse_entity_link_options,
)

_LINK_SCHEMA: dict[str, pl.DataType] = {
    "entity_link_id": pl.String(),
    "document_id": pl.String(),
    "mention_id": pl.String(),
    "entity_namespace": pl.String(),
    "canonical_id": pl.String(),
    "match_method": pl.String(),
    # Reserved for score-producing resolvers; the deterministic alias-table
    # resolver always emits null rather than implying a calibrated confidence.
    "match_score": pl.Float64(),
    "status": pl.String(),
    "resolver": pl.String(),
    "resolver_version": pl.String(),
    "alias_set_version": pl.String(),
}


@dataclass(frozen=True)
class _Mention:
    mention_id: str
    document_id: str
    text: Any
    signal: Any
    passthrough: dict[str, Any] = field(default_factory=dict)
    included: dict[str, Any] = field(default_factory=dict)


def validate_link_options(options: Mapping[str, Any]) -> None:
    parse_entity_link_options(options)


def declared_link_dependencies(options: Mapping[str, Any]) -> tuple[str, str]:
    """The mentions and alias models these options require. `_dep_frames`
    enforces the same pair at runtime; declaring it lets the compiler reject a
    misspelled or stale `depends_on` before any model is materialized."""
    parsed = parse_entity_link_options(options)
    return (parsed.mentions, parsed.aliases)


def declared_link_incremental_contract(options: Mapping[str, Any]) -> IncrementalContract:
    """Parents are documents in the `mentions` model; the `aliases` model is a
    whole-table reference input, so an alias/reference edit re-links every
    document (issue #218). Child rows are keyed by `entity_link_id`. This holds
    for every resolver, since the shared driver fixes those output columns."""
    parsed = parse_entity_link_options(options)
    return IncrementalContract(
        parent_key="document_id",
        child_key="entity_link_id",
        parent_source=parsed.mentions,
        parent_source_key=parsed.document_id_field,
        reference_deps=(parsed.aliases,),
    )


def run_links(
    deps: dict[str, pl.DataFrame],
    ctx: TransformContext,
) -> pl.DataFrame:
    options = parse_entity_link_options(ctx.options)
    resolver = get_resolver(options.resolver)
    mentions_frame, aliases_frame = _dep_frames(deps, options)
    resolver.validate_frames(mentions_frame, aliases_frame, options)
    mentions, schema = _input_mentions(mentions_frame, options, resolver)
    if not mentions:
        return pl.DataFrame(schema=schema)

    reference = resolver.build_reference(aliases_frame, options)
    reference_version = reference.fingerprint

    rows: list[dict[str, Any]] = []
    ambiguous_pairs: list[tuple[str, str]] = []
    for mention in mentions:
        resolutions = (
            reference.resolve(mention.signal) if mention.signal is not None else {}
        )
        if not resolutions:
            rows.append(
                _link_row(
                    mention,
                    options,
                    reference_version,
                    resolver.version,
                    namespace=None,
                    canonical_id=None,
                    method=None,
                    score=None,
                    status="unmatched",
                )
            )
            continue
        for namespace in sorted(resolutions):
            resolution = resolutions[namespace]
            if resolution.status == "ambiguous":
                ambiguous_pairs.append((mention.mention_id, namespace))
            for candidate in resolution.candidates:
                rows.append(
                    _link_row(
                        mention,
                        options,
                        reference_version,
                        resolver.version,
                        namespace=namespace,
                        canonical_id=candidate.canonical_id,
                        method=resolution.method,
                        score=candidate.score,
                        status=resolution.status,
                    )
                )

    if ambiguous_pairs and options.on_ambiguity == "error":
        shown = ", ".join(
            f"(mention {mention_id}, namespace {namespace})"
            for mention_id, namespace in ambiguous_pairs[:5]
        )
        raise ValueError(
            f"{len(ambiguous_pairs)} mention/namespace pairs are ambiguous and "
            f"on_ambiguity is 'error'; first: {shown}"
        )
    return pl.DataFrame(rows, schema=schema, strict=False)


def _dep_frames(
    deps: dict[str, pl.DataFrame],
    options: Any,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    expected = {options.mentions, options.aliases}
    if set(deps) != expected:
        raise ValueError(
            "Entity linking expects dependencies named by the `mentions` and "
            f"`aliases` options ({sorted(expected)}); got: {sorted(deps)}"
        )
    return deps[options.mentions], deps[options.aliases]


def _input_mentions(
    frame: pl.DataFrame,
    options: Any,
    resolver: EntityResolver,
) -> tuple[list[_Mention], dict[str, pl.DataType]]:
    text_needed = resolver.text_required() or options.include_mention_text
    required = {
        options.mention_id_field,
        options.document_id_field,
        *(({options.mention_text_field}) if text_needed else set()),
        *resolver.required_mention_columns(options),
    }
    passthrough_fields = {
        name: field_name
        for name, field_name in (
            ("label", options.label_field),
            ("start", options.start_field),
            ("end", options.end_field),
        )
        if field_name is not None
    }
    missing = sorted(
        {
            field_name
            for field_name in (*required, *passthrough_fields.values())
            if field_name not in frame.columns
        }
    )
    if missing:
        text_hint = (
            " If mention text is absent, the upstream NLP transform may need "
            "`include_text: true`."
            if text_needed and options.mention_text_field in missing
            else ""
        )
        raise ValueError(
            f"Mentions model '{options.mentions}' is missing configured columns "
            f"{missing}; got: {sorted(frame.columns)}.{text_hint}"
        )

    schema = dict(_LINK_SCHEMA)
    for output_name, field_name in passthrough_fields.items():
        schema[output_name] = frame.schema[field_name]
    if options.include_mention_text:
        schema["mention_text"] = pl.String()

    missing_fields = sorted(set(options.include_fields) - set(frame.columns))
    if missing_fields:
        raise ValueError(f"include_fields contains unknown columns: {missing_fields}")
    collisions = sorted(set(options.include_fields) & (set(schema) | {"mention_text"}))
    if collisions:
        raise ValueError(
            f"include_fields collides with entity-link output columns: {collisions}"
        )
    schema.update({field_name: frame.schema[field_name] for field_name in options.include_fields})

    has_text_column = options.mention_text_field in frame.columns
    mentions: list[_Mention] = []
    seen_ids: set[str] = set()
    for row in frame.iter_rows(named=True):
        mention_id = _required_id(row, options.mention_id_field, "Mention ID")
        if mention_id in seen_ids:
            raise ValueError(
                f"Mention ID column '{options.mention_id_field}' contains "
                f"duplicate value '{mention_id}'"
            )
        seen_ids.add(mention_id)
        document_id = _required_id(row, options.document_id_field, "Document ID")
        mentions.append(
            _Mention(
                mention_id=mention_id,
                document_id=document_id,
                text=row[options.mention_text_field] if has_text_column else None,
                signal=resolver.mention_signal(row, options),
                passthrough={
                    output_name: row[field_name]
                    for output_name, field_name in passthrough_fields.items()
                },
                included={
                    field_name: row[field_name] for field_name in options.include_fields
                },
            )
        )
    return mentions, schema


def _required_id(row: Mapping[str, Any], field_name: str, description: str) -> str:
    raw = row[field_name]
    if raw is None or not str(raw).strip():
        raise ValueError(
            f"{description} column '{field_name}' contains null or empty values"
        )
    return str(raw)


def _link_row(
    mention: _Mention,
    options: Any,
    reference_version: str,
    resolver_version: str,
    *,
    namespace: str | None,
    canonical_id: str | None,
    method: str | None,
    score: float | None,
    status: LinkStatus,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "entity_link_id": entity_link_id(
            mention_id=mention.mention_id,
            document_id=mention.document_id,
            entity_namespace=namespace,
            canonical_id=canonical_id,
            match_method=method,
            status=status,
        ),
        "document_id": mention.document_id,
        "mention_id": mention.mention_id,
        "entity_namespace": namespace,
        "canonical_id": canonical_id,
        "match_method": method,
        "match_score": score,
        "status": status,
        "resolver": options.resolver,
        "resolver_version": resolver_version,
        "alias_set_version": reference_version,
        **mention.passthrough,
        **mention.included,
    }
    if options.include_mention_text:
        row["mention_text"] = mention.text
    return row
