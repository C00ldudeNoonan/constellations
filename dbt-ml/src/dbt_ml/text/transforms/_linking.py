"""Shared execution for the built-in entity-linking child-table transform."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import polars as pl

from ...transforms import TransformContext
from ..linking import (
    ALIAS_RESOLVER_VERSION,
    EntityLinkOptions,
    LinkStatus,
    alias_set_fingerprint,
    entity_link_id,
    normalize_alias_text,
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
    text: str | None
    passthrough: dict[str, Any]
    included: dict[str, Any]


def validate_link_options(options: Mapping[str, Any]) -> None:
    EntityLinkOptions.model_validate(options)


def run_links(
    deps: dict[str, pl.DataFrame],
    ctx: TransformContext,
) -> pl.DataFrame:
    options = EntityLinkOptions.model_validate(ctx.options)
    mentions_frame, aliases_frame = _dep_frames(deps, options)
    mentions, schema = _input_mentions(mentions_frame, options)
    if not mentions:
        return pl.DataFrame(schema=schema)

    alias_rows = _alias_rows(aliases_frame, options)
    alias_version = alias_set_fingerprint(alias_rows)
    lookups = _alias_lookups(alias_rows, options)

    rows: list[dict[str, Any]] = []
    ambiguous_pairs: list[tuple[str, str]] = []
    for mention in mentions:
        resolved = _resolve(mention.text, options, lookups)
        if not resolved:
            rows.append(
                _link_row(mention, options, alias_version, None, None, None, "unmatched")
            )
            continue
        for namespace in sorted(resolved):
            method, canonical_ids = resolved[namespace]
            status: LinkStatus = "matched" if len(canonical_ids) == 1 else "ambiguous"
            if status == "ambiguous":
                ambiguous_pairs.append((mention.mention_id, namespace))
            for canonical_id in sorted(canonical_ids):
                rows.append(
                    _link_row(
                        mention,
                        options,
                        alias_version,
                        namespace,
                        canonical_id,
                        method,
                        status,
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
    options: EntityLinkOptions,
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
    options: EntityLinkOptions,
) -> tuple[list[_Mention], dict[str, pl.DataType]]:
    required = {
        "mention_id_field": options.mention_id_field,
        "document_id_field": options.document_id_field,
        "mention_text_field": options.mention_text_field,
    }
    passthrough_fields = {
        name: field
        for name, field in (
            ("label", options.label_field),
            ("start", options.start_field),
            ("end", options.end_field),
        )
        if field is not None
    }
    missing = sorted(
        {
            field
            for field in (*required.values(), *passthrough_fields.values())
            if field not in frame.columns
        }
    )
    if missing:
        raise ValueError(
            f"Mentions model '{options.mentions}' is missing configured columns "
            f"{missing}; got: {sorted(frame.columns)}. If mention text is absent, "
            "the upstream NLP transform may need `include_text: true`."
        )

    schema = dict(_LINK_SCHEMA)
    for output_name, field in passthrough_fields.items():
        schema[output_name] = frame.schema[field]
    if options.include_mention_text:
        schema["mention_text"] = pl.String()

    missing_fields = sorted(set(options.include_fields) - set(frame.columns))
    if missing_fields:
        raise ValueError(f"include_fields contains unknown columns: {missing_fields}")
    collisions = sorted(
        set(options.include_fields) & (set(schema) | {"mention_text"})
    )
    if collisions:
        raise ValueError(
            f"include_fields collides with entity-link output columns: {collisions}"
        )
    schema.update({field: frame.schema[field] for field in options.include_fields})

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

        raw_text = row[options.mention_text_field]
        if raw_text is not None and not isinstance(raw_text, str):
            raise ValueError(
                f"Mention text column '{options.mention_text_field}' must contain "
                "strings or nulls"
            )
        mentions.append(
            _Mention(
                mention_id=mention_id,
                document_id=document_id,
                text=raw_text,
                passthrough={
                    output_name: row[field]
                    for output_name, field in passthrough_fields.items()
                },
                included={field: row[field] for field in options.include_fields},
            )
        )
    return mentions, schema


def _required_id(row: Mapping[str, Any], field: str, description: str) -> str:
    raw = row[field]
    if raw is None or not str(raw).strip():
        raise ValueError(f"{description} column '{field}' contains null or empty values")
    return str(raw)


def _alias_rows(
    frame: pl.DataFrame,
    options: EntityLinkOptions,
) -> list[dict[str, str]]:
    required = (
        options.alias_text_field,
        options.namespace_field,
        options.canonical_id_field,
    )
    missing = sorted({field for field in required if field not in frame.columns})
    if missing:
        raise ValueError(
            f"Alias model '{options.aliases}' is missing configured columns "
            f"{missing}; got: {sorted(frame.columns)}"
        )

    rows: list[dict[str, str]] = []
    for position, row in enumerate(frame.iter_rows(named=True)):
        values: dict[str, str] = {}
        for output_name, field in (
            ("alias", options.alias_text_field),
            ("entity_namespace", options.namespace_field),
            ("canonical_id", options.canonical_id_field),
        ):
            raw = row[field]
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(
                    f"Alias model '{options.aliases}' column '{field}' must contain "
                    f"non-empty strings (row {position})"
                )
            values[output_name] = raw
        rows.append(values)
    return rows


def _alias_lookups(
    alias_rows: list[dict[str, str]],
    options: EntityLinkOptions,
) -> dict[str, dict[str, dict[str, set[str]]]]:
    """Per-method match key → namespace → canonical IDs, built in one pass so
    candidate lookup during resolution is dictionary access, not per-mention
    scans of the alias table."""
    lookups: dict[str, dict[str, dict[str, set[str]]]] = {
        method: {} for method in options.match_methods
    }
    for row in alias_rows:
        for method, table in lookups.items():
            key = row["alias"] if method == "exact" else normalize_alias_text(row["alias"])
            table.setdefault(key, {}).setdefault(row["entity_namespace"], set()).add(
                row["canonical_id"]
            )
    return lookups


def _resolve(
    text: str | None,
    options: EntityLinkOptions,
    lookups: dict[str, dict[str, dict[str, set[str]]]],
) -> dict[str, tuple[str, set[str]]]:
    """Namespace → (method, canonical IDs). Methods run in configured order and
    the first method producing candidates for a namespace wins that namespace;
    later methods can still contribute other namespaces."""
    resolved: dict[str, tuple[str, set[str]]] = {}
    if text is None or not text.strip():
        return resolved
    for method in options.match_methods:
        key = text if method == "exact" else normalize_alias_text(text)
        hits = lookups[method].get(key)
        if not hits:
            continue
        for namespace, canonical_ids in hits.items():
            if namespace not in resolved:
                resolved[namespace] = (method, canonical_ids)
    return resolved


def _link_row(
    mention: _Mention,
    options: EntityLinkOptions,
    alias_version: str,
    namespace: str | None,
    canonical_id: str | None,
    method: str | None,
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
        "match_score": None,
        "status": status,
        "resolver": options.resolver,
        "resolver_version": ALIAS_RESOLVER_VERSION,
        "alias_set_version": alias_version,
        **mention.passthrough,
        **mention.included,
    }
    if options.include_mention_text:
        row["mention_text"] = mention.text
    return row
