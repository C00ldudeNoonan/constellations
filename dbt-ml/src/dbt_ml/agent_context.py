from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import polars as pl

from .hashing import HASH_DIGEST_SIZE, canonical_fingerprint, canonical_json

AGENT_CONTEXT_CONTRACT = "agent_context/v1"
_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class AgentContextGrain(StrEnum):
    DOCUMENT_REGISTRY = "document_registry"
    DOCUMENT_CHUNKS = "document_chunks"
    CONTEXT_ENTITY_LINKS = "context_entity_links"


class FreshnessStatus(StrEnum):
    FRESH = "fresh"
    SOURCE_STALE = "source_stale"
    PIPELINE_STALE = "pipeline_stale"
    UNKNOWN = "unknown"


class AgentContextValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ContractField:
    name: str
    data_type: str
    nullable: bool
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "data_type": self.data_type,
            "nullable": self.nullable,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class ContractRelation:
    grain: AgentContextGrain
    primary_key: tuple[str, ...]
    foreign_keys: Mapping[str, str]
    fields: tuple[ContractField, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": AGENT_CONTEXT_CONTRACT,
            "grain": self.grain.value,
            "primary_key": list(self.primary_key),
            "foreign_keys": dict(self.foreign_keys),
            "fields": [field.to_dict() for field in self.fields],
        }


def _field(
    name: str,
    data_type: str,
    *,
    nullable: bool = False,
    description: str,
) -> ContractField:
    return ContractField(name, data_type, nullable, description)


_TEMPORAL_FIELDS = (
    _field(
        "validity_known",
        "boolean",
        description="Whether business-time validity is known.",
    ),
    _field(
        "valid_from",
        "timestamp",
        nullable=True,
        description="Inclusive UTC business-time boundary.",
    ),
    _field(
        "valid_to",
        "timestamp",
        nullable=True,
        description="Exclusive UTC business-time boundary; null is open-ended.",
    ),
    _field(
        "recorded_from",
        "timestamp",
        description="Inclusive UTC system-time boundary.",
    ),
    _field(
        "recorded_to",
        "timestamp",
        nullable=True,
        description="Exclusive UTC system-time boundary; null is current.",
    ),
)

_POLICY_FIELDS = (
    _field(
        "tenant_id",
        "string",
        nullable=True,
        description="Trusted tenant partition identifier.",
    ),
    _field(
        "is_public",
        "boolean",
        description="Whether the record is explicitly public.",
    ),
    _field(
        "access_groups",
        "array[string]",
        description="Canonical set of trusted access-group identifiers.",
    ),
    _field(
        "classification",
        "string",
        nullable=True,
        description="Optional classification label.",
    ),
    _field(
        "policy_ref",
        "string",
        nullable=True,
        description="Safe identifier for the governing policy.",
    ),
    _field(
        "policy_version",
        "string",
        nullable=True,
        description="Safe policy version identifier.",
    ),
    _field(
        "authorization_resolved",
        "boolean",
        description="Whether policy metadata was resolved from a trusted source.",
    ),
)

_FRESHNESS_FIELDS = (
    _field(
        "source_updated_at",
        "timestamp",
        nullable=True,
        description="UTC update time reported by the source.",
    ),
    _field(
        "source_observed_at",
        "timestamp",
        nullable=True,
        description="UTC time at which the source version was observed.",
    ),
    _field(
        "ingested_at",
        "timestamp",
        description="UTC ingestion time.",
    ),
    _field(
        "materialized_at",
        "timestamp",
        description="UTC materialization time.",
    ),
    _field(
        "freshness_checked_at",
        "timestamp",
        nullable=True,
        description="UTC time of the latest successful freshness check.",
    ),
    _field(
        "refresh_due_at",
        "timestamp",
        nullable=True,
        description="UTC deadline for the next successful pipeline refresh.",
    ),
    _field(
        "stale_after",
        "timestamp",
        nullable=True,
        description="UTC time after which the source version is stale.",
    ),
)

# Fields a document_chunks row must carry verbatim from its parent
# document_registry row (validate_agent_context_relations enforces they match
# via policy_fingerprint equality).
_DOCUMENT_CARRY_FIELDS: tuple[str, ...] = tuple(
    field.name for field in (*_TEMPORAL_FIELDS, *_POLICY_FIELDS, *_FRESHNESS_FIELDS)
)

_PROVENANCE_FIELDS = (
    _field(
        "upstream_unique_id",
        "string",
        description="Safe upstream dbt or dbt-ml resource unique_id.",
    ),
    _field(
        "invocation_id",
        "string",
        description="Safe materialization invocation identifier.",
    ),
    _field(
        "parser_identity",
        "string",
        nullable=True,
        description="Safe parser implementation and version identity.",
    ),
    _field(
        "transform_identity",
        "string",
        nullable=True,
        description="Safe transform implementation and version identity.",
    ),
    _field(
        "prompt_fingerprint",
        "string",
        nullable=True,
        description="One-way prompt identity; never prompt text.",
    ),
    _field(
        "schema_fingerprint",
        "string",
        nullable=True,
        description="One-way extraction schema identity.",
    ),
    _field(
        "provider_identity",
        "string",
        nullable=True,
        description="Safe provider contract and implementation identity.",
    ),
    _field(
        "model_identity",
        "string",
        nullable=True,
        description="Safe provider model identifier.",
    ),
    _field(
        "provenance_fingerprint",
        "string",
        description="One-way identity of the complete safe provenance chain.",
    ),
)

_DOCUMENT_REGISTRY = ContractRelation(
    grain=AgentContextGrain.DOCUMENT_REGISTRY,
    primary_key=("document_version_id",),
    foreign_keys={},
    fields=(
        _field(
            "document_id",
            "string",
            description="Stable logical document identity across source versions.",
        ),
        _field(
            "document_version_id",
            "string",
            description="Immutable identity of one source/content version.",
        ),
        _field(
            "source_system",
            "string",
            description="Stable source-system namespace.",
        ),
        _field(
            "source_key",
            "string",
            description="Stable logical key within the source system.",
        ),
        _field(
            "source_uri",
            "string",
            nullable=True,
            description="Human-resolvable source URI when safe to retain.",
        ),
        _field(
            "source_version",
            "string",
            description="Immutable source-native version identifier.",
        ),
        _field(
            "source_content_hash",
            "string",
            description="One-way hash of canonical source content.",
        ),
        *_TEMPORAL_FIELDS,
        *_POLICY_FIELDS,
        *_FRESHNESS_FIELDS,
        *_PROVENANCE_FIELDS,
    ),
)

_DOCUMENT_CHUNKS = ContractRelation(
    grain=AgentContextGrain.DOCUMENT_CHUNKS,
    primary_key=("context_id",),
    foreign_keys={"document_version_id": "document_registry.document_version_id"},
    fields=(
        _field(
            "context_id",
            "string",
            description="Stable retrieval-record identity for this document version.",
        ),
        _field(
            "chunk_id",
            "string",
            description="Stable identity of canonical chunk content and locator.",
        ),
        _field(
            "document_id",
            "string",
            description="Stable logical parent document identity.",
        ),
        _field(
            "document_version_id",
            "string",
            description="Immutable parent document-version identity.",
        ),
        _field(
            "chunk_index",
            "integer",
            description="Zero-based canonical chunk ordinal within the document.",
        ),
        _field("text", "string", description="Canonical indexable chunk text."),
        _field(
            "chunk_content_hash",
            "string",
            description="One-way hash of canonical chunk text.",
        ),
        _field(
            "source_uri",
            "string",
            nullable=True,
            description="Human-resolvable parent source URI.",
        ),
        _field(
            "citation_page_number",
            "integer",
            nullable=True,
            description="One-based source page number.",
        ),
        _field(
            "citation_section_path",
            "array[string]",
            nullable=True,
            description="Ordered heading path within the source.",
        ),
        _field(
            "citation_char_start",
            "integer",
            nullable=True,
            description="Inclusive character offset in the canonical source text.",
        ),
        _field(
            "citation_char_end",
            "integer",
            nullable=True,
            description="Exclusive character offset in the canonical source text.",
        ),
        _field(
            "citation_speaker",
            "string",
            nullable=True,
            description="Speaker identity for transcript context.",
        ),
        _field(
            "citation_start_seconds",
            "float",
            nullable=True,
            description="Inclusive transcript/audio offset in seconds.",
        ),
        _field(
            "citation_end_seconds",
            "float",
            nullable=True,
            description="Exclusive transcript/audio offset in seconds.",
        ),
        _field(
            "citation_locator",
            "json",
            nullable=True,
            description="Supplemental typed locator metadata.",
        ),
        *_TEMPORAL_FIELDS,
        *_POLICY_FIELDS,
        *_FRESHNESS_FIELDS,
        _field(
            "chunker_identity",
            "string",
            description="Safe chunker implementation and config identity.",
        ),
        *_PROVENANCE_FIELDS,
    ),
)

_CONTEXT_ENTITY_LINKS = ContractRelation(
    grain=AgentContextGrain.CONTEXT_ENTITY_LINKS,
    primary_key=("context_entity_link_id",),
    foreign_keys={"context_id": "document_chunks.context_id"},
    fields=(
        _field(
            "context_entity_link_id",
            "string",
            description="Stable identity of one context-to-entity relationship.",
        ),
        _field(
            "context_id",
            "string",
            description="Referenced document/chunk context identity.",
        ),
        _field(
            "entity_namespace",
            "string",
            description="Namespace shared with the dbt entity contract.",
        ),
        _field(
            "entity_name",
            "string",
            description="dbt Semantic Layer-compatible entity name.",
        ),
        _field(
            "entity_id",
            "string",
            description="Stable namespaced identity of the typed entity key.",
        ),
        _field(
            "entity_key",
            "json",
            description="Canonical type-preserving serialized entity key.",
        ),
        _field(
            "dbt_unique_id",
            "string",
            nullable=True,
            description="Originating dbt resource unique_id when available.",
        ),
        _field(
            "relationship_type",
            "string",
            description="Stable relationship semantic, such as applies_to.",
        ),
        _field(
            "link_method",
            "string",
            description="Deterministic or inferred link method identity.",
        ),
        _field(
            "confidence",
            "float",
            nullable=True,
            description="Optional inferred-link confidence from 0 through 1.",
        ),
        _field(
            "recorded_from",
            "timestamp",
            description="Inclusive UTC system-time boundary.",
        ),
        _field(
            "recorded_to",
            "timestamp",
            nullable=True,
            description="Exclusive UTC system-time boundary; null is current.",
        ),
        _field(
            "link_provenance_fingerprint",
            "string",
            description="One-way identity of link derivation provenance.",
        ),
    ),
)

_RELATIONS = {
    relation.grain: relation
    for relation in (_DOCUMENT_REGISTRY, _DOCUMENT_CHUNKS, _CONTEXT_ENTITY_LINKS)
}


def contract_relation(grain: AgentContextGrain | str) -> ContractRelation:
    try:
        normalized = AgentContextGrain(grain)
    except ValueError:
        raise AgentContextValidationError(
            f"Unknown {AGENT_CONTEXT_CONTRACT} grain '{grain}'"
        ) from None
    return _RELATIONS[normalized]


def contract_descriptor(grain: AgentContextGrain | str) -> dict[str, Any]:
    return contract_relation(grain).to_dict()


def empty_agent_context_frame(
    grain: AgentContextGrain | str,
) -> pl.DataFrame:
    relation = contract_relation(grain)
    return pl.DataFrame([_empty_series(field) for field in relation.fields])


def _empty_series(field: ContractField) -> pl.Series:
    if field.data_type in {"string", "json"}:
        return pl.Series(field.name, [], dtype=pl.String)
    if field.data_type == "integer":
        return pl.Series(field.name, [], dtype=pl.Int64)
    if field.data_type == "float":
        return pl.Series(field.name, [], dtype=pl.Float64)
    if field.data_type == "boolean":
        return pl.Series(field.name, [], dtype=pl.Boolean)
    if field.data_type == "timestamp":
        return pl.Series(
            field.name,
            [],
            dtype=pl.Datetime(time_unit="us", time_zone="UTC"),
        )
    if field.data_type == "array[string]":
        return pl.Series(field.name, [], dtype=pl.List(pl.String))
    raise AssertionError(f"Unsupported agent context type {field.data_type}")


def make_document_id(source_system: str, source_key: str) -> str:
    return canonical_fingerprint(
        {
            "source_system": _non_empty(source_system, "source_system"),
            "source_key": _non_empty(source_key, "source_key"),
        },
        domain="dbt-ml-agent-context-document",
    )


def make_document_version_id(
    document_id: str,
    source_version: str,
    source_content_hash: str,
) -> str:
    return canonical_fingerprint(
        {
            "document_id": _stable_id(document_id, "document_id"),
            "source_version": _non_empty(source_version, "source_version"),
            "source_content_hash": _non_empty(source_content_hash, "source_content_hash"),
        },
        domain="dbt-ml-agent-context-document-version",
    )


def make_chunk_id(document_id: str, chunk_index: int, text: str) -> str:
    document_id = _stable_id(document_id, "document_id")
    if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
        raise ValueError("chunk_index must be a non-negative integer")
    if not isinstance(text, str) or not text:
        raise ValueError("text must be a non-empty string")
    digest = hashlib.blake2b(digest_size=HASH_DIGEST_SIZE)
    digest.update(f"{document_id}|{chunk_index}|".encode())
    digest.update(text.encode())
    return digest.hexdigest()


def make_context_id(document_version_id: str, chunk_id: str) -> str:
    return canonical_fingerprint(
        {
            "document_version_id": _stable_id(document_version_id, "document_version_id"),
            "chunk_id": _stable_id(chunk_id, "chunk_id"),
        },
        domain="dbt-ml-agent-context-record",
    )


def canonical_entity_key(value: Any) -> str:
    return canonical_json(value)


def make_entity_id(
    entity_namespace: str,
    entity_name: str,
    entity_key: str,
) -> str:
    entity_key = _canonical_json_text(entity_key, "entity_key")
    return canonical_fingerprint(
        {
            "entity_namespace": _non_empty(entity_namespace, "entity_namespace"),
            "entity_name": _non_empty(entity_name, "entity_name"),
            "entity_key": entity_key,
        },
        domain="dbt-ml-agent-context-entity",
    )


def make_context_entity_link_id(
    context_id: str,
    entity_id: str,
    relationship_type: str,
) -> str:
    return canonical_fingerprint(
        {
            "context_id": _stable_id(context_id, "context_id"),
            "entity_id": _stable_id(entity_id, "entity_id"),
            "relationship_type": _non_empty(relationship_type, "relationship_type"),
        },
        domain="dbt-ml-agent-context-entity-link",
    )


def entity_link_method(resolver: str, resolver_version: str) -> str:
    """Canonical ``link_method`` identity for a ``link_entities`` resolver, e.g.
    ``entity_link:alias_table:1``. Recording the resolver and its version keeps
    the governed link's derivation auditable and invalidates the projected row
    when either changes."""
    return (
        f"entity_link:{_non_empty(resolver, 'resolver')}:"
        f"{_non_empty(resolver_version, 'resolver_version')}"
    )


def project_entity_link(
    *,
    context_id: str,
    entity_namespace: str,
    entity_name: str,
    canonical_id: str,
    relationship_type: str,
    link_method: str,
    recorded_from: datetime,
    confidence: float | None = None,
    dbt_unique_id: str | None = None,
    recorded_to: datetime | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project one resolved entity link — a ``canonical_id`` produced by the
    ``link_entities`` transform — into a governed ``context_entity_links`` row
    (issue #145).

    The ``canonical_id`` becomes the row's ``entity_key``, so a governed metric
    that keys the same ``(entity_namespace, entity_name, canonical_id)`` resolves
    to the *identical* ``entity_id``. That shared id is the join key that lets an
    agent combine documentary evidence with structured metrics across the two MCP
    planes (issues #132/#147) without either side understanding the other's
    schema.

    Pass only resolved links: ``canonical_id`` must be non-empty, so an
    ``unmatched`` mention (which carries no canonical key) is never published to
    the governed context. Callers typically also drop ``ambiguous`` rows, or keep
    them only with a caller-chosen ``relationship_type`` — this helper does not
    guess. When ``provenance`` is omitted, a canonical fingerprint over the
    link's identity, relationship, and method is recorded.
    """
    canonical = _non_empty(canonical_id, "canonical_id")
    entity_key = canonical_entity_key(canonical)
    entity_id = make_entity_id(entity_namespace, entity_name, entity_key)
    method = _non_empty(link_method, "link_method")
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if provenance is None:
        provenance = {
            "context_id": context_id,
            "entity_id": entity_id,
            "relationship_type": relationship_type,
            "link_method": method,
        }
    return {
        "context_entity_link_id": make_context_entity_link_id(
            context_id, entity_id, relationship_type
        ),
        "context_id": context_id,
        "entity_namespace": entity_namespace,
        "entity_name": entity_name,
        "entity_id": entity_id,
        "entity_key": entity_key,
        "dbt_unique_id": dbt_unique_id,
        "relationship_type": relationship_type,
        "link_method": method,
        "confidence": confidence,
        "recorded_from": recorded_from,
        "recorded_to": recorded_to,
        "link_provenance_fingerprint": make_provenance_fingerprint(dict(provenance)),
    }


def project_document_registry_row(
    *,
    text: str,
    source_system: str,
    source_key: str,
    source_version: str,
    upstream_unique_id: str,
    invocation_id: str,
    recorded_from: datetime,
    ingested_at: datetime,
    materialized_at: datetime,
    source_uri: str | None = None,
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
    recorded_to: datetime | None = None,
    tenant_id: str | None = None,
    is_public: bool = False,
    access_groups: Iterable[str] | str = (),
    classification: str | None = None,
    policy_ref: str | None = None,
    policy_version: str | None = None,
    authorization_resolved: bool = False,
    source_updated_at: datetime | None = None,
    source_observed_at: datetime | None = None,
    freshness_checked_at: datetime | None = None,
    refresh_due_at: datetime | None = None,
    stale_after: datetime | None = None,
    parser_identity: str | None = None,
    transform_identity: str | None = None,
    prompt_fingerprint: str | None = None,
    schema_fingerprint: str | None = None,
    provider_identity: str | None = None,
    model_identity: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project a single extracted document's fields into a `document_registry`
    row (issue #300). Computes `document_id`, `document_version_id`,
    `source_content_hash`, and `provenance_fingerprint` with this module's own
    id algorithms, so the result validates against
    `validate_agent_context_frame`. `is_public`/`authorization_resolved`/
    `access_groups` default deny-closed, matching the contract's policy
    semantics; this is a pure projector, not a second validator —
    `validate_agent_context_frame` still rejects an inconsistent combination.

    Extraction's own built-in `document_id`/`content_hash` columns use a
    different algorithm than this contract (they key incremental state and
    cannot be repointed), so this helper always derives its own
    `document_id`/`source_content_hash` from `source_system`/`source_key`/
    `text` rather than accepting extraction's columns directly.

    `access_groups` accepts either a list/tuple of group names or a
    JSON-array-encoded string (`_string_array` handles both) — the built-in
    extraction pipeline can scalarize a source list column to the latter
    shape, and iterating a raw string character-by-character would silently
    corrupt authorization data rather than raise.
    """
    document_id = make_document_id(source_system, source_key)
    source_content_hash = content_hash(text)
    document_version_id = make_document_version_id(
        document_id, source_version, source_content_hash
    )
    normalized_groups = tuple(sorted(set(_string_array(access_groups, allow_none=False))))
    if provenance is None:
        provenance = {
            "document_id": document_id,
            "source_version": source_version,
            "source_content_hash": source_content_hash,
            "upstream_unique_id": upstream_unique_id,
            "invocation_id": invocation_id,
            "parser_identity": parser_identity,
            "transform_identity": transform_identity,
            "prompt_fingerprint": prompt_fingerprint,
            "schema_fingerprint": schema_fingerprint,
            "provider_identity": provider_identity,
            "model_identity": model_identity,
        }
    return {
        "document_id": document_id,
        "document_version_id": document_version_id,
        "source_system": source_system,
        "source_key": source_key,
        "source_uri": source_uri,
        "source_version": source_version,
        "source_content_hash": source_content_hash,
        "validity_known": valid_from is not None,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "recorded_from": recorded_from,
        "recorded_to": recorded_to,
        "tenant_id": tenant_id,
        "is_public": is_public,
        "access_groups": normalized_groups,
        "classification": classification,
        "policy_ref": policy_ref,
        "policy_version": policy_version,
        "authorization_resolved": authorization_resolved,
        "source_updated_at": source_updated_at,
        "source_observed_at": source_observed_at,
        "ingested_at": ingested_at,
        "materialized_at": materialized_at,
        "freshness_checked_at": freshness_checked_at,
        "refresh_due_at": refresh_due_at,
        "stale_after": stale_after,
        "upstream_unique_id": upstream_unique_id,
        "invocation_id": invocation_id,
        "parser_identity": parser_identity,
        "transform_identity": transform_identity,
        "prompt_fingerprint": prompt_fingerprint,
        "schema_fingerprint": schema_fingerprint,
        "provider_identity": provider_identity,
        "model_identity": model_identity,
        "provenance_fingerprint": make_provenance_fingerprint(dict(provenance)),
    }


def project_document_chunk_row(
    document_registry_row: Mapping[str, Any],
    *,
    chunk_index: int,
    text: str,
    upstream_unique_id: str,
    invocation_id: str,
    chunker_identity: str,
    source_uri: str | None = None,
    citation_page_number: int | None = None,
    citation_section_path: Sequence[str] | None = None,
    citation_char_start: int | None = None,
    citation_char_end: int | None = None,
    citation_speaker: str | None = None,
    citation_start_seconds: float | None = None,
    citation_end_seconds: float | None = None,
    citation_locator: Mapping[str, Any] | None = None,
    parser_identity: str | None = None,
    transform_identity: str | None = None,
    prompt_fingerprint: str | None = None,
    schema_fingerprint: str | None = None,
    provider_identity: str | None = None,
    model_identity: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project one chunk of a `document_registry_row` (as returned by
    `project_document_registry_row`, or any mapping shaped like one — e.g. one
    row from that model's materialized output) into a `document_chunks` row
    (issue #300). Computes `chunk_id`, `context_id`, and `chunk_content_hash`
    with this module's own id algorithms, and copies every bitemporal/policy/
    freshness field from `document_registry_row` verbatim, so the
    cross-relation equality `validate_agent_context_relations` requires holds
    by construction rather than by caller discipline.

    `chunk_index` alone always satisfies the "citation locator must not be
    empty" rule (`citation_locator()` always includes it), so every
    `citation_*` argument may be left `None` — no character-offset tracking in
    the chunker is required for a chunk row built this way to validate.
    """
    document_id = document_registry_row["document_id"]
    document_version_id = document_registry_row["document_version_id"]
    chunk_id = make_chunk_id(document_id, chunk_index, text)
    context_id = make_context_id(document_version_id, chunk_id)
    resolved_source_uri = (
        source_uri if source_uri is not None else document_registry_row.get("source_uri")
    )
    if provenance is None:
        provenance = {
            "document_version_id": document_version_id,
            # chunk_id is itself a hash of (document_id, chunk_index, text), so
            # including it makes the fingerprint vary with chunk content too.
            "chunk_id": chunk_id,
            "chunker_identity": chunker_identity,
            "upstream_unique_id": upstream_unique_id,
            "invocation_id": invocation_id,
            "parser_identity": parser_identity,
            "transform_identity": transform_identity,
            "prompt_fingerprint": prompt_fingerprint,
            "schema_fingerprint": schema_fingerprint,
            "provider_identity": provider_identity,
            "model_identity": model_identity,
        }
    row: dict[str, Any] = {
        name: document_registry_row[name] for name in _DOCUMENT_CARRY_FIELDS
    }
    row.update(
        {
            "context_id": context_id,
            "chunk_id": chunk_id,
            "document_id": document_id,
            "document_version_id": document_version_id,
            "chunk_index": chunk_index,
            "text": text,
            "chunk_content_hash": content_hash(text),
            "source_uri": resolved_source_uri,
            "citation_page_number": citation_page_number,
            "citation_section_path": (
                list(citation_section_path) if citation_section_path is not None else None
            ),
            "citation_char_start": citation_char_start,
            "citation_char_end": citation_char_end,
            "citation_speaker": citation_speaker,
            "citation_start_seconds": citation_start_seconds,
            "citation_end_seconds": citation_end_seconds,
            "citation_locator": (
                json.dumps(
                    dict(citation_locator),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                if citation_locator is not None
                else None
            ),
            "chunker_identity": chunker_identity,
            "upstream_unique_id": upstream_unique_id,
            "invocation_id": invocation_id,
            "parser_identity": parser_identity,
            "transform_identity": transform_identity,
            "prompt_fingerprint": prompt_fingerprint,
            "schema_fingerprint": schema_fingerprint,
            "provider_identity": provider_identity,
            "model_identity": model_identity,
            "provenance_fingerprint": make_provenance_fingerprint(dict(provenance)),
        }
    )
    return row


def content_hash(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("content must be a string")
    return hashlib.blake2b(text.encode("utf-8"), digest_size=HASH_DIGEST_SIZE).hexdigest()


def make_provenance_fingerprint(values: Mapping[str, Any]) -> str:
    if not values:
        raise ValueError("provenance values must not be empty")
    return canonical_fingerprint(dict(values), domain="dbt-ml-agent-context-provenance")


def policy_fingerprint(row: Mapping[str, Any]) -> str:
    values = {field.name: row.get(field.name) for field in _POLICY_FIELDS}
    values["access_groups"] = list(_string_array(row.get("access_groups"), allow_none=True))
    return canonical_fingerprint(
        values,
        domain="dbt-ml-agent-context-policy",
    )


def retrieval_projection_fingerprint(row: Mapping[str, Any]) -> str:
    names = {
        "context_id",
        "chunk_id",
        "document_id",
        "document_version_id",
        "text",
        "chunk_content_hash",
        "source_uri",
        "provenance_fingerprint",
        *(field.name for field in _TEMPORAL_FIELDS),
        *(field.name for field in _POLICY_FIELDS),
        *(field.name for field in _FRESHNESS_FIELDS),
        "citation_page_number",
        "citation_section_path",
        "citation_char_start",
        "citation_char_end",
        "citation_speaker",
        "citation_start_seconds",
        "citation_end_seconds",
        "citation_locator",
    }
    values = {name: row.get(name) for name in sorted(names)}
    values["access_groups"] = list(_string_array(row.get("access_groups"), allow_none=True))
    for name in ("citation_section_path", "citation_locator"):
        values[name] = _json_value(values[name])
    return canonical_fingerprint(
        values,
        domain="dbt-ml-agent-context-retrieval-projection",
    )


def is_publishable_context(row: Mapping[str, Any]) -> bool:
    """Return whether a record has enough trusted policy data to be indexed."""
    if row.get("authorization_resolved") is not True:
        return False
    if row.get("is_public") is True:
        return True
    groups = _string_array(row.get("access_groups"), allow_none=True) or ()
    return any(
        (
            _optional_non_empty(row.get("tenant_id")),
            bool(groups),
            _optional_non_empty(row.get("classification")),
            _optional_non_empty(row.get("policy_ref")),
        )
    )


def freshness_status(row: Mapping[str, Any], *, now: datetime | None = None) -> FreshnessStatus:
    now = _utc(now or datetime.now(UTC), "now")
    refresh_due = _optional_utc(row.get("refresh_due_at"), "refresh_due_at")
    checked = _optional_utc(row.get("freshness_checked_at"), "freshness_checked_at")
    if refresh_due is not None and now >= refresh_due:
        return FreshnessStatus.PIPELINE_STALE
    stale_after = _optional_utc(row.get("stale_after"), "stale_after")
    if stale_after is not None and now >= stale_after:
        return FreshnessStatus.SOURCE_STALE
    if checked is None:
        return FreshnessStatus.UNKNOWN
    return FreshnessStatus.FRESH


def interval_contains(
    instant: datetime,
    start: datetime,
    end: datetime | None,
) -> bool:
    instant = _utc(instant, "instant")
    start = _utc(start, "start")
    end = _optional_utc(end, "end")
    return start <= instant and (end is None or instant < end)


def active_as_of(
    rows: Iterable[Mapping[str, Any]],
    *,
    valid_at: datetime,
    recorded_at: datetime,
    include_unknown_validity: bool = False,
) -> tuple[Mapping[str, Any], ...]:
    selected: list[Mapping[str, Any]] = []
    for row in rows:
        known = row.get("validity_known") is True
        valid = (
            include_unknown_validity
            if not known
            else interval_contains(
                valid_at,
                _utc(row.get("valid_from"), "valid_from"),
                _optional_utc(row.get("valid_to"), "valid_to"),
            )
        )
        recorded = interval_contains(
            recorded_at,
            _utc(row.get("recorded_from"), "recorded_from"),
            _optional_utc(row.get("recorded_to"), "recorded_to"),
        )
        if valid and recorded:
            selected.append(row)
    return tuple(
        sorted(
            selected,
            key=lambda row: (
                str(row.get("document_id", "")),
                str(row.get("document_version_id", "")),
                str(row.get("context_id", "")),
            ),
        )
    )


def citation_locator(row: Mapping[str, Any]) -> dict[str, Any]:
    locator = {
        "source_uri": row.get("source_uri"),
        "chunk_index": row.get("chunk_index"),
        "page_number": row.get("citation_page_number"),
        "section_path": _json_value(row.get("citation_section_path")),
        "char_start": row.get("citation_char_start"),
        "char_end": row.get("citation_char_end"),
        "speaker": row.get("citation_speaker"),
        "start_seconds": row.get("citation_start_seconds"),
        "end_seconds": row.get("citation_end_seconds"),
        "extra": _json_value(row.get("citation_locator")),
    }
    return {key: value for key, value in locator.items() if value is not None}


def validate_agent_context_frame(
    frame: pl.DataFrame,
    grain: AgentContextGrain | str,
) -> None:
    relation = contract_relation(grain)
    missing = [field.name for field in relation.fields if field.name not in frame]
    if missing:
        raise AgentContextValidationError(
            f"{AGENT_CONTEXT_CONTRACT} {relation.grain.value} is missing columns: "
            f"{', '.join(missing)}"
        )
    rows = list(frame.iter_rows(named=True))
    seen: set[tuple[Any, ...]] = set()
    for position, row in enumerate(rows):
        for field in relation.fields:
            value = row[field.name]
            if value is None and not field.nullable:
                raise AgentContextValidationError(
                    f"{relation.grain.value} row {position} has null {field.name}"
                )
            if value is not None:
                _validate_field_value(field, value, position, relation.grain)
        key = tuple(row[name] for name in relation.primary_key)
        if key in seen:
            raise AgentContextValidationError(
                f"{relation.grain.value} contains duplicate primary key {key!r}"
            )
        seen.add(key)
        _validate_row_semantics(row, relation.grain, position)
    if relation.grain is AgentContextGrain.DOCUMENT_REGISTRY:
        _validate_document_version_intervals(rows)


def validate_agent_context_relations(
    document_registry: pl.DataFrame,
    document_chunks: pl.DataFrame,
    context_entity_links: pl.DataFrame,
) -> None:
    validate_agent_context_frame(document_registry, AgentContextGrain.DOCUMENT_REGISTRY)
    validate_agent_context_frame(document_chunks, AgentContextGrain.DOCUMENT_CHUNKS)
    validate_agent_context_frame(context_entity_links, AgentContextGrain.CONTEXT_ENTITY_LINKS)
    documents = {row["document_version_id"]: row for row in document_registry.iter_rows(named=True)}
    contexts = {row["context_id"]: row for row in document_chunks.iter_rows(named=True)}
    for row in document_chunks.iter_rows(named=True):
        document = documents.get(row["document_version_id"])
        if document is None or document["document_id"] != row["document_id"]:
            raise AgentContextValidationError(
                "document_chunks contains an unresolved document version"
            )
        if policy_fingerprint(document) != policy_fingerprint(row):
            raise AgentContextValidationError(
                "document_chunks policy fields must match document_registry"
            )
    for row in context_entity_links.iter_rows(named=True):
        if row["context_id"] not in contexts:
            raise AgentContextValidationError(
                "context_entity_links contains an unresolved context_id"
            )


def _validate_row_semantics(
    row: Mapping[str, Any],
    grain: AgentContextGrain,
    position: int,
) -> None:
    if grain in {
        AgentContextGrain.DOCUMENT_REGISTRY,
        AgentContextGrain.DOCUMENT_CHUNKS,
    }:
        _validate_temporal(row, position, grain)
        _validate_policy(row, position, grain)
        _validate_freshness(row, position, grain)
    if grain is AgentContextGrain.DOCUMENT_REGISTRY:
        _validate_stable_id(row, "source_content_hash", position, grain)
        _validate_stable_id(row, "provenance_fingerprint", position, grain)
        expected_document_id = make_document_id(row["source_system"], row["source_key"])
        expected_version_id = make_document_version_id(
            expected_document_id,
            row["source_version"],
            row["source_content_hash"],
        )
        _expect_id(row, "document_id", expected_document_id, position, grain)
        _expect_id(row, "document_version_id", expected_version_id, position, grain)
    elif grain is AgentContextGrain.DOCUMENT_CHUNKS:
        _validate_stable_id(row, "document_id", position, grain)
        _validate_stable_id(row, "document_version_id", position, grain)
        _validate_stable_id(row, "chunk_content_hash", position, grain)
        _validate_stable_id(row, "provenance_fingerprint", position, grain)
        expected_chunk_id = make_chunk_id(row["document_id"], row["chunk_index"], row["text"])
        expected_context_id = make_context_id(row["document_version_id"], expected_chunk_id)
        _expect_id(row, "chunk_id", expected_chunk_id, position, grain)
        _expect_id(row, "context_id", expected_context_id, position, grain)
        if row["chunk_content_hash"] != content_hash(row["text"]):
            _invalid(grain, position, "chunk_content_hash does not match text")
        _validate_citation(row, position, grain)
    else:
        _validate_canonical_json(row, "entity_key", position, grain)
        _validate_stable_id(
            row,
            "link_provenance_fingerprint",
            position,
            grain,
        )
        expected_entity_id = make_entity_id(
            row["entity_namespace"], row["entity_name"], row["entity_key"]
        )
        expected_link_id = make_context_entity_link_id(
            row["context_id"], expected_entity_id, row["relationship_type"]
        )
        _expect_id(row, "entity_id", expected_entity_id, position, grain)
        _expect_id(
            row,
            "context_entity_link_id",
            expected_link_id,
            position,
            grain,
        )
        _validate_interval(
            row["recorded_from"],
            row.get("recorded_to"),
            "recorded",
            position,
            grain,
        )
        confidence = row.get("confidence")
        if confidence is not None and not 0 <= confidence <= 1:
            _invalid(grain, position, "confidence must be between 0 and 1")


def _validate_temporal(row: Mapping[str, Any], position: int, grain: AgentContextGrain) -> None:
    if row["validity_known"] is True:
        if row.get("valid_from") is None:
            _invalid(grain, position, "known validity requires valid_from")
        _validate_interval(row["valid_from"], row.get("valid_to"), "valid", position, grain)
    elif row.get("valid_from") is not None or row.get("valid_to") is not None:
        _invalid(
            grain,
            position,
            "unknown validity requires null valid_from and valid_to",
        )
    _validate_interval(
        row["recorded_from"],
        row.get("recorded_to"),
        "recorded",
        position,
        grain,
    )


def _validate_document_version_intervals(rows: list[dict[str, Any]]) -> None:
    by_document: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for position, row in enumerate(rows):
        by_document.setdefault(row["document_id"], []).append((position, row))
    for versions in by_document.values():
        versions.sort(key=lambda item: _utc(item[1]["recorded_from"], "recorded_from"))
        previous_end: datetime | None = None
        previous_open = False
        for position, row in versions:
            start = _utc(row["recorded_from"], "recorded_from")
            if previous_open or (previous_end is not None and start < previous_end):
                _invalid(
                    AgentContextGrain.DOCUMENT_REGISTRY,
                    position,
                    "recorded intervals must not overlap for one document_id",
                )
            previous_end = _optional_utc(row.get("recorded_to"), "recorded_to")
            previous_open = previous_end is None


def _validate_policy(row: Mapping[str, Any], position: int, grain: AgentContextGrain) -> None:
    groups = _string_array(row["access_groups"], allow_none=False)
    if len(groups) != len(set(groups)):
        _invalid(grain, position, "access_groups must not contain duplicates")
    if tuple(sorted(groups)) != groups:
        _invalid(grain, position, "access_groups must be canonically sorted")
    if row["is_public"] is not True and row["authorization_resolved"] is True:
        if not is_publishable_context(row):
            _invalid(
                grain,
                position,
                "non-public resolved policy requires at least one policy attribute",
            )


def _validate_freshness(row: Mapping[str, Any], position: int, grain: AgentContextGrain) -> None:
    ingested = _utc(row["ingested_at"], "ingested_at")
    materialized = _utc(row["materialized_at"], "materialized_at")
    if materialized < ingested:
        _invalid(grain, position, "materialized_at cannot precede ingested_at")
    checked = _optional_utc(row.get("freshness_checked_at"), "freshness_checked_at")
    if checked is not None and checked < ingested:
        _invalid(grain, position, "freshness_checked_at cannot precede ingested_at")


def _validate_citation(row: Mapping[str, Any], position: int, grain: AgentContextGrain) -> None:
    page = row.get("citation_page_number")
    if page is not None and page < 1:
        _invalid(grain, position, "citation_page_number must be at least 1")
    char_start = row.get("citation_char_start")
    char_end = row.get("citation_char_end")
    if (char_start is None) != (char_end is None):
        _invalid(grain, position, "citation character offsets must be paired")
    if char_start is not None and (char_start < 0 or char_end <= char_start):
        _invalid(grain, position, "citation character offsets are invalid")
    start_seconds = row.get("citation_start_seconds")
    end_seconds = row.get("citation_end_seconds")
    if (start_seconds is None) != (end_seconds is None):
        _invalid(grain, position, "citation time offsets must be paired")
    if start_seconds is not None and (start_seconds < 0 or end_seconds <= start_seconds):
        _invalid(grain, position, "citation time offsets are invalid")
    if not citation_locator(row):
        _invalid(grain, position, "citation locator must not be empty")


def _validate_interval(
    start: Any,
    end: Any,
    label: str,
    position: int,
    grain: AgentContextGrain,
) -> None:
    normalized_start = _utc(start, f"{label}_from")
    normalized_end = _optional_utc(end, f"{label}_to")
    if normalized_end is not None and normalized_end <= normalized_start:
        _invalid(grain, position, f"{label}_to must be later than {label}_from")


def _validate_field_value(
    field: ContractField,
    value: Any,
    position: int,
    grain: AgentContextGrain,
) -> None:
    valid = True
    if field.data_type in {"string", "json"}:
        valid = isinstance(value, str) and bool(value)
        if field.data_type == "json" and valid:
            try:
                json.loads(value)
            except json.JSONDecodeError:
                valid = False
    elif field.data_type == "integer":
        valid = not isinstance(value, bool) and isinstance(value, int)
    elif field.data_type == "float":
        valid = not isinstance(value, bool) and isinstance(value, int | float)
    elif field.data_type == "boolean":
        valid = isinstance(value, bool)
    elif field.data_type == "timestamp":
        try:
            _utc(value, field.name)
        except ValueError:
            valid = False
    elif field.data_type == "array[string]":
        try:
            _string_array(value, allow_none=False)
        except ValueError:
            valid = False
    if not valid:
        _invalid(
            grain,
            position,
            f"{field.name} must have type {field.data_type}",
        )


def _string_array(value: Any, *, allow_none: bool) -> tuple[str, ...]:
    if value is None and allow_none:
        return ()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            raise ValueError("array[string] must be a JSON array or sequence") from None
    if not isinstance(value, list | tuple) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError("array[string] must contain non-empty strings")
    return tuple(value)


def _canonical_json_text(value: object, name: str) -> str:
    value = _non_empty(value, name)
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        raise ValueError(f"{name} must be canonical JSON") from None
    encoded = json.dumps(
        decoded,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    if value != encoded:
        raise ValueError(f"{name} must be canonical JSON")
    return value


def _json_value(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _utc(value: Any, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    return value.astimezone(UTC)


def _optional_utc(value: Any, name: str) -> datetime | None:
    return None if value is None else _utc(value, name)


def _non_empty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_non_empty(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _stable_id(value: Any, name: str) -> str:
    normalized = _non_empty(value, name)
    if not _ID_PATTERN.fullmatch(normalized):
        raise ValueError(f"{name} must be a 32-character lowercase hex identifier")
    return normalized


def _expect_id(
    row: Mapping[str, Any],
    name: str,
    expected: str,
    position: int,
    grain: AgentContextGrain,
) -> None:
    if row[name] != expected:
        _invalid(grain, position, f"{name} does not match its v1 algorithm")


def _validate_stable_id(
    row: Mapping[str, Any],
    name: str,
    position: int,
    grain: AgentContextGrain,
) -> None:
    try:
        _stable_id(row[name], name)
    except ValueError as error:
        _invalid(grain, position, str(error))


def _validate_canonical_json(
    row: Mapping[str, Any],
    name: str,
    position: int,
    grain: AgentContextGrain,
) -> None:
    try:
        _canonical_json_text(row[name], name)
    except ValueError as error:
        _invalid(grain, position, str(error))


def _invalid(grain: AgentContextGrain, position: int, message: str) -> None:
    raise AgentContextValidationError(
        f"{AGENT_CONTEXT_CONTRACT} {grain.value} row {position}: {message}"
    )
