"""Entity-linking resolver contracts and the deterministic alias-table resolver.

The alias-table resolver joins entity mentions to an operator-owned alias
dimension with exact and normalized text matching. It never guesses: every
mention outcome is an explicit ``matched``, ``ambiguous``, or ``unmatched``
status, and ambiguous candidates are preserved as separate rows.
"""
from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictBool, field_validator, model_validator

from ..hashing import canonical_fingerprint

# Bumped whenever matching semantics change (normalization rules, method
# precedence, status assignment) so downstream consumers can invalidate rows
# produced by an older resolver even though the package version moved for
# unrelated reasons.
ALIAS_RESOLVER_VERSION = "1"

ALIAS_SET_FINGERPRINT_DOMAIN = "dbt-ml.entity-alias-set"
ENTITY_LINK_FINGERPRINT_DOMAIN = "dbt-ml.entity-link"

MatchMethod = Literal["exact", "normalized"]
LinkStatus = Literal["matched", "ambiguous", "unmatched"]


class EntityLinkOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    resolver: Literal["alias_table"] = "alias_table"
    mentions: str
    aliases: str
    mention_id_field: str = "entity_id"
    document_id_field: str = "document_id"
    mention_text_field: str = "entity_text"
    label_field: str | None = "label"
    start_field: str | None = "start"
    end_field: str | None = "end"
    alias_text_field: str = "alias"
    namespace_field: str = "entity_namespace"
    canonical_id_field: str = "canonical_id"
    match_methods: tuple[MatchMethod, ...] = ("exact", "normalized")
    on_ambiguity: Literal["keep", "error"] = "keep"
    include_fields: tuple[str, ...] = ()
    include_mention_text: StrictBool = False

    @field_validator(
        "mentions",
        "aliases",
        "mention_id_field",
        "document_id_field",
        "mention_text_field",
        "alias_text_field",
        "namespace_field",
        "canonical_id_field",
    )
    @classmethod
    def _non_empty_string(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized

    @field_validator("label_field", "start_field", "end_field")
    @classmethod
    def _non_empty_optional_string(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty; use null to disable")
        return normalized

    @field_validator("match_methods")
    @classmethod
    def _ordered_unique_methods(
        cls, values: tuple[MatchMethod, ...]
    ) -> tuple[MatchMethod, ...]:
        if not values:
            raise ValueError("match_methods must not be empty")
        if len(values) != len(set(values)):
            raise ValueError("match_methods entries must be unique")
        return values

    @field_validator("include_fields")
    @classmethod
    def _unique_include_fields(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("include_fields entries must not be empty")
        if len(normalized) != len(set(normalized)):
            raise ValueError("include_fields entries must be unique")
        return normalized

    @model_validator(mode="after")
    def _consistent_references(self) -> EntityLinkOptions:
        if self.mentions == self.aliases:
            raise ValueError(
                "mentions and aliases must reference two different upstream models"
            )
        forbidden = {
            field
            for field in (
                self.mention_id_field,
                self.document_id_field,
                self.mention_text_field,
            )
            if field in self.include_fields
        }
        if forbidden:
            raise ValueError(
                "include_fields must not repeat the mention ID, document ID, or "
                f"mention text field: {sorted(forbidden)}"
            )
        return self


def normalize_alias_text(value: str) -> str:
    """The resolver's ``normalized`` match key: NFKC fold, casefold, and
    whitespace collapse. Deliberately conservative — spelling variants and
    legal-suffix conventions belong in the operator-owned alias table."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def alias_set_fingerprint(rows: Iterable[Mapping[str, str]]) -> str:
    """One-way identity of the complete alias set (namespace, alias, canonical
    ID triples). Recorded on every link row so alias-table edits are visible to
    downstream invalidation without retaining the alias contents."""
    canonical_rows = sorted(
        (row["entity_namespace"], row["alias"], row["canonical_id"]) for row in rows
    )
    return canonical_fingerprint(
        canonical_rows, domain=ALIAS_SET_FINGERPRINT_DOMAIN
    )


def entity_link_id(
    *,
    mention_id: str,
    document_id: str,
    entity_namespace: str | None,
    canonical_id: str | None,
    match_method: str | None,
    status: LinkStatus,
) -> str:
    return canonical_fingerprint(
        {
            "mention_id": mention_id,
            "document_id": document_id,
            "entity_namespace": entity_namespace or "",
            "canonical_id": canonical_id or "",
            "match_method": match_method or "",
            "status": status,
        },
        domain=ENTITY_LINK_FINGERPRINT_DOMAIN,
    )
