"""Contracts for document-level aggregate features over NLP child tables.

Feature columns are fixed at compile time: every configurable rollup is an
explicit list of POS tags, entity labels, or canonical namespaces, so the output
schema never depends on the data that happens to be in the warehouse.
"""
from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

BaseFeature = Literal[
    "token_count",
    "sentence_count",
    "entity_count",
    "unique_lemma_count",
    "lexical_diversity",
    "stop_ratio",
    "alpha_ratio",
]

BASE_FEATURES: tuple[BaseFeature, ...] = (
    "token_count",
    "sentence_count",
    "entity_count",
    "unique_lemma_count",
    "lexical_diversity",
    "stop_ratio",
    "alpha_ratio",
)

# Features derived from the optional entity table; requesting them without an
# `entities:` dependency is a configuration error rather than a silent null.
ENTITY_FEATURES: frozenset[str] = frozenset({"entity_count"})

_COLUMN_SAFE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def pos_count_column(value: str) -> str:
    return f"pos_{value.lower()}_count"


def pos_ratio_column(value: str) -> str:
    return f"pos_{value.lower()}_ratio"


def entity_label_column(value: str) -> str:
    return f"entity_{value.lower()}_count"


def link_namespace_column(value: str) -> str:
    return f"linked_{value.lower()}_count"


def link_status_column(value: str) -> str:
    return f"link_{value.lower()}_count"


class DocumentFeatureOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tokens: str
    entities: str | None = None
    links: str | None = None
    documents: str | None = None

    document_id_field: str = "document_id"
    documents_id_field: str = "document_id"
    pos_field: str = "pos"
    lemma_field: str = "lemma"
    sentence_index_field: str = "sentence_index"
    label_field: str = "label"
    namespace_field: str = "entity_namespace"
    canonical_id_field: str = "canonical_id"
    status_field: str = "status"

    emit: tuple[BaseFeature, ...] = BASE_FEATURES
    pos_counts: tuple[str, ...] = ()
    pos_ratios: tuple[str, ...] = ()
    entity_label_counts: tuple[str, ...] = ()
    link_namespace_counts: tuple[str, ...] = ()
    link_status_counts: tuple[str, ...] = ()
    include_fields: tuple[str, ...] = ()

    @field_validator(
        "tokens",
        "document_id_field",
        "documents_id_field",
        "pos_field",
        "lemma_field",
        "sentence_index_field",
        "label_field",
        "namespace_field",
        "canonical_id_field",
        "status_field",
    )
    @classmethod
    def _non_empty_string(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized

    @field_validator("entities", "links", "documents")
    @classmethod
    def _non_empty_optional_string(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty; omit the option to disable")
        return normalized

    @model_validator(mode="before")
    @classmethod
    def _default_emit_to_available_features(cls, data: Any) -> Any:
        """Left unset, `emit` is every base feature the configured dependencies
        can actually produce. Naming an unavailable feature explicitly stays an
        error — the default adapts, an explicit request does not."""
        if not isinstance(data, dict) or "emit" in data:
            return data
        has_entities = data.get("entities") is not None
        return {
            **data,
            "emit": tuple(
                feature
                for feature in BASE_FEATURES
                if has_entities or feature not in ENTITY_FEATURES
            ),
        }

    @field_validator("emit")
    @classmethod
    def _unique_features(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("emit must not be empty")
        if len(values) != len(set(values)):
            raise ValueError("emit entries must be unique")
        return values

    @field_validator(
        "pos_counts",
        "pos_ratios",
        "entity_label_counts",
        "link_namespace_counts",
        "link_status_counts",
        "include_fields",
    )
    @classmethod
    def _unique_nonempty(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("entries must not be empty")
        if len(normalized) != len(set(normalized)):
            raise ValueError("entries must be unique")
        return normalized

    @model_validator(mode="after")
    def _consistent_configuration(self) -> DocumentFeatureOptions:
        self._validate_dependency_names()
        self._validate_feature_availability()
        self._validate_output_columns()
        return self

    def _validate_dependency_names(self) -> None:
        configured = [
            name
            for name in (self.tokens, self.entities, self.links, self.documents)
            if name is not None
        ]
        if len(configured) != len(set(configured)):
            raise ValueError(
                "tokens, entities, links, and documents must name different models"
            )

    def _validate_feature_availability(self) -> None:
        if self.entities is None:
            unavailable = sorted(set(self.emit) & ENTITY_FEATURES)
            if unavailable:
                raise ValueError(
                    f"emit requests {unavailable}, which needs an `entities:` "
                    "dependency"
                )
            if self.entity_label_counts:
                raise ValueError(
                    "entity_label_counts needs an `entities:` dependency"
                )
        if self.links is None and (
            self.link_namespace_counts or self.link_status_counts
        ):
            raise ValueError(
                "link_namespace_counts and link_status_counts need a `links:` "
                "dependency"
            )
        if self.documents is None and self.include_fields:
            raise ValueError("include_fields needs a `documents:` dependency")

    def _validate_output_columns(self) -> None:
        """Reject configurations whose feature columns collide. Values are folded
        to lowercase for column names, so `ORG` and `org` are the same column."""
        seen: dict[str, str] = {}
        for column, source in self.output_columns():
            if not _COLUMN_SAFE.fullmatch(column):
                raise ValueError(
                    f"{source} produces invalid output column name '{column}'"
                )
            if column in seen:
                raise ValueError(
                    f"{source} collides with {seen[column]} on output column "
                    f"'{column}'"
                )
            seen[column] = source

    def output_columns(self) -> list[tuple[str, str]]:
        """Every feature column this configuration emits, as (column, source)."""
        columns: list[tuple[str, str]] = [
            (feature, "emit") for feature in self.emit
        ]
        for value in self.pos_counts:
            columns.append((pos_count_column(value), f"pos_counts[{value}]"))
        for value in self.pos_ratios:
            columns.append((pos_ratio_column(value), f"pos_ratios[{value}]"))
        for value in self.entity_label_counts:
            columns.append(
                (entity_label_column(value), f"entity_label_counts[{value}]")
            )
        for value in self.link_namespace_counts:
            columns.append(
                (link_namespace_column(value), f"link_namespace_counts[{value}]")
            )
        for value in self.link_status_counts:
            columns.append(
                (link_status_column(value), f"link_status_counts[{value}]")
            )
        for value in self.include_fields:
            columns.append((value, f"include_fields[{value}]"))
        return columns

    def declared_dependencies(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in (self.tokens, self.entities, self.links, self.documents)
            if name is not None
        )
