"""Contracts for deterministic document-level tone/sentiment (issue #216).

Tone is scored by matching a document's tokens against an operator-owned tone
lexicon (rows of ``term``, ``category``, optional ``weight``) and aggregating per
document. It is deliberately deterministic: the same versioned lexicon and the
same options always produce the same scores, and the lexicon's content is
fingerprinted as ``lexicon_version`` so an edit is visible downstream without
retaining the lexicon. No learned model or LLM is involved — a general sentiment
score is never presented as an economic fact.

Emitted signal columns are fixed at compile time: ``emit`` is an explicit list of
lexicon categories, so the output schema never depends on the lexicon rows that
happen to be in the warehouse (mirroring the document-feature contract).
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from ..hashing import canonical_fingerprint

# Bumped whenever scoring semantics change (matching, negation, normalization,
# status assignment) so downstream consumers can invalidate rows produced by an
# older scorer even though the package version moved for unrelated reasons.
TONE_SCORER_VERSION = "1"

TONE_LEXICON_FINGERPRINT_DOMAIN = "dbt-ml.tone-lexicon"

# English negators used when negation is enabled and the operator does not
# override them. Other languages should pass their own `negators` (the lexicon
# and these are the only language-specific inputs).
DEFAULT_NEGATORS: tuple[str, ...] = (
    "not",
    "no",
    "never",
    "without",
    "cannot",
    "n't",
    "nor",
    "none",
    "neither",
)

# Fixed (always-present) output columns, reserved so an `emit` category or an
# `include_fields` passthrough cannot collide with them.
RESERVED_COLUMNS: frozenset[str] = frozenset(
    {
        "document_id",
        "token_count",
        "matched_token_count",
        "coverage",
        "status",
        "scorer",
        "scorer_version",
        "lexicon_version",
    }
)

_COLUMN_SAFE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def category_score_column(value: str) -> str:
    return f"{value.lower()}_score"


def category_hits_column(value: str) -> str:
    return f"{value.lower()}_hits"


class ToneOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tokens: str
    lexicon: str
    emit: tuple[str, ...]
    # Optional parent spine. With it, the documents table defines which
    # documents get a row (a document with no tokens still appears with
    # token_count 0 and status insufficient_text) and is the source of
    # include_fields; without it, only documents present in the token table
    # appear. Mirrors the document-feature contract.
    documents: str | None = None

    document_id_field: str = "document_id"
    documents_id_field: str = "document_id"
    match_field: str = "lemma"
    language: str = "en"
    language_field: str = "nlp_language"
    token_index_field: str = "token_index"
    sentence_index_field: str = "sentence_index"

    lexicon_term_field: str = "term"
    lexicon_category_field: str = "category"
    # Optional per-term weight column; None (or an absent column) means every
    # matched term contributes 1.0.
    lexicon_weight_field: str | None = "weight"

    negation: StrictBool = True
    negators: tuple[str, ...] = DEFAULT_NEGATORS
    negation_window: int = Field(default=3, ge=1, le=50)
    min_tokens: int = Field(default=1, ge=0)

    include_fields: tuple[str, ...] = ()
    include_matched_terms: StrictBool = False

    @field_validator(
        "tokens",
        "lexicon",
        "document_id_field",
        "documents_id_field",
        "match_field",
        "language",
        "language_field",
        "token_index_field",
        "sentence_index_field",
        "lexicon_term_field",
        "lexicon_category_field",
    )
    @classmethod
    def _non_empty_string(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized

    @field_validator("documents")
    @classmethod
    def _non_empty_optional_documents(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty; omit the option to disable")
        return normalized

    @field_validator("lexicon_weight_field")
    @classmethod
    def _non_empty_optional_string(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty; use null to weight every term 1.0")
        return normalized

    @field_validator("emit", "include_fields", "negators")
    @classmethod
    def _unique_nonempty(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("entries must not be empty")
        if len(normalized) != len(set(normalized)):
            raise ValueError("entries must be unique")
        return normalized

    @field_validator("emit")
    @classmethod
    def _emit_not_empty(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("emit must name at least one lexicon category")
        return values

    @model_validator(mode="after")
    def _consistent_configuration(self) -> ToneOptions:
        configured = [name for name in (self.tokens, self.lexicon, self.documents) if name]
        if len(configured) != len(set(configured)):
            raise ValueError("tokens, lexicon, and documents must name different models")
        if self.include_fields and self.documents is None:
            raise ValueError("include_fields needs a `documents:` dependency")
        self._validate_output_columns()
        return self

    def _validate_output_columns(self) -> None:
        """Reject configurations whose emitted columns collide with each other or
        with the fixed columns. Category names fold to lowercase, so `Positive`
        and `positive` are the same column."""
        seen: dict[str, str] = {column: "reserved" for column in RESERVED_COLUMNS}
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
        """Every configurable output column, as (column, source), for collision
        checks and output ordering. Excludes the always-present fixed columns."""
        columns: list[tuple[str, str]] = []
        for value in self.emit:
            columns.append((category_score_column(value), f"emit[{value}]"))
            columns.append((category_hits_column(value), f"emit[{value}]"))
        for value in self.include_fields:
            columns.append((value, f"include_fields[{value}]"))
        return columns

    def declared_dependencies(self) -> tuple[str, ...]:
        names = [self.tokens, self.lexicon]
        if self.documents is not None:
            names.append(self.documents)
        return tuple(names)


def tone_lexicon_fingerprint(rows: Iterable[Mapping[str, object]]) -> str:
    """One-way identity of the effective tone lexicon: deduplicated, sorted
    (term, category, weight) triples fingerprinted under a stable domain.
    Recorded on every tone row so a lexicon edit is visible to downstream
    invalidation without retaining the lexicon contents. Deduplicated because
    repeated identical entries cannot change any score, so they must not signal
    a spurious invalidation."""
    canonical_rows = sorted(
        {
            (str(row["term"]), str(row["category"]), _canonical_weight(row.get("weight")))
            for row in rows
        }
    )
    return canonical_fingerprint(canonical_rows, domain=TONE_LEXICON_FINGERPRINT_DOMAIN)


def _canonical_weight(value: object) -> str:
    # A missing or null weight is identical to an explicit 1.0 for scoring, so it
    # fingerprints the same.
    if value is None:
        return "1.0"
    if isinstance(value, (int, float, str)):
        return repr(float(value))
    raise ValueError(f"tone lexicon weight must be numeric, got {type(value).__name__}")
