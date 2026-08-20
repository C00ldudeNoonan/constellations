"""Contracts for deterministic keyphrase extraction from NLP token child tables.

Keyphrases are scored by extracting contiguous lemma n-grams from the token
child table and ranking them by normalized term frequency within the document.
No IDF, no learned model, no optional extra — the same token table and the same
options always produce the same ranked list.

The output is a child table (one row per ``phrase_id``) with a stable identifier
derived from ``(document_id, phrase_lemma)``. Phrase text is opt-in because it
is a verbatim excerpt of the source document and may contain sensitive content.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

# Bumped whenever scoring semantics change (n-gram construction, boundary
# filtering, score normalization, rank tie-breaking) so downstream consumers
# can invalidate rows produced by an older extractor.
KEYPHRASE_EXTRACTOR_VERSION = "1"

KEYPHRASE_EXTRACTOR_NAME = "ngram_freq"

KEYPHRASE_DOMAIN = "dbt-ml.keyphrase"

# POS tags excluded from phrase boundaries. Interior tokens are unrestricted
# so that phrases like "rate of return" remain valid 3-grams.
DEFAULT_STOP_POS: tuple[str, ...] = ("PUNCT", "SPACE", "NUM", "SYM", "X")

# Fixed output columns — reserved so no future option can collide.
RESERVED_COLUMNS: frozenset[str] = frozenset(
    {
        "phrase_id",
        "document_id",
        "rank",
        "score",
        "phrase_lemma",
        "phrase_text",
        "phrase_length",
        "token_start",
        "token_end",
        "sentence_index",
        "extractor",
        "extractor_version",
    }
)

_COLUMN_SAFE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class KeyphraseOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tokens: str

    document_id_field: str = "document_id"
    lemma_field: str = "lemma"
    text_field: str = "token_text"
    pos_field: str = "pos"
    is_stop_field: str = "is_stop"
    token_index_field: str = "token_index"
    sentence_index_field: str = "sentence_index"
    language: str = "en"
    language_field: str = "nlp_language"

    min_phrase_length: int = Field(default=1, ge=1, le=10)
    max_phrase_length: int = Field(default=3, ge=1, le=10)
    top_k: int = Field(default=10, ge=1)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)

    stop_pos: tuple[str, ...] = DEFAULT_STOP_POS
    # Phrase text is opt-in; the token table already carries token_text
    # (from text_field) but it is a verbatim excerpt and may be sensitive.
    include_phrase_text: StrictBool = False

    @field_validator(
        "tokens",
        "document_id_field",
        "lemma_field",
        "text_field",
        "pos_field",
        "is_stop_field",
        "token_index_field",
        "sentence_index_field",
        "language",
        "language_field",
    )
    @classmethod
    def _non_empty_string(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized

    @field_validator("stop_pos")
    @classmethod
    def _unique_nonempty(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(v.strip() for v in values)
        if any(not v for v in normalized):
            raise ValueError("entries must not be empty")
        if len(normalized) != len(set(normalized)):
            raise ValueError("entries must be unique")
        return normalized

    @model_validator(mode="after")
    def _consistent_configuration(self) -> KeyphraseOptions:
        if self.max_phrase_length < self.min_phrase_length:
            raise ValueError(
                f"max_phrase_length ({self.max_phrase_length}) must be >= "
                f"min_phrase_length ({self.min_phrase_length})"
            )
        return self

    def declared_dependencies(self) -> tuple[str, ...]:
        return (self.tokens,)
