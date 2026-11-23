"""Relation-extractor options and their parsing, free of the extractors.

The compiler checks a project's relation transforms before any run, and validating
a project is configuration work: it must not import the text-processing package
(issue #823, ALE-791). The option models, the discriminated union and the parser
live here; the extractors that act on them stay in `stel.text.relations`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    TypeAdapter,
    field_validator,
    model_validator,
)

CoOccurrenceScope = Literal["sentence", "window"]


def _require_non_empty(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must not be empty")
    return normalized


def _reject_coerced_number(value: Any) -> Any:
    """Reject a boolean or string before Pydantic's lax coercion turns it into a
    float, so ``threshold: true`` or ``threshold: "0.9"`` fails the strict
    preflight rather than silently changing which relations are asserted."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("must be a number, not a boolean or a string")
    return value


class RelationBaseOptions(BaseModel):
    """Fields shared by every extractor: the mentions model, mention identity,
    the evidence locators, and privacy controls. Extractor-specific pairing
    inputs live on the concrete subclasses."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mentions: str

    mention_id_field: str = "entity_id"
    document_id_field: str = "document_id"
    sentence_index_field: str = "sentence_index"
    start_field: str = "start"
    end_field: str = "end"
    label_field: str | None = "label"
    mention_text_field: str = "entity_text"

    # Mentions columns carrying the document's effective date and the instant it
    # was observed (ADR-0056). Each is read per document, never from the clock, so
    # a rerun republishes identical rows; unset leaves `valid_from` or
    # `observed_at` null rather than guessing.
    effective_date_field: str | None = None
    observed_at_field: str | None = None

    # Only mentions whose label is in this allow-list participate; empty = all.
    labels: tuple[str, ...] = ()
    # Fail closed on a pathological document rather than materialize a runaway
    # quadratic pair explosion.
    max_pairs_per_document: int = Field(default=10_000, ge=1, le=1_000_000)
    # Evidence text is a verbatim excerpt and may be sensitive, so it is withheld
    # unless explicitly requested (and then the mentions model must carry it).
    include_mention_text: StrictBool = False

    @field_validator(
        "mentions",
        "mention_id_field",
        "document_id_field",
        "sentence_index_field",
        "start_field",
        "end_field",
        "mention_text_field",
    )
    @classmethod
    def _non_empty_string(cls, value: str) -> str:
        return _require_non_empty(value)

    @field_validator("label_field", "effective_date_field", "observed_at_field")
    @classmethod
    def _non_empty_optional_string(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty; use null to disable")
        return normalized

    @field_validator("labels")
    @classmethod
    def _unique_labels(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("labels entries must not be empty")
        if len(normalized) != len(set(normalized)):
            raise ValueError("labels entries must be unique")
        return normalized

    @model_validator(mode="after")
    def _labels_require_label_field(self) -> RelationBaseOptions:
        if self.labels and self.label_field is None:
            raise ValueError(
                "labels filtering requires label_field; set label_field or clear labels"
            )
        return self


class CoOccurrenceExtractorOptions(RelationBaseOptions):
    extractor: Literal["co_occurrence"] = "co_occurrence"
    # `sentence`: mentions sharing a sentence_index. `window`: mentions whose
    # spans fall within `max_char_gap` characters of each other (may cross
    # sentences, so sentence_index is recorded only when both sides agree).
    scope: CoOccurrenceScope = "sentence"
    max_char_gap: int = Field(default=100, ge=0, le=1_000_000)
    # Co-occurrence is untyped, so this is an operator-chosen label (schema-
    # controlled by being a single fixed choice) and every row is symmetric.
    relation_type: str = "co_occurs_with"

    @field_validator("relation_type")
    @classmethod
    def _non_empty_relation_type(cls, value: str) -> str:
        return _require_non_empty(value)


class RelationRule(BaseModel):
    """One directed, typed rule: assert ``relation_type`` from a subject mention
    of ``subject_label`` to an object mention of ``object_label`` when the two
    co-occur in scope. Directed by construction — undirected proximity is what
    the co-occurrence extractor is for."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    subject_label: str
    object_label: str
    relation_type: str

    @field_validator("subject_label", "object_label", "relation_type")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        return _require_non_empty(value)


class RuleExtractorOptions(RelationBaseOptions):
    extractor: Literal["rule"] = "rule"
    scope: CoOccurrenceScope = "sentence"
    max_char_gap: int = Field(default=100, ge=0, le=1_000_000)
    # The schema-controlled set of typed rules. The distinct `relation_type`
    # values are exactly the relations this model can emit.
    rules: tuple[RelationRule, ...]

    @field_validator("rules")
    @classmethod
    def _unique_non_empty_rules(
        cls, values: tuple[RelationRule, ...]
    ) -> tuple[RelationRule, ...]:
        if not values:
            raise ValueError("rules must not be empty")
        keys = [(r.subject_label, r.object_label, r.relation_type) for r in values]
        if len(keys) != len(set(keys)):
            raise ValueError(
                "rules must be unique on (subject_label, object_label, relation_type)"
            )
        return values

    @model_validator(mode="after")
    def _rules_require_label_field(self) -> RuleExtractorOptions:
        if self.label_field is None:
            raise ValueError(
                "the rule extractor matches on mention labels and requires "
                "label_field; set it to the mentions model's label column"
            )
        return self


class ModelAssertionExtractorOptions(RelationBaseOptions):
    """A learned/LLM extractor (issue #240). For each in-scope candidate pair it
    asks a governed inference provider whether one of the ``relation_types`` holds
    and with what confidence. The type set is a schema-controlled allow-list: the
    model may only assert those, enforced in the output schema and re-validated."""

    extractor: Literal["model_assertion"] = "model_assertion"
    scope: CoOccurrenceScope = "sentence"
    max_char_gap: int = Field(default=100, ge=0, le=1_000_000)
    # The relations the model may assert. Also the output-schema enum, so the
    # model is constrained to this set both at request and validation time.
    relation_types: tuple[str, ...]
    directed: StrictBool = True
    # Assertions at/above are `asserted`; below are `no_relation`.
    threshold: float = 0.5
    # Optional operator instructions folded into the system prompt.
    prompt: str | None = None

    @field_validator("relation_types")
    @classmethod
    def _unique_non_empty_types(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if not normalized:
            raise ValueError("relation_types must not be empty")
        if any(not value for value in normalized):
            raise ValueError("relation_types entries must not be empty")
        if len(normalized) != len(set(normalized)):
            raise ValueError("relation_types entries must be unique")
        return normalized

    @field_validator("threshold", mode="before")
    @classmethod
    def _strict_threshold(cls, value: Any) -> Any:
        return _reject_coerced_number(value)

    @field_validator("threshold")
    @classmethod
    def _threshold_in_unit_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        return value

    @field_validator("prompt")
    @classmethod
    def _non_empty_prompt(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("prompt must not be empty; use null to omit")
        return normalized


RelationExtractorConfig = Annotated[
    CoOccurrenceExtractorOptions | RuleExtractorOptions | ModelAssertionExtractorOptions,
    Field(discriminator="extractor"),
]

_RELATION_ADAPTER: TypeAdapter[
    CoOccurrenceExtractorOptions | RuleExtractorOptions | ModelAssertionExtractorOptions
] = TypeAdapter(RelationExtractorConfig)


def parse_relation_options(
    options: Mapping[str, Any],
) -> CoOccurrenceExtractorOptions | RuleExtractorOptions | ModelAssertionExtractorOptions:
    """Validate raw options into the extractor-specific model. ``extractor`` is
    optional and defaults to ``co_occurrence``; an unknown value fails
    discriminator validation with the valid tags named."""
    if isinstance(options, Mapping) and "extractor" not in options:
        options = {**options, "extractor": "co_occurrence"}
    return _RELATION_ADAPTER.validate_python(options)
