"""Entity-link options and their parsing, free of the resolver implementations.

The compiler validates `link_entities` options before any run, and validating a
project is configuration work: it must not import polars or the text-processing
package (issue #823, ALE-791). The option models, the discriminated union and the
parsers live here; the resolvers that act on them stay in `stel.text.linking`.
"""

from __future__ import annotations

import math
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

from .config.vocabulary import VALUES_FROM_PATTERN

MatchMethod = Literal["exact", "normalized"]
SimilarityMetric = Literal["cosine", "euclidean", "dot"]
FuzzyMetric = Literal["trigram_dice", "jaccard_token"]


def _require_non_empty(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must not be empty")
    return normalized


def _reject_coerced_number(value: Any) -> Any:
    """Reject a boolean or string before Pydantic's lax coercion turns it into a
    float. Without this, ``threshold: true`` silently becomes ``1.0`` and
    ``threshold: "0.5"`` becomes ``0.5``, quietly changing which links match —
    contrary to the strict preflight-validation contract. ``int`` is still
    accepted (YAML ``0``/``1``) and coerced to ``float`` downstream."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("must be a number, not a boolean or a string")
    return value


class EntityLinkBaseOptions(BaseModel):
    """Fields shared by every resolver: mention identity, privacy controls, and
    the passthrough/include projection. Resolver-specific matching inputs live on
    the concrete subclasses."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mentions: str
    aliases: str
    mention_id_field: str = "entity_id"
    document_id_field: str = "document_id"
    mention_text_field: str = "entity_text"
    label_field: str | None = "label"
    start_field: str | None = "start"
    end_field: str | None = "end"
    namespace_field: str = "entity_namespace"
    canonical_id_field: str = "canonical_id"
    on_ambiguity: Literal["keep", "error"] = "keep"
    include_fields: tuple[str, ...] = ()
    include_mention_text: StrictBool = False

    @field_validator(
        "mentions",
        "aliases",
        "mention_id_field",
        "document_id_field",
        "mention_text_field",
        "namespace_field",
        "canonical_id_field",
    )
    @classmethod
    def _non_empty_string(cls, value: str) -> str:
        return _require_non_empty(value)

    @field_validator("label_field", "start_field", "end_field")
    @classmethod
    def _non_empty_optional_string(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty; use null to disable")
        return normalized

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
    def _consistent_references(self) -> EntityLinkBaseOptions:
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


class AliasTableResolverOptions(EntityLinkBaseOptions):
    resolver: Literal["alias_table"] = "alias_table"
    alias_text_field: str = "alias"
    match_methods: tuple[MatchMethod, ...] = ("exact", "normalized")

    @field_validator("alias_text_field")
    @classmethod
    def _non_empty_alias_field(cls, value: str) -> str:
        return _require_non_empty(value)

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


class VectorSimilarityResolverOptions(EntityLinkBaseOptions):
    resolver: Literal["vector_similarity"] = "vector_similarity"
    mention_vector_field: str = "embedding"
    alias_vector_field: str = "embedding"
    metric: SimilarityMetric = "cosine"
    # No default: matching by similarity is meaningless without an operator-chosen
    # acceptance bar, so require it explicitly rather than guessing one.
    threshold: float
    # Candidates within this margin of a namespace's top score are ambiguous
    # rather than silently resolved to the arg-max. 0.0 flags only exact ties.
    ambiguity_margin: float = 0.0
    # Guards against comparing vectors from unrelated embedding spaces: when both
    # sides carry this column (the `embed` kind emits `embedding_config_hash`), a
    # mismatch fails the run. Set to null to skip the check for vectors from a
    # source that does not record an embedding identity.
    embedding_config_hash_field: str | None = "embedding_config_hash"

    @field_validator("mention_vector_field", "alias_vector_field")
    @classmethod
    def _non_empty_vector_field(cls, value: str) -> str:
        return _require_non_empty(value)

    @field_validator("embedding_config_hash_field")
    @classmethod
    def _non_empty_optional_hash_field(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty; use null to disable")
        return normalized

    _reject_coerced = field_validator(
        "threshold", "ambiguity_margin", mode="before"
    )(_reject_coerced_number)

    @field_validator("threshold")
    @classmethod
    def _finite_threshold(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("threshold must be a finite number")
        return value

    @field_validator("ambiguity_margin")
    @classmethod
    def _non_negative_margin(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0:
            raise ValueError("ambiguity_margin must be a finite, non-negative number")
        return value


class FuzzyResolverOptions(EntityLinkBaseOptions):
    resolver: Literal["fuzzy"] = "fuzzy"
    alias_text_field: str = "alias"
    # `trigram_dice` (character-trigram Dice) is robust to spelling variants,
    # legal suffixes, and typos — the reason to reach past `alias_table`'s
    # exact/normalized matching. `jaccard_token` compares whitespace tokens and
    # suits reordered multi-word names. Both are in [0, 1], higher is better.
    metric: FuzzyMetric = "trigram_dice"
    # NFKC-fold, casefold, and whitespace-collapse both sides before scoring, so
    # fuzzy matching is case- and width-insensitive by default. Disable to score
    # the raw surface forms.
    normalize: StrictBool = True
    # No default: a similarity bar is meaningless without an operator-chosen
    # acceptance threshold, so require it explicitly.
    threshold: float
    # Candidates within this margin of a namespace's top score are ambiguous
    # rather than silently resolved to the arg-max. 0.0 flags only exact ties.
    ambiguity_margin: float = 0.0

    _reject_coerced = field_validator(
        "threshold", "ambiguity_margin", mode="before"
    )(_reject_coerced_number)

    @field_validator("alias_text_field")
    @classmethod
    def _non_empty_alias_field(cls, value: str) -> str:
        return _require_non_empty(value)

    @field_validator("threshold")
    @classmethod
    def _threshold_in_unit_interval(cls, value: float) -> float:
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(
                "threshold must be a finite number in (0, 1]; fuzzy similarity is "
                "always in [0, 1]"
            )
        return value

    @field_validator("ambiguity_margin")
    @classmethod
    def _non_negative_margin(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0:
            raise ValueError("ambiguity_margin must be a finite, non-negative number")
        return value


EntityLinkConfig = Annotated[
    AliasTableResolverOptions
    | VectorSimilarityResolverOptions
    | FuzzyResolverOptions,
    Field(discriminator="resolver"),
]

# Back-compat alias: the pre-registry public name resolves to the default
# resolver's options model.
EntityLinkOptions = AliasTableResolverOptions

_ENTITY_LINK_ADAPTER: TypeAdapter[
    AliasTableResolverOptions | VectorSimilarityResolverOptions | FuzzyResolverOptions
] = TypeAdapter(EntityLinkConfig)


def parse_entity_link_options(
    options: Mapping[str, Any],
) -> AliasTableResolverOptions | VectorSimilarityResolverOptions | FuzzyResolverOptions:
    """Validate raw options into the resolver-specific model. ``resolver`` is
    optional and defaults to ``alias_table`` (phase-1 behavior); an unknown value
    fails discriminator validation with the valid tags named."""
    if isinstance(options, Mapping) and "resolver" not in options:
        options = {**options, "resolver": "alias_table"}
    return _ENTITY_LINK_ADAPTER.validate_python(options)


def parse_vocabulary_alias_source(aliases: str) -> str | None:
    """The vocabulary name in an `aliases: vocab.<name>` reference, or `None`
    when `aliases` names an upstream model instead (issue #627). Reuses
    `values_from`'s exact syntax — the two are the same kind of reference, a
    project-declared vocabulary instead of a materialized table."""
    match = VALUES_FROM_PATTERN.match(aliases)
    return match.group(1) if match else None
