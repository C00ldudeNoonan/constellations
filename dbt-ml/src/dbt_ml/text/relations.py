"""Typed relation extraction over the NLP entity-mention child table (issue #220).

A *relation* connects two entity mentions in the same document. The design
keeps three kinds of relationship strictly distinguishable so a consumer never
mistakes proximity for a semantic assertion:

- **co-occurrence** — the deterministic, offline built-in. Two mentions
  co-occur when they share a sentence (``scope: sentence``) or fall within a
  character window (``scope: window``). This asserts nothing beyond "these
  mentions appeared together"; every row is labelled ``method = "co_occurrence"``
  and is symmetric (``directed = false``).
- **rule-derived** — a future deterministic extractor that asserts a typed,
  directed relation from explicit rules. It slots into the same registry with
  ``method = "rule"``.
- **model assertion** — a learned or LLM extractor, deferred behind this
  registry (``method = "model_assertion"``); the generic structured-LLM path
  (#144) remains the way to run one today.

Only the co-occurrence extractor ships here. The grain, the ``method`` column,
and the extractor registry are shaped so the other two plug in without touching
transform execution — the same sequencing used for entity-linking resolvers
(#217). Raw mention text is kept out of the output unless explicitly opted in.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    TypeAdapter,
    field_validator,
    model_validator,
)

from ..hashing import canonical_fingerprint

# Bumped whenever an extractor's semantics change (pairing rules, ordering,
# status assignment) so downstream consumers can invalidate rows produced by an
# older extractor even when the package version moved for unrelated reasons.
CO_OCCURRENCE_EXTRACTOR_VERSION = "1"
RULE_EXTRACTOR_VERSION = "1"
MODEL_ASSERTION_EXTRACTOR_VERSION = "1"

RELATION_FINGERPRINT_DOMAIN = "dbt-ml.entity-relation"

# The method *category* recorded on every row. It is deliberately coarser than
# the extractor name so a consumer can filter "proximity only" without knowing
# which concrete extractor produced the row.
RelationMethod = Literal["co_occurrence", "rule", "model_assertion"]
# A learned extractor emits ``no_relation``/``ambiguous`` for evaluation; the
# deterministic co-occurrence extractor only ever asserts.
RelationStatus = Literal["asserted", "ambiguous", "no_relation"]
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


class _RelationBaseOptions(BaseModel):
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

    @field_validator("label_field")
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
    def _labels_require_label_field(self) -> _RelationBaseOptions:
        if self.labels and self.label_field is None:
            raise ValueError(
                "labels filtering requires label_field; set label_field or clear labels"
            )
        return self


class CoOccurrenceExtractorOptions(_RelationBaseOptions):
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


class RuleExtractorOptions(_RelationBaseOptions):
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


class ModelAssertionExtractorOptions(_RelationBaseOptions):
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


# --- Extractor contract ------------------------------------------------------


@dataclass(frozen=True)
class Mention:
    """One entity mention prepared for pairing. ``text`` is populated only when
    the caller opted into evidence text."""

    mention_id: str
    sentence_index: int | None
    start: int
    end: int
    label: str | None
    text: str | None = None


@dataclass(frozen=True)
class Relation:
    """One extractor-produced relation between two mentions of a document. The
    driver adds the stable ``relation_id`` and extractor identity."""

    subject: Mention
    object: Mention
    relation_type: str
    directed: bool
    status: RelationStatus
    confidence: float | None
    sentence_index: int | None


def _order_key(mention: Mention) -> tuple[int, int, str]:
    """Total order used to give each unordered pair a single, stable
    (subject, object) orientation."""
    return (mention.start, mention.end, mention.mention_id)


def _candidate_pairs(
    mentions: Sequence[Mention],
    *,
    scope: CoOccurrenceScope,
    max_char_gap: int,
    max_pairs: int,
) -> list[tuple[Mention, Mention]]:
    """Ordered ``(earlier, later)`` mention pairs that share a sentence
    (``scope='sentence'``) or fall within ``max_char_gap`` characters
    (``scope='window'``). Fails closed once the pair count would exceed
    ``max_pairs`` so a pathological document cannot explode quadratically.
    Shared by every extractor that pairs mentions in scope."""
    pairs: list[tuple[Mention, Mention]] = []

    def _emit(subject: Mention, obj: Mention) -> None:
        if len(pairs) >= max_pairs:
            raise ValueError(
                f"relation extraction produced more than max_pairs_per_document "
                f"({max_pairs}) candidate pairs for a single document; narrow "
                "`scope`/`labels`/`max_char_gap` or raise the cap"
            )
        pairs.append((subject, obj))

    if scope == "sentence":
        by_sentence: dict[int, list[Mention]] = {}
        for mention in mentions:
            if mention.sentence_index is None:
                raise ValueError(
                    "relation scope 'sentence' requires a non-null sentence_index "
                    "on every mention; rebuild the mention table with a spaCy "
                    "pipeline that sets sentence boundaries, or use scope 'window'"
                )
            by_sentence.setdefault(mention.sentence_index, []).append(mention)
        for sentence_index in sorted(by_sentence):
            group = by_sentence[sentence_index]
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    _emit(group[i], group[j])
        return pairs

    for i in range(len(mentions)):
        subject = mentions[i]
        for j in range(i + 1, len(mentions)):
            obj = mentions[j]
            # Mentions are order-key sorted, so `obj` starts at or after
            # `subject`. Gap is the characters between the spans; overlap
            # (negative gap) clamps in and always qualifies.
            if obj.start - subject.end > max_char_gap:
                continue
            _emit(subject, obj)
    return pairs


def _shared_sentence(subject: Mention, obj: Mention) -> int | None:
    return subject.sentence_index if subject.sentence_index == obj.sentence_index else None


@dataclass(frozen=True)
class RelationAssertion:
    """One provider-returned assertion for a candidate pair, before the extractor
    validates it against the allow-list and candidate set."""

    subject_mention_id: str
    object_mention_id: str
    relation_type: str
    confidence: float


class RelationInference(Protocol):
    """Governed inference for provider-based extractors. The driver builds the
    concrete implementation from the resolved `llm:` profile; unit tests inject a
    deterministic fake. Implementations must not let raw evidence text or provider
    response bodies escape into logs or artifacts."""

    def assert_relations(
        self,
        pairs: Sequence[tuple[Mention, Mention]],
        options: ModelAssertionExtractorOptions,
    ) -> list[RelationAssertion]: ...


class RelationExtractor(ABC):
    name: str
    version: str
    method: RelationMethod

    def text_required(self) -> bool:
        """Whether the mention text column must be present regardless of
        ``include_mention_text``. Co-occurrence pairs on offsets, not text."""
        return False

    def requires_inference(self) -> bool:
        """Whether the driver must supply an inference provider (`infer`).
        Deterministic extractors are offline and return ``False``."""
        return False

    @abstractmethod
    def required_mention_columns(self, options: Any) -> tuple[str, ...]:
        """Extra mention columns this extractor reads beyond id/document/label
        and the evidence offsets the driver always requires."""

    @abstractmethod
    def extract(
        self,
        mentions: Sequence[Mention],
        options: Any,
        *,
        infer: RelationInference | None = None,
    ) -> list[Relation]:
        """Produce the relations for one document's mentions. Mentions are
        pre-filtered by the label allow-list and sorted by ``_order_key``.
        ``infer`` is supplied only when ``requires_inference()`` is true."""


# --- Co-occurrence extractor -------------------------------------------------


class CoOccurrenceExtractor(RelationExtractor):
    name = "co_occurrence"
    version = CO_OCCURRENCE_EXTRACTOR_VERSION
    method: RelationMethod = "co_occurrence"

    def required_mention_columns(self, options: Any) -> tuple[str, ...]:
        return ()

    def extract(
        self,
        mentions: Sequence[Mention],
        options: CoOccurrenceExtractorOptions,
        *,
        infer: RelationInference | None = None,
    ) -> list[Relation]:
        del infer
        if len(mentions) < 2:
            return []
        pairs = _candidate_pairs(
            mentions,
            scope=options.scope,
            max_char_gap=options.max_char_gap,
            max_pairs=options.max_pairs_per_document,
        )
        return [
            Relation(
                subject=subject,
                object=obj,
                relation_type=options.relation_type,
                directed=False,
                status="asserted",
                confidence=None,
                sentence_index=_shared_sentence(subject, obj),
            )
            for subject, obj in pairs
        ]


# --- Rule extractor ----------------------------------------------------------


class RuleExtractor(RelationExtractor):
    """Deterministic typed extractor. For each mention pair in scope it asserts a
    directed relation whenever the pair's labels match a configured rule, in
    either orientation, so the subject/object of the emitted row follow the
    rule's direction rather than text position."""

    name = "rule"
    version = RULE_EXTRACTOR_VERSION
    method: RelationMethod = "rule"

    def required_mention_columns(self, options: Any) -> tuple[str, ...]:
        return ()

    def extract(
        self,
        mentions: Sequence[Mention],
        options: RuleExtractorOptions,
        *,
        infer: RelationInference | None = None,
    ) -> list[Relation]:
        del infer
        if len(mentions) < 2:
            return []
        pairs = _candidate_pairs(
            mentions,
            scope=options.scope,
            max_char_gap=options.max_char_gap,
            max_pairs=options.max_pairs_per_document,
        )
        relations: list[Relation] = []
        for earlier, later in pairs:
            sentence_index = _shared_sentence(earlier, later)
            for rule in options.rules:
                # A rule is directed subject_label -> object_label; try both
                # orientations of the unordered pair so the emitted subject/object
                # follow the rule, not text position.
                for subject, obj in ((earlier, later), (later, earlier)):
                    if (
                        subject.label == rule.subject_label
                        and obj.label == rule.object_label
                    ):
                        relations.append(
                            Relation(
                                subject=subject,
                                object=obj,
                                relation_type=rule.relation_type,
                                directed=True,
                                status="asserted",
                                confidence=None,
                                sentence_index=sentence_index,
                            )
                        )
        return relations


# --- Model-assertion extractor -----------------------------------------------


class ModelAssertionExtractor(RelationExtractor):
    """Learned/LLM extractor (issue #240). Asks a governed inference provider,
    per document, which of the schema-controlled ``relation_types`` hold between
    the in-scope candidate mention pairs. Provider-returned assertions are
    validated against the candidate set and the allow-list before shaping."""

    name = "model_assertion"
    version = MODEL_ASSERTION_EXTRACTOR_VERSION
    method: RelationMethod = "model_assertion"

    def text_required(self) -> bool:
        return True

    def requires_inference(self) -> bool:
        return True

    def required_mention_columns(self, options: Any) -> tuple[str, ...]:
        return ()

    def extract(
        self,
        mentions: Sequence[Mention],
        options: ModelAssertionExtractorOptions,
        *,
        infer: RelationInference | None = None,
    ) -> list[Relation]:
        if infer is None:
            raise ValueError(
                "the model_assertion extractor requires an inference provider; "
                "the driver supplies it from the resolved `llm:` profile"
            )
        if len(mentions) < 2:
            return []
        pairs = _candidate_pairs(
            mentions,
            scope=options.scope,
            max_char_gap=options.max_char_gap,
            max_pairs=options.max_pairs_per_document,
        )
        if not pairs:
            return []

        by_id = {mention.mention_id: mention for mention in mentions}
        candidate_unordered = {
            frozenset((subject.mention_id, obj.mention_id)) for subject, obj in pairs
        }
        allow = set(options.relation_types)

        # Directed (subject, object) -> relation_type -> best confidence. Drop
        # assertions the model hallucinated (unknown mention, non-candidate pair,
        # or out-of-allow-list type) so it can only assert governed relations.
        grouped: dict[tuple[str, str], dict[str, float]] = {}
        for assertion in infer.assert_relations(pairs, options):
            subject_id = assertion.subject_mention_id
            object_id = assertion.object_mention_id
            if subject_id == object_id or subject_id not in by_id or object_id not in by_id:
                continue
            if frozenset((subject_id, object_id)) not in candidate_unordered:
                continue
            if assertion.relation_type not in allow:
                continue
            confidence = float(assertion.confidence)
            types = grouped.setdefault((subject_id, object_id), {})
            if assertion.relation_type not in types or confidence > types[assertion.relation_type]:
                types[assertion.relation_type] = confidence

        relations: list[Relation] = []
        for (subject_id, object_id), types in grouped.items():
            subject = by_id[subject_id]
            obj = by_id[object_id]
            sentence_index = _shared_sentence(subject, obj)
            # A single directed pair mapped to several relation types is a
            # conflict the model could not resolve — preserve it as ambiguous.
            ambiguous = len(types) > 1
            for relation_type, confidence in sorted(types.items()):
                if ambiguous:
                    status: RelationStatus = "ambiguous"
                else:
                    status = "asserted" if confidence >= options.threshold else "no_relation"
                relations.append(
                    Relation(
                        subject=subject,
                        object=obj,
                        relation_type=relation_type,
                        directed=options.directed,
                        status=status,
                        confidence=confidence,
                        sentence_index=sentence_index,
                    )
                )
        return relations


RELATION_EXTRACTORS: dict[str, RelationExtractor] = {
    extractor.name: extractor
    for extractor in (CoOccurrenceExtractor(), RuleExtractor(), ModelAssertionExtractor())
}


def get_relation_extractor(name: str) -> RelationExtractor:
    try:
        return RELATION_EXTRACTORS[name]
    except KeyError as e:  # pragma: no cover - options validation rejects first
        raise ValueError(f"Unknown relation extractor {name!r}") from e


def relation_id(
    *,
    document_id: str,
    subject_mention_id: str,
    object_mention_id: str,
    relation_type: str,
    method: str,
) -> str:
    return canonical_fingerprint(
        {
            "document_id": document_id,
            "subject_mention_id": subject_mention_id,
            "object_mention_id": object_mention_id,
            "relation_type": relation_type,
            "method": method,
        },
        domain=RELATION_FINGERPRINT_DOMAIN,
    )
