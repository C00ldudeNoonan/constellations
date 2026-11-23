"""Typed relation extraction over the NLP entity-mention child table (issue #220).

A *relation* connects two entity mentions in the same document. Three
extractors ship here, kept strictly distinguishable so a consumer never
mistakes proximity for a semantic assertion:

- **co-occurrence** — the deterministic, offline built-in. Two mentions
  co-occur when they share a sentence (``scope: sentence``) or fall within a
  character window (``scope: window``). This asserts nothing beyond "these
  mentions appeared together"; every row is labelled ``method = "co_occurrence"``
  and is symmetric (``directed = false``).
- **rule-derived** (issue #217) — a deterministic extractor that asserts a
  typed, directed relation from explicit ``RelationRule`` entries
  (``method = "rule"``).
- **model assertion** (issue #240) — a governed inference provider decides
  which of a schema-controlled set of relation types holds between each
  candidate pair (``method = "model_assertion"``).

The grain, the ``method`` column, and the extractor registry let a consumer
filter "proximity only" without knowing which concrete extractor produced a
row. A project's `classes:`/`relations:` declaration (issue #626) checks a
rule's relation type and subject/object labels, and a model-assertion
extractor's allowed relation types, against a declared domain/range — see
`relation_contracts.py`. Raw mention text is kept out of the output unless
explicitly opted in.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from ..hashing import canonical_fingerprint
from ..relation_options import (
    CoOccurrenceExtractorOptions,
    CoOccurrenceScope,
    ModelAssertionExtractorOptions,
    RuleExtractorOptions,
)

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
