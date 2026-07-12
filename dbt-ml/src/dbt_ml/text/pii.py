"""PII detection + redaction via Microsoft Presidio.

Presidio uses spaCy under the hood for NER-backed entity types (PERSON,
LOCATION, ORGANIZATION). The default English model is `en_core_web_lg`;
dbt_ml's wrapper defaults to `en_core_web_sm` for a much smaller install.

First-time setup:

    python -m spacy download en_core_web_sm

If the model is missing, calls into Presidio will raise a clearer
PIIError pointing at the install command.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from ..optional_dependencies import (
    OptionalDependencyError,
    import_optional_dependency,
)

if TYPE_CHECKING:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine


class PIIError(OptionalDependencyError):
    pass


@dataclass(frozen=True)
class PIIEntity:
    """A single detected PII span in a text."""

    type: str       # PHONE_NUMBER, EMAIL_ADDRESS, PERSON, US_SSN, ...
    start: int      # char offset in original text (inclusive)
    end: int        # char offset (exclusive)
    score: float    # 0.0–1.0 Presidio confidence
    text: str       # the substring that was matched

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        """Serialize detection metadata, excluding matched text by default."""
        result = {
            "type": self.type,
            "start": self.start,
            "end": self.end,
            "score": round(self.score, 4),
        }
        if include_text:
            result["text"] = self.text
        return result


@functools.lru_cache(maxsize=1)
def _get_analyzer(model: str = "en_core_web_sm") -> AnalyzerEngine:
    try:
        analyzer_module = import_optional_dependency(
            "presidio_analyzer", extra="pii", feature="PII detection"
        )
        nlp_module = import_optional_dependency(
            "presidio_analyzer.nlp_engine", extra="pii", feature="PII detection"
        )
    except OptionalDependencyError as error:
        raise PIIError(str(error)) from error

    try:
        nlp_engine = nlp_module.NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": model}],
            }
        ).create_engine()
    except OSError as e:
        raise PIIError(
            f"spaCy model '{model}' is not installed. "
            f"Run: python -m spacy download {model}"
        ) from e

    return cast(
        "AnalyzerEngine",
        analyzer_module.AnalyzerEngine(
            nlp_engine=nlp_engine, supported_languages=["en"]
        ),
    )


@functools.lru_cache(maxsize=1)
def _get_anonymizer() -> AnonymizerEngine:
    try:
        module = import_optional_dependency(
            "presidio_anonymizer", extra="pii", feature="PII redaction"
        )
    except OptionalDependencyError as error:
        raise PIIError(str(error)) from error
    return cast("AnonymizerEngine", module.AnonymizerEngine())


def detect_pii(
    text: str,
    *,
    entities: list[str] | None = None,
    language: str = "en",
    score_threshold: float = 0.0,
    spacy_model: str = "en_core_web_sm",
) -> list[PIIEntity]:
    """Return PII spans found in `text`.

    `entities` is an optional allow-list (e.g. ["EMAIL_ADDRESS", "PHONE_NUMBER"]);
    pass None to detect all of Presidio's default entity types.
    `score_threshold` filters out low-confidence matches.
    """
    if not text:
        return []
    analyzer = _get_analyzer(spacy_model)
    results = analyzer.analyze(
        text=text,
        entities=entities,
        language=language,
        score_threshold=score_threshold,
    )
    return [
        PIIEntity(
            type=r.entity_type,
            start=r.start,
            end=r.end,
            score=float(r.score),
            text=text[r.start : r.end],
        )
        for r in results
    ]


def redact_pii(
    text: str,
    *,
    entities: list[str] | None = None,
    language: str = "en",
    score_threshold: float = 0.0,
    replacement: str = "[{type}]",
    spacy_model: str = "en_core_web_sm",
) -> tuple[str, list[PIIEntity]]:
    """Replace PII spans in `text` with `replacement`. Returns (redacted_text, detected_entities).

    `replacement` may include `{type}` which is substituted with the entity type
    (e.g. "[PHONE_NUMBER]"). Pass a literal like "[REDACTED]" to ignore the type.
    """
    if not text:
        return text, []

    detected = detect_pii(
        text,
        entities=entities,
        language=language,
        score_threshold=score_threshold,
        spacy_model=spacy_model,
    )
    if not detected:
        return text, []

    # Presidio can emit overlapping spans (e.g. EMAIL_ADDRESS containing a URL).
    # Collapse to highest-score per cluster so replacement offsets don't corrupt.
    merged = _merge_overlapping(detected)

    out = text
    for e in sorted(merged, key=lambda x: x.start, reverse=True):
        marker = replacement.format(type=e.type)
        out = out[: e.start] + marker + out[e.end :]
    return out, merged


def _merge_overlapping(spans: list[PIIEntity]) -> list[PIIEntity]:
    """Collapse overlapping spans, keeping the highest-scoring one per cluster."""
    if not spans:
        return spans
    ordered = sorted(spans, key=lambda s: (s.start, -s.score))
    out: list[PIIEntity] = [ordered[0]]
    for s in ordered[1:]:
        last = out[-1]
        if s.start < last.end:
            # Overlap — keep whichever has higher score
            if s.score > last.score:
                out[-1] = s
        else:
            out.append(s)
    return out
