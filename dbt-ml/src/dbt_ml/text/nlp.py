"""Optional NLP provider contracts and the spaCy implementation."""
from __future__ import annotations

import functools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..optional_dependencies import (
    OptionalDependencyError,
    import_optional_dependency,
    optional_dependency_version,
)


class NLPError(OptionalDependencyError):
    """An actionable NLP dependency, model, or provider error."""


class NLPBaseOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["spacy"] = "spacy"
    model: str = "en_core_web_sm"
    language: str = "en"
    disable: tuple[str, ...] = ()
    batch_size: int = Field(default=32, ge=1, le=10_000)
    document_id_field: str = "document_id"
    text_field: str = "text"
    include_fields: tuple[str, ...] = ()

    @field_validator("model", "language", "document_id_field", "text_field")
    @classmethod
    def _non_empty_string(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized

    @field_validator("disable", "include_fields")
    @classmethod
    def _unique_string_list(
        cls, values: tuple[str, ...], info: Any
    ) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError(f"{info.field_name} entries must not be empty")
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"{info.field_name} entries must be unique")
        return normalized

    @model_validator(mode="after")
    def _safe_projection(self) -> NLPBaseOptions:
        forbidden = {
            field
            for field in (self.document_id_field, self.text_field)
            if field in self.include_fields
        }
        if forbidden:
            raise ValueError(
                "include_fields must not repeat the document ID or raw text field: "
                f"{sorted(forbidden)}"
            )
        return self


class NLPTokenOptions(NLPBaseOptions):
    include_space: bool = False


class NLPEntityOptions(NLPBaseOptions):
    include_text: bool = False


@dataclass(frozen=True)
class NLPIdentity:
    provider: str
    provider_version: str
    model: str
    model_version: str
    language: str


@dataclass(frozen=True)
class NLPToken:
    index: int
    sentence_index: int | None
    start: int
    end: int
    text: str
    lemma: str
    pos: str
    tag: str
    is_stop: bool
    is_alpha: bool
    is_space: bool


@dataclass(frozen=True)
class NLPEntity:
    index: int
    sentence_index: int | None
    start: int
    end: int
    label: str
    text: str
    confidence: float | None = None


@dataclass(frozen=True)
class NLPDocument:
    tokens: tuple[NLPToken, ...]
    entities: tuple[NLPEntity, ...]


class NLPProvider(Protocol):
    @property
    def identity(self) -> NLPIdentity: ...

    def pipe(
        self, texts: Iterable[str], *, batch_size: int
    ) -> Iterable[NLPDocument]: ...


class _SpacyPipeline(Protocol):
    lang: str
    meta: Mapping[str, Any]

    def pipe(self, texts: Iterable[str], *, batch_size: int) -> Iterable[Any]: ...


@dataclass(frozen=True)
class SpacyNLPProvider:
    pipeline: _SpacyPipeline
    identity: NLPIdentity

    def pipe(
        self, texts: Iterable[str], *, batch_size: int
    ) -> Iterable[NLPDocument]:
        for document in self.pipeline.pipe(texts, batch_size=batch_size):
            yield _convert_spacy_document(document)


def get_nlp_provider(options: NLPBaseOptions) -> NLPProvider:
    if options.provider != "spacy":
        raise NLPError(f"Unsupported NLP provider: {options.provider}")
    return _spacy_provider(options.model, options.language, options.disable)


@functools.lru_cache(maxsize=8)
def _spacy_provider(
    model: str,
    language: str,
    disable: tuple[str, ...],
) -> NLPProvider:
    try:
        spacy = import_optional_dependency(
            "spacy",
            extra="nlp",
            feature="NLP enrichment",
        )
    except OptionalDependencyError as error:
        raise NLPError(str(error)) from error

    try:
        pipeline = cast(
            "_SpacyPipeline",
            spacy.load(model, disable=list(disable)),
        )
    except OSError as error:
        raise NLPError(
            f"spaCy model '{model}' is not installed. "
            f"Run: python -m spacy download {model}"
        ) from error

    actual_language = str(pipeline.lang or pipeline.meta.get("lang", "unknown"))
    if actual_language != language:
        raise NLPError(
            f"spaCy model '{model}' uses language '{actual_language}', "
            f"but transform.language is '{language}'"
        )
    identity = NLPIdentity(
        provider="spacy",
        provider_version=optional_dependency_version("spacy"),
        model=model,
        model_version=str(pipeline.meta.get("version", "unknown")),
        language=actual_language,
    )
    return SpacyNLPProvider(pipeline=pipeline, identity=identity)


def _convert_spacy_document(document: Any) -> NLPDocument:
    sentence_by_token: dict[int, int] = {}
    try:
        for sentence_index, sentence in enumerate(document.sents):
            for token in sentence:
                sentence_by_token[int(token.i)] = sentence_index
    except ValueError:
        # Some deliberately disabled pipelines do not establish sentence
        # boundaries. The schema makes sentence_index nullable for this case.
        pass

    tokens = tuple(
        NLPToken(
            index=int(token.i),
            sentence_index=sentence_by_token.get(int(token.i)),
            start=int(token.idx),
            end=int(token.idx) + len(str(token.text)),
            text=str(token.text),
            lemma=str(token.lemma_),
            pos=str(token.pos_),
            tag=str(token.tag_),
            is_stop=bool(token.is_stop),
            is_alpha=bool(token.is_alpha),
            is_space=bool(token.is_space),
        )
        for token in document
    )
    entities = tuple(
        NLPEntity(
            index=index,
            sentence_index=sentence_by_token.get(int(entity.start)),
            start=int(entity.start_char),
            end=int(entity.end_char),
            label=str(entity.label_),
            text=str(entity.text),
        )
        for index, entity in enumerate(document.ents)
    )
    return NLPDocument(tokens=tokens, entities=entities)
