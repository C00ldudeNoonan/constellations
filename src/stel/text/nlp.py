"""Optional NLP provider contracts and the spaCy implementation."""
from __future__ import annotations

import functools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)

from ..optional_dependencies import (
    OptionalDependencyError,
    import_optional_dependency,
    optional_dependency_version,
)
from .nlp_registry import (
    NLPModelSpec,
    NLPProviderRegistration,
    get_nlp_registration,
    register_nlp_provider,
)
from .nlp_routing import NLPLanguageRouting, NLPRoute


class NLPError(OptionalDependencyError):
    """An actionable NLP dependency, model, or provider error."""


class NLPBaseOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # What a transform needs from its provider. Checked against the registry's
    # declared capabilities at compile time (issue #222).
    required_capabilities: ClassVar[frozenset[str]] = frozenset()

    provider: str = "spacy"
    model: str = "en_core_web_sm"
    language: str = "en"
    disable: tuple[str, ...] = ()
    batch_size: int = Field(default=32, ge=1, le=10_000)
    document_id_field: str = "document_id"
    text_field: str = "text"
    include_fields: tuple[str, ...] = ()
    language_routing: NLPLanguageRouting | None = None

    @field_validator("provider", "model", "language", "document_id_field", "text_field")
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

    @model_validator(mode="after")
    def _routing_replaces_single_model(self) -> NLPBaseOptions:
        if self.language_routing is None:
            return self
        conflicting = sorted(
            self.model_fields_set & {"provider", "model", "language", "disable"}
        )
        if conflicting:
            raise ValueError(
                "language_routing replaces the single provider/model: move "
                f"{conflicting} into routes (or the fallback) instead"
            )
        return self

    @model_validator(mode="after")
    def _routes_are_supported(self) -> NLPBaseOptions:
        """Fail on an unknown provider, a missing capability or an unsupported
        language here, before any source is read or model loaded."""
        for label, route, language in self.routes_to_validate():
            registration = get_nlp_registration(route.provider)
            missing = sorted(self.required_capabilities - registration.capabilities)
            if missing:
                raise ValueError(
                    f"{label}: provider '{route.provider}' does not provide "
                    f"{missing}, which this transform needs"
                )
            if not registration.supports_language(language):
                raise ValueError(
                    f"{label}: provider '{route.provider}' does not support "
                    f"language '{language}'"
                )
        return self

    def routes_to_validate(self) -> list[tuple[str, NLPRoute, str]]:
        """Every (label, route, expected language) this configuration can use."""
        routing = self.language_routing
        if routing is None:
            single = NLPRoute(
                provider=self.provider,
                model=self.model,
                language=self.language,
                disable=self.disable,
            )
            return [("transform", single, self.language)]
        entries = [
            (f"route '{key}'", route, routing.route_language(key))
            for key, route in routing.routes.items()
        ]
        fallback = routing.fallback
        if fallback is not None and fallback.language is not None:
            entries.append(("fallback route", fallback, fallback.language))
        return entries


class NLPTokenOptions(NLPBaseOptions):
    required_capabilities: ClassVar[frozenset[str]] = frozenset(
        ("tokens", "lemmas", "pos")
    )

    include_space: bool = False


class NLPEntityOptions(NLPBaseOptions):
    required_capabilities: ClassVar[frozenset[str]] = frozenset(("entities",))

    include_text: StrictBool = False


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
    """The provider for a single-model (unrouted) configuration."""
    route = NLPRoute(
        provider=options.provider,
        model=options.model,
        language=options.language,
        disable=options.disable,
    )
    return load_nlp_route(route, options.language)


def load_nlp_route(route: NLPRoute, language: str) -> NLPProvider:
    """Build the provider a route names, through the registry only."""
    registration = get_nlp_registration(route.provider)
    provider = registration.loader(
        NLPModelSpec(model=route.model, language=language, disable=route.disable)
    )
    if provider.identity.provider != registration.name:
        raise NLPError(
            f"NLP provider '{registration.name}' returned a provider that "
            f"identifies as '{provider.identity.provider}'"
        )
    return provider


def _load_spacy(spec: NLPModelSpec) -> NLPProvider:
    return _spacy_provider(spec.model, spec.language, spec.disable)


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


register_nlp_provider(
    NLPProviderRegistration(
        name="spacy",
        loader=_load_spacy,
        # spaCy has no per-entity confidence in the pipelines stel reads.
        capabilities=frozenset(("tokens", "lemmas", "pos", "entities", "sentences")),
        # The installed model decides; the language is checked when it loads.
        languages=None,
    )
)
