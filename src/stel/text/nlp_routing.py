"""Language routing for the NLP child-table transforms (issue #222, ADR-0064).

A project maps languages to provider + model pairs. Each document is resolved to
exactly one route, deterministically, and the decision is recorded on every
output row. Nothing here loads a provider or touches a warehouse: the models are
validated at compile time and `decide_route` is a pure function of the options
and the document's language.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .language import detect_language
from .nlp_registry import (
    is_language_code,
    normalize_language_code,
    primary_language,
)

RoutePolicy = Literal["fail", "skip", "fallback"]
LanguageSource = Literal["declared", "detected"]

FALLBACK_ROUTE_KEY = "fallback"


class NLPRoute(BaseModel):
    """One provider + model, and the language the model is expected to report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = "spacy"
    model: str
    language: str | None = None
    disable: tuple[str, ...] = ()

    @field_validator("provider", "model")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized

    @field_validator("language")
    @classmethod
    def _language_code(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_language_code(value)
        if not is_language_code(normalized):
            raise ValueError("must be a language code such as 'en' or 'zh-cn'")
        return normalized

    @field_validator("disable")
    @classmethod
    def _unique_disable(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized) or len(normalized) != len(
            set(normalized)
        ):
            raise ValueError("disable entries must be unique and not empty")
        return normalized


class NLPLanguageRouting(BaseModel):
    """How documents are mapped to routes.

    `language_field` names an upstream column holding each document's language
    (declared); without it the language is detected from the text with
    `detect_language`. `on_undetected` covers a document whose language is
    unknown (too short to detect, detection failed, or a null declared value);
    `on_unrouted` covers a known language with no route. `fallback` is the route
    used by whichever policy says `fallback`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language_field: str | None = None
    routes: dict[str, NLPRoute] = Field(min_length=1)
    on_undetected: RoutePolicy = "fail"
    on_unrouted: RoutePolicy = "fail"
    fallback: NLPRoute | None = None

    @field_validator("language_field")
    @classmethod
    def _field_not_empty(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized

    @field_validator("routes")
    @classmethod
    def _normalize_route_keys(cls, routes: dict[str, NLPRoute]) -> dict[str, NLPRoute]:
        normalized: dict[str, NLPRoute] = {}
        for key, route in routes.items():
            code = normalize_language_code(key)
            if not is_language_code(code):
                raise ValueError(
                    f"route key '{key}' must be a language code such as 'en' or 'zh-cn'"
                )
            if code in normalized:
                raise ValueError(f"route key '{code}' is declared twice")
            normalized[code] = route
        return normalized

    @model_validator(mode="after")
    def _fallback_matches_policy(self) -> Self:
        wants_fallback = "fallback" in (self.on_undetected, self.on_unrouted)
        if wants_fallback and self.fallback is None:
            raise ValueError("a 'fallback' policy requires a fallback route")
        if self.fallback is not None and not wants_fallback:
            raise ValueError(
                "fallback is set but neither on_undetected nor on_unrouted uses it"
            )
        if self.fallback is not None and self.fallback.language is None:
            raise ValueError(
                "fallback.language is required: there is no language key to infer it from"
            )
        return self

    def route_language(self, key: str) -> str:
        """The language a route's model must report (explicit, else the key's
        primary subtag, since `zh-cn` is served by a `zh` model)."""
        route = self.routes[key]
        return route.language or primary_language(key)


class NLPRoutingError(ValueError):
    """A document's language has no route and the policy is `fail`."""


@dataclass(frozen=True)
class RouteDecision:
    """One document's outcome. `route_key` is a key of `routes`, the literal
    `fallback`, or `None` when the document is skipped."""

    route_key: str | None
    language_source: LanguageSource
    observed_language: str | None


def observed_language(
    routing: NLPLanguageRouting,
    *,
    declared: str | None,
    text: str,
    detect: Callable[[str], str | None] | None = None,
) -> str | None:
    """The document's language as routing sees it, normalized, or `None`.

    Detection is deterministic (`detect_language` pins the detector seed)."""
    if routing.language_field is not None:
        if declared is None or not declared.strip():
            return None
        return normalize_language_code(declared)
    detected = (detect or detect_language)(text)
    return normalize_language_code(detected) if detected else None


def decide_route(
    routing: NLPLanguageRouting,
    *,
    document_id: str,
    observed: str | None,
) -> RouteDecision:
    source: LanguageSource = (
        "declared" if routing.language_field is not None else "detected"
    )
    if observed is None:
        return _apply_policy(
            routing.on_undetected,
            source,
            observed,
            f"Document '{document_id}' has no determinable language",
        )
    for key in (observed, primary_language(observed)):
        if key in routing.routes:
            return RouteDecision(key, source, observed)
    return _apply_policy(
        routing.on_unrouted,
        source,
        observed,
        f"Document '{document_id}' has language '{observed}', which has no route "
        f"(routes: {sorted(routing.routes)})",
    )


def _apply_policy(
    policy: RoutePolicy,
    source: LanguageSource,
    observed: str | None,
    message: str,
) -> RouteDecision:
    if policy == "fail":
        raise NLPRoutingError(
            f"{message}; set on_undetected/on_unrouted to 'skip' or 'fallback' "
            "to handle it"
        )
    if policy == "skip":
        return RouteDecision(None, source, observed)
    return RouteDecision(FALLBACK_ROUTE_KEY, source, observed)
