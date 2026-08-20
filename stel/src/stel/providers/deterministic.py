from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any

from .base import (
    EmbeddingProvider,
    EmbeddingRequest,
    EmbeddingResult,
    InferenceProvider,
    InferenceRequest,
    InferenceResult,
    ProviderCredential,
    ProviderRuntimeOptions,
    ProviderUsage,
)
from .registry import register_embedding_provider, register_inference_provider


@register_embedding_provider
class DeterministicEmbeddingProvider(EmbeddingProvider):
    """Offline provider for contract tests and reproducible local examples."""

    provider_name = "deterministic"
    implementation_version = "1"
    requires_credentials = False
    accepts_api_key_env = False

    def _embed(
        self,
        request: EmbeddingRequest,
        *,
        credential: ProviderCredential | None,
        runtime: ProviderRuntimeOptions,
    ) -> EmbeddingResult:
        del credential, runtime
        dimensions = request.dimensions or 8
        vectors = tuple(
            _deterministic_vector(request.model, text, dimensions) for text in request.texts
        )
        return EmbeddingResult(
            vectors=vectors,
            model=request.model,
            dimensions=dimensions,
            input_ids=request.input_ids,
            usage=ProviderUsage(input_tokens=sum(len(text.split()) for text in request.texts)),
        )


@register_inference_provider
class DeterministicInferenceProvider(InferenceProvider):
    """Offline structured-completion provider for contract tests and examples.

    Returns schema-shaped values derived deterministically from the model, the
    input content, and the requested field — enough to exercise native `llm:`
    pipelines end to end without a live provider or credentials. Values are
    reproducible but not meaningful; do not use for real extraction.
    """

    provider_name = "deterministic"
    implementation_version = "1"
    requires_credentials = False
    accepts_api_key_env = False
    default_model = "deterministic-v1"

    def complete(
        self,
        request: InferenceRequest,
        *,
        credential: ProviderCredential | None,
        runtime: ProviderRuntimeOptions,
    ) -> InferenceResult:
        del credential, runtime
        output = _deterministic_object(
            request.output_schema,
            model=request.model,
            content=request.content,
            path="",
        )
        if not isinstance(output, dict):
            output = {}
        return InferenceResult(
            output=output,
            usage=ProviderUsage(
                input_tokens=len(request.content.split()),
                output_tokens=len(request.output_schema.get("properties", {})),
            ),
        )


# A fan-out `llm:` model asks for an array of objects; keep the count fixed and
# small so example runs and tests stay fast and deterministic.
_DETERMINISTIC_ARRAY_ITEMS = 2


def _deterministic_object(
    schema: Mapping[str, Any],
    *,
    model: str,
    content: str,
    path: str,
) -> Any:
    node_type = schema.get("type", "string")
    if node_type == "object":
        properties = schema.get("properties", {})
        return {
            name: _deterministic_object(
                prop if isinstance(prop, Mapping) else {},
                model=model,
                content=content,
                path=f"{path}.{name}",
            )
            for name, prop in properties.items()
        }
    if node_type == "array":
        items_schema = schema.get("items", {"type": "string"})
        return [
            _deterministic_object(
                items_schema if isinstance(items_schema, Mapping) else {},
                model=model,
                content=content,
                path=f"{path}[{index}]",
            )
            for index in range(_DETERMINISTIC_ARRAY_ITEMS)
        ]
    return _deterministic_scalar(node_type, model=model, content=content, path=path)


def _deterministic_scalar(
    node_type: str,
    *,
    model: str,
    content: str,
    path: str,
) -> Any:
    digest = hashlib.blake2b(
        f"{model}\0{content}\0{path}".encode(), digest_size=8
    ).digest()
    number = int.from_bytes(digest, "big")
    if node_type == "integer":
        return number % 1000
    if node_type == "number":
        return round((number % 100000) / 100.0, 2)
    if node_type == "boolean":
        return bool(number & 1)
    return f"det-{digest.hex()[:8]}"


def _deterministic_vector(
    model: str,
    text: str,
    dimensions: int,
) -> tuple[float, ...]:
    values: list[float] = []
    counter = 0
    while len(values) < dimensions:
        digest = hashlib.blake2b(
            f"{model}\0{text}\0{counter}".encode(),
            digest_size=32,
        ).digest()
        values.extend((byte / 127.5) - 1.0 for byte in digest)
        counter += 1
    selected = values[:dimensions]
    norm = math.sqrt(sum(value * value for value in selected)) or 1.0
    return tuple(value / norm for value in selected)
