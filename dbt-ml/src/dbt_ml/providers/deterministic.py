from __future__ import annotations

import hashlib
import math

from .base import (
    EmbeddingProvider,
    EmbeddingRequest,
    EmbeddingResult,
    ProviderCredential,
    ProviderRuntimeOptions,
    ProviderUsage,
)
from .registry import register_embedding_provider


@register_embedding_provider
class DeterministicEmbeddingProvider(EmbeddingProvider):
    """Offline provider for contract tests and reproducible local examples."""

    provider_name = "deterministic"
    implementation_version = "1"
    requires_credentials = False

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
