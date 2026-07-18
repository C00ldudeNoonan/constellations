from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

from .config.model import EmbedConfig, ModelConfig
from .credentials import CredentialReference
from .dag import parse_ref
from .hashing import canonical_fingerprint
from .providers import (
    EmbeddingRequest,
    ProviderConfigurationError,
    ProviderRuntimeOptions,
    ProviderUsage,
    get_embedding_provider,
)


@dataclass(frozen=True, slots=True)
class EmbeddingIdentity:
    provider: str
    model: str
    dimensions: int
    implementation: str
    config_hash: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (
                self.provider,
                self.model,
                self.implementation,
                self.config_hash,
            )
        ):
            raise ValueError("embedding identity is invalid")
        if (
            isinstance(self.dimensions, bool)
            or not isinstance(self.dimensions, int)
            or self.dimensions < 1
        ):
            raise ValueError("embedding identity is invalid")
        if self.config_hash != _embedding_config_hash(
            provider=self.provider,
            model=self.model,
            dimensions=self.dimensions,
            implementation=self.implementation,
        ):
            raise ValueError("embedding identity config_hash does not match its fields")

    @classmethod
    def from_config(cls, config: EmbedConfig) -> Self:
        provider = get_embedding_provider(config.provider)
        implementation = provider.implementation_identity()
        return cls(
            provider=config.provider,
            model=config.model,
            dimensions=config.dimensions,
            implementation=implementation,
            config_hash=_embedding_config_hash(
                provider=config.provider,
                model=config.model,
                dimensions=config.dimensions,
                implementation=implementation,
            ),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Self:
        required = {
            "provider": str,
            "model": str,
            "dimensions": int,
            "implementation": str,
            "config_hash": str,
        }
        if set(value) != set(required) or any(
            isinstance(value[name], bool) or not isinstance(value[name], expected)
            for name, expected in required.items()
        ):
            raise ValueError("embedding identity is invalid")
        return cls(
            provider=value["provider"],
            model=value["model"],
            dimensions=value["dimensions"],
            implementation=value["implementation"],
            config_hash=value["config_hash"],
        )

    def to_dict(self) -> dict[str, str | int]:
        return {
            "provider": self.provider,
            "model": self.model,
            "dimensions": self.dimensions,
            "implementation": self.implementation,
            "config_hash": self.config_hash,
        }


@dataclass(frozen=True, slots=True)
class EmbeddedTexts:
    vectors: tuple[tuple[float, ...], ...]
    usage: ProviderUsage


def resolve_search_embedding_identity(
    model: ModelConfig,
    models: Mapping[str, ModelConfig] | Sequence[ModelConfig],
) -> EmbeddingIdentity | None:
    search = model.search
    if search is None or search.vector is None or search.vector.embedding != "inherit":
        return None
    models_by_name = (
        models if isinstance(models, Mapping) else {item.name: item for item in models}
    )
    dependencies = model.depends_on or []
    if len(dependencies) != 1:
        raise ValueError("inherited search embeddings require exactly one upstream model")
    upstream_name = parse_ref(dependencies[0])
    upstream = models_by_name.get(upstream_name)
    if upstream is None or upstream.embed is None:
        raise ValueError(
            "inherited search embeddings require a direct upstream embed model"
        )
    if search.vector.field != upstream.embed.vector_field:
        raise ValueError(
            "inherited search vector field must match the upstream embed vector field"
        )
    if search.vector.dimensions != upstream.embed.dimensions:
        raise ValueError(
            "inherited search dimensions must match the upstream embed dimensions"
        )
    return EmbeddingIdentity.from_config(upstream.embed)


def effective_search_config(
    model: ModelConfig,
    models: Mapping[str, ModelConfig] | Sequence[ModelConfig],
) -> dict[str, Any]:
    search = model.search
    if search is None:
        raise ValueError("effective search configuration requires a search model")
    payload = search.model_dump(mode="python")
    identity = resolve_search_embedding_identity(model, models)
    if identity is not None:
        vector = dict(payload["vector"])
        vector["embedding"] = identity.to_dict()
        payload["vector"] = vector
    return payload


def _embedding_config_hash(
    *,
    provider: str,
    model: str,
    dimensions: int,
    implementation: str,
) -> str:
    return canonical_fingerprint(
        {
            "provider": provider,
            "model": model,
            "dimensions": dimensions,
            "implementation": implementation,
        },
        domain="embedding-config",
        version=1,
    )


def embed_texts(
    texts: Sequence[str],
    identity: EmbeddingIdentity,
    *,
    input_ids: Sequence[str] | None = None,
    credential_env: str | CredentialReference | None = None,
    max_retries: int = 4,
) -> EmbeddedTexts:
    provider = get_embedding_provider(identity.provider)
    if provider.implementation_identity() != identity.implementation:
        raise ProviderConfigurationError(
            f"Embedding provider '{identity.provider}' implementation no longer "
            "matches the recorded embedding identity",
            safe_for_display=True,
        )
    credential = provider.resolve_credential(credential_env)
    result = provider.embed(
        EmbeddingRequest(
            model=identity.model,
            texts=tuple(texts),
            dimensions=identity.dimensions,
            input_ids=tuple(input_ids) if input_ids is not None else None,
        ),
        credential=credential,
        runtime=ProviderRuntimeOptions(max_retries=max_retries),
    )
    return EmbeddedTexts(result.vectors, result.usage)


def embed_query(
    text: str,
    identity: EmbeddingIdentity | Mapping[str, Any],
    *,
    credential_env: str | CredentialReference | None = None,
    max_retries: int = 4,
) -> tuple[float, ...]:
    resolved = (
        identity
        if isinstance(identity, EmbeddingIdentity)
        else EmbeddingIdentity.from_mapping(identity)
    )
    return embed_texts(
        (text,),
        resolved,
        credential_env=credential_env,
        max_retries=max_retries,
    ).vectors[0]
