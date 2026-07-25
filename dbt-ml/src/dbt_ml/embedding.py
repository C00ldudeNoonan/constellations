from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Self

from .config.model import EmbedConfig, ModelConfig
from .credentials import CredentialReference
from .dag import parse_ref
from .hashing import canonical_fingerprint
from .profile import (
    ResolvedEmbeddingOptions,
    ResolvedProfile,
    resolve_embedding_options,
)
from .providers import (
    EmbeddingRequest,
    ProviderConfigurationError,
    ProviderRuntimeOptions,
    ProviderUsage,
    get_embedding_provider,
    profile_options_fingerprint,
)


@dataclass(frozen=True, slots=True)
class EmbeddingIdentity:
    provider: str
    model: str
    dimensions: int
    implementation: str
    config_hash: str
    provider_options_identity: str | None = None

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
        if self.provider_options_identity is not None and (
            not isinstance(self.provider_options_identity, str)
            or not self.provider_options_identity
        ):
            raise ValueError("embedding identity is invalid")
        if self.config_hash != _embedding_config_hash(
            provider=self.provider,
            model=self.model,
            dimensions=self.dimensions,
            implementation=self.implementation,
            provider_options_identity=self.provider_options_identity,
        ):
            raise ValueError("embedding identity config_hash does not match its fields")

    @classmethod
    def from_config(
        cls,
        config: EmbedConfig,
        *,
        profile_options: Mapping[str, Any] | None = None,
    ) -> Self:
        provider = get_embedding_provider(
            config.provider,
            profile_options=profile_options,
        )
        implementation = provider.implementation_identity()
        options_identity = profile_options_fingerprint(provider.profile_options)
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
                provider_options_identity=options_identity,
            ),
            provider_options_identity=options_identity,
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
        optional = {"provider_options_identity": str}
        if not set(value).issubset(set(required) | set(optional)) or not set(
            required
        ).issubset(value) or any(
            isinstance(value[name], bool) or not isinstance(value[name], expected)
            for name, expected in required.items()
        ):
            raise ValueError("embedding identity is invalid")
        options_identity = value.get("provider_options_identity")
        if options_identity is not None and (
            not isinstance(options_identity, str) or not options_identity
        ):
            raise ValueError("embedding identity is invalid")
        return cls(
            provider=value["provider"],
            model=value["model"],
            dimensions=value["dimensions"],
            implementation=value["implementation"],
            config_hash=value["config_hash"],
            provider_options_identity=options_identity,
        )

    def to_dict(self) -> dict[str, str | int]:
        payload: dict[str, str | int] = {
            "provider": self.provider,
            "model": self.model,
            "dimensions": self.dimensions,
            "implementation": self.implementation,
            "config_hash": self.config_hash,
        }
        if self.provider_options_identity is not None:
            payload["provider_options_identity"] = self.provider_options_identity
        return payload


@dataclass(frozen=True, slots=True)
class EmbeddedTexts:
    vectors: tuple[tuple[float, ...], ...]
    usage: ProviderUsage
    provider_requests: int


def _model_index(
    models: Mapping[str, ModelConfig] | Sequence[ModelConfig],
) -> Mapping[str, ModelConfig]:
    if isinstance(models, Mapping):
        index: dict[str, ModelConfig] = {}
        for name, model in models.items():
            if not isinstance(name, str) or not isinstance(model, ModelConfig):
                raise TypeError("model index must map names to ModelConfig values")
            index[name] = model
        return index
    return {item.name: item for item in models}


def resolve_search_embedding_identity(
    model: ModelConfig,
    models: Mapping[str, ModelConfig] | Sequence[ModelConfig],
    *,
    profile_options: Mapping[str, Any] | None = None,
) -> EmbeddingIdentity | None:
    search = model.search
    if search is None or search.vector is None or search.vector.embedding != "inherit":
        return None
    models_by_name = _model_index(models)
    dependencies = model.depends_on or []
    if len(dependencies) != 1:
        raise ValueError("inherited search embeddings require exactly one upstream model")
    upstream_name = parse_ref(dependencies[0])
    upstream: ModelConfig | None = models_by_name.get(upstream_name)
    if upstream is None:
        raise ValueError(
            "inherited search embeddings require a direct upstream embed model"
        )
    embed_config = upstream.embed
    if embed_config is None:
        raise ValueError(
            "inherited search embeddings require a direct upstream embed model"
        )
    if search.vector.field != embed_config.vector_field:
        raise ValueError(
            "inherited search vector field must match the upstream embed vector field"
        )
    if search.vector.dimensions != embed_config.dimensions:
        raise ValueError(
            "inherited search dimensions must match the upstream embed dimensions"
        )
    return EmbeddingIdentity.from_config(
        embed_config,
        profile_options=profile_options,
    )


def effective_search_config(
    model: ModelConfig,
    models: Mapping[str, ModelConfig] | Sequence[ModelConfig],
    *,
    profile_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    search = model.search
    if search is None:
        raise ValueError("effective search configuration requires a search model")
    payload = search.model_dump(mode="python")
    identity = resolve_search_embedding_identity(
        model,
        models,
        profile_options=profile_options,
    )
    if identity is not None:
        vector = dict(payload["vector"])
        vector["embedding"] = identity.to_dict()
        payload["vector"] = vector
    return payload


def resolve_search_embedding_options(
    model: ModelConfig,
    models: Mapping[str, ModelConfig] | Sequence[ModelConfig],
    resolved: ResolvedProfile,
) -> ResolvedEmbeddingOptions | None:
    search = model.search
    if search is None or search.vector is None or search.vector.embedding != "inherit":
        return None
    dependencies = model.depends_on or []
    if len(dependencies) != 1:
        return None
    models_by_name = _model_index(models)
    upstream: ModelConfig | None = models_by_name.get(parse_ref(dependencies[0]))
    if upstream is None:
        return None
    embed_config = upstream.embed
    if embed_config is None:
        return None
    return resolve_embedding_options(embed_config.provider, resolved)


def _embedding_config_hash(
    *,
    provider: str,
    model: str,
    dimensions: int,
    implementation: str,
    provider_options_identity: str | None = None,
) -> str:
    return canonical_fingerprint(
        {
            "provider": provider,
            "model": model,
            "dimensions": dimensions,
            "implementation": implementation,
            "provider_options_identity": provider_options_identity,
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
    profile_options: Mapping[str, Any] | None = None,
    max_retries: int = 4,
    timeout_seconds: float = 60.0,
    input_type: Literal["document", "query"] = "document",
) -> EmbeddedTexts:
    provider = get_embedding_provider(
        identity.provider,
        profile_options=profile_options,
    )
    if provider.implementation_identity() != identity.implementation:
        raise ProviderConfigurationError(
            f"Embedding provider '{identity.provider}' implementation no longer "
            "matches the recorded embedding identity",
            safe_for_display=True,
        )
    options_identity = profile_options_fingerprint(provider.profile_options)
    if options_identity != identity.provider_options_identity:
        raise ProviderConfigurationError(
            f"Embedding provider '{identity.provider}' profile options no longer "
            "match the recorded embedding identity",
            safe_for_display=True,
        )
    credential = provider.resolve_credential(credential_env)
    result = provider.embed(
        EmbeddingRequest(
            model=identity.model,
            texts=tuple(texts),
            dimensions=identity.dimensions,
            input_ids=tuple(input_ids) if input_ids is not None else None,
            input_type=input_type,
        ),
        credential=credential,
        runtime=ProviderRuntimeOptions(
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
        ),
    )
    return EmbeddedTexts(result.vectors, result.usage, result.provider_requests)


def embed_query(
    text: str,
    identity: EmbeddingIdentity | Mapping[str, Any],
    *,
    credential_env: str | CredentialReference | None = None,
    profile_options: Mapping[str, Any] | None = None,
    max_retries: int = 4,
    timeout_seconds: float = 60.0,
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
        profile_options=profile_options,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
        input_type="query",
    ).vectors[0]
