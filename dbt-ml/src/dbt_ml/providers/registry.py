from __future__ import annotations

import inspect
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import EntryPoint, distributions
from typing import Any, Protocol, overload

from pydantic import BaseModel

from ..endpoints import EndpointUrlError, OpenAICompatibleBaseUrl
from .base import (
    PROVIDER_CONTRACT_VERSION,
    BaseProvider,
    EmbeddingProvider,
    InferenceProvider,
    ProviderConfigurationError,
    ProviderNotFoundError,
    ProviderRegistrationError,
    implementation_identity_for,
    parse_profile_options,
    validate_profile_options_model,
)

_INFERENCE_PROVIDERS: dict[str, type[InferenceProvider]] = {}
_EMBEDDING_PROVIDERS: dict[str, type[EmbeddingProvider]] = {}
_PROVIDER_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_ENV_VAR_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DISTRIBUTION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_IMPLEMENTATION_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class _InferenceProviderDecorator(Protocol):
    def __call__[T: InferenceProvider](self, provider_cls: type[T], /) -> type[T]: ...


class _EmbeddingProviderDecorator(Protocol):
    def __call__[T: EmbeddingProvider](self, provider_cls: type[T], /) -> type[T]: ...

# Entry-point plugin discovery (issue #71). The group suffix is the provider
# contract major version: a plugin advertises the contract it was built
# against by choosing the group and never loads under any other.
_GROUP_BASES: dict[str, str] = {
    "inference": "dbt_ml.inference_providers",
    "embedding": "dbt_ml.embedding_providers",
}
_GROUP_VERSION = re.compile(r"\.v(\d+)$")
# capability -> provider name -> (distribution, advertised contract version)
_INCOMPATIBLE_PLUGINS: dict[str, dict[str, tuple[str, str]]] = {
    "inference": {},
    "embedding": {},
}
# (capability, provider name) -> distribution, for diagnostics only
_PLUGIN_DISTRIBUTIONS: dict[tuple[str, str], str] = {}
_DISCOVERY_COMPLETE = False


def entry_point_group(capability: str) -> str:
    return f"{_GROUP_BASES[capability]}.v{PROVIDER_CONTRACT_VERSION}"


@overload
def register_inference_provider[T: InferenceProvider](
    provider_cls: type[T], /
) -> type[T]: ...


@overload
def register_inference_provider(
    provider_cls: None = None, /
) -> _InferenceProviderDecorator: ...


def register_inference_provider(
    provider_cls: type[InferenceProvider] | None = None, /
) -> (
    type[InferenceProvider]
    | _InferenceProviderDecorator
):
    def decorator[T: InferenceProvider](cls: type[T]) -> type[T]:
        _validate_provider_class(cls, capability="inference")
        _validate_inference_metadata(cls)
        registered_cls: type[InferenceProvider] = cls
        _register(_INFERENCE_PROVIDERS, registered_cls, capability="inference")
        return cls

    return decorator if provider_cls is None else decorator(provider_cls)


@overload
def register_embedding_provider[T: EmbeddingProvider](
    provider_cls: type[T], /
) -> type[T]: ...


@overload
def register_embedding_provider(
    provider_cls: None = None, /
) -> _EmbeddingProviderDecorator: ...


def register_embedding_provider(
    provider_cls: type[EmbeddingProvider] | None = None, /
) -> (
    type[EmbeddingProvider]
    | _EmbeddingProviderDecorator
):
    def decorator[T: EmbeddingProvider](cls: type[T]) -> type[T]:
        _validate_provider_class(cls, capability="embedding")
        registered_cls: type[EmbeddingProvider] = cls
        _register(_EMBEDDING_PROVIDERS, registered_cls, capability="embedding")
        return cls

    return decorator if provider_cls is None else decorator(provider_cls)


def discover_providers(*, force: bool = False) -> None:
    """Load separately packaged providers from versioned entry-point groups.

    Runs once, deterministically (entry-point name, then distribution), and
    fails closed before any source, credential, or provider I/O: duplicate
    names, built-in shadowing, name mismatches, and load failures raise
    instead of being skipped. Plugins advertising only other contract
    versions are recorded so lookups report a version mismatch rather than
    "not found"."""
    global _DISCOVERY_COMPLETE
    if _DISCOVERY_COMPLETE and not force:
        return
    # Scan distributions directly: the stdlib entry_points() view collapses
    # same-named entry points across distributions, which would let one
    # plugin silently shadow another instead of failing the duplicate.
    current: dict[str, list[EntryPoint]] = {name: [] for name in _GROUP_BASES}
    other_versions: dict[str, list[tuple[EntryPoint, str]]] = {
        name: [] for name in _GROUP_BASES
    }
    for dist in distributions():
        for item in dist.entry_points:
            for capability, base in _GROUP_BASES.items():
                if item.group == entry_point_group(capability):
                    current[capability].append(item)
                elif item.group.startswith(base + "."):
                    match = _GROUP_VERSION.search(item.group)
                    version = match.group(1) if match else "unknown"
                    other_versions[capability].append((item, version))
    for capability in _GROUP_BASES:
        for item in sorted(
            current[capability], key=lambda ep: (ep.name, _distribution(ep))
        ):
            _load_plugin(item, capability=capability)
        for item, version in sorted(
            other_versions[capability],
            key=lambda pair: (pair[0].name, _distribution(pair[0])),
        ):
            _INCOMPATIBLE_PLUGINS[capability].setdefault(
                item.name, (_distribution(item), version)
            )
    _DISCOVERY_COMPLETE = True


def _distribution(item: EntryPoint) -> str:
    dist = getattr(item, "dist", None)
    name = getattr(dist, "name", None)
    return name if isinstance(name, str) and name else "<unknown distribution>"


def _load_plugin(item: EntryPoint, *, capability: str) -> None:
    distribution = _distribution(item)
    registry: dict[str, type[Any]] = (
        _INFERENCE_PROVIDERS if capability == "inference" else _EMBEDDING_PROVIDERS
    )
    try:
        loaded = item.load()
    except Exception as error:
        raise ProviderRegistrationError(
            f"{capability} provider entry point '{item.name}' from distribution "
            f"'{distribution}' failed to load [{type(error).__name__}]"
        ) from None
    provider_name = getattr(loaded, "provider_name", None)
    if provider_name != item.name:
        raise ProviderRegistrationError(
            f"{capability} provider entry point '{item.name}' from distribution "
            f"'{distribution}' names a provider class with provider_name="
            f"{provider_name!r}; the entry-point name must match"
        )
    existing = registry.get(item.name)
    if existing is not None and existing is not loaded:
        other = _PLUGIN_DISTRIBUTIONS.get((capability, item.name), "built-in")
        raise ProviderRegistrationError(
            f"{capability} provider '{item.name}' is claimed by both "
            f"'{other}' and '{distribution}'; provider names must be unique"
        )
    try:
        if capability == "inference":
            register_inference_provider(loaded)
        else:
            register_embedding_provider(loaded)
    except ProviderRegistrationError as error:
        raise ProviderRegistrationError(
            f"distribution '{distribution}': {error}"
        ) from None
    _PLUGIN_DISTRIBUTIONS[(capability, item.name)] = distribution


def _missing_provider_error(
    name: str, *, capability: str, registry: dict[str, type[Any]]
) -> ProviderNotFoundError | ProviderConfigurationError:
    display_name = name if _is_provider_name(name) else "<invalid>"
    incompatible = _INCOMPATIBLE_PLUGINS[capability].get(name) if isinstance(
        name, str
    ) else None
    if incompatible is not None:
        distribution, found_version = incompatible
        return ProviderConfigurationError(
            f"{capability} provider '{display_name}' from distribution "
            f"'{distribution}' targets provider contract v{found_version}, but "
            f"this dbt-ml release requires v{PROVIDER_CONTRACT_VERSION}; "
            "install a compatible plugin release",
            safe_for_display=True,
        )
    return ProviderNotFoundError(
        f"{capability.capitalize()} provider '{display_name}' is not registered. "
        f"Available: {sorted(registry)}"
    )


def get_inference_provider(
    name: str, *, profile_options: Mapping[str, Any] | None = None
) -> InferenceProvider:
    discover_providers()
    cls = _INFERENCE_PROVIDERS.get(name) if isinstance(name, str) else None
    if cls is None:
        raise _missing_provider_error(
            name, capability="inference", registry=_INFERENCE_PROVIDERS
        )
    return _instantiate(
        cls, name=name, capability="inference", profile_options=profile_options
    )


def get_embedding_provider(
    name: str, *, profile_options: Mapping[str, Any] | None = None
) -> EmbeddingProvider:
    discover_providers()
    cls = _EMBEDDING_PROVIDERS.get(name) if isinstance(name, str) else None
    if cls is None:
        raise _missing_provider_error(
            name, capability="embedding", registry=_EMBEDDING_PROVIDERS
        )
    return _instantiate(
        cls, name=name, capability="embedding", profile_options=profile_options
    )


def list_inference_providers() -> list[str]:
    discover_providers()
    return sorted(_INFERENCE_PROVIDERS)


def list_embedding_providers() -> list[str]:
    discover_providers()
    return sorted(_EMBEDDING_PROVIDERS)


@dataclass(frozen=True)
class ProviderInventoryEntry:
    """Safe descriptor of one discovered provider for `dbt-ml providers list`."""

    capability: str
    name: str
    distribution: str
    status: str
    detail: str


def provider_inventory() -> list[ProviderInventoryEntry]:
    discover_providers()
    entries: list[ProviderInventoryEntry] = []
    for capability, registry in (
        ("inference", _INFERENCE_PROVIDERS),
        ("embedding", _EMBEDDING_PROVIDERS),
    ):
        for name in sorted(registry):
            # Identity is class-level: listing providers must never execute a
            # provider constructor.
            entries.append(
                ProviderInventoryEntry(
                    capability=capability,
                    name=name,
                    distribution=_PLUGIN_DISTRIBUTIONS.get(
                        (capability, name), "built-in"
                    ),
                    status="available",
                    detail=implementation_identity_for(registry[name]),
                )
            )
        for name, (distribution, version) in sorted(
            _INCOMPATIBLE_PLUGINS[capability].items()
        ):
            entries.append(
                ProviderInventoryEntry(
                    capability=capability,
                    name=name,
                    distribution=distribution,
                    status="incompatible",
                    detail=(
                        f"targets provider contract v{version}; this release "
                        f"requires v{PROVIDER_CONTRACT_VERSION}"
                    ),
                )
            )
    return entries


def _register[ProviderType: BaseProvider](
    registry: dict[str, type[ProviderType]],
    cls: type[ProviderType],
    *,
    capability: str,
) -> None:
    name = cls.name()
    existing = registry.get(name)
    if existing is not None and existing is not cls:
        raise ProviderRegistrationError(
            f"{capability.capitalize()} provider '{name}' is already registered "
            f"by {existing.__module__}.{existing.__qualname__}"
        )
    registry[name] = cls


def _validate_provider_class(
    cls: type[BaseProvider],
    *,
    capability: str,
) -> None:
    expected_type = (
        InferenceProvider if capability == "inference" else EmbeddingProvider
    )
    if not isinstance(cls, type) or not issubclass(cls, expected_type):
        raise ProviderRegistrationError(
            f"{capability} provider must subclass {expected_type.__name__}"
        )
    if inspect.isabstract(cls):
        raise ProviderRegistrationError(
            f"{capability} provider must implement its required methods"
        )
    try:
        inspect.signature(cls).bind()
    except (TypeError, ValueError):
        raise ProviderRegistrationError(
            f"{capability} provider constructor must not require arguments"
        ) from None
    name = getattr(cls, "provider_name", None)
    if not isinstance(name, str) or not _PROVIDER_NAME.fullmatch(name):
        raise ProviderRegistrationError(
            f"{capability} provider name must match {_PROVIDER_NAME.pattern}"
        )
    implementation_version = getattr(cls, "implementation_version", None)
    if not isinstance(implementation_version, str) or not (
        _IMPLEMENTATION_VERSION.fullmatch(implementation_version)
    ):
        raise ProviderRegistrationError(
            f"{capability} provider implementation_version must match "
            f"{_IMPLEMENTATION_VERSION.pattern}"
        )
    if not isinstance(cls.requires_credentials, bool):
        raise ProviderRegistrationError(
            f"{capability} provider requires_credentials must be boolean"
        )
    if not isinstance(cls.accepts_api_key_env, bool):
        raise ProviderRegistrationError(
            f"{capability} provider accepts_api_key_env must be boolean"
        )
    if cls.requires_credentials and not cls.accepts_api_key_env:
        raise ProviderRegistrationError(
            f"{capability} provider requiring credentials must accept api_key_env"
        )
    env_var = cls.default_credential_env
    if env_var is not None and (
        not isinstance(env_var, str) or not _ENV_VAR_NAME.fullmatch(env_var)
    ):
        raise ProviderRegistrationError(
            f"{capability} provider default_credential_env is invalid"
        )
    packages = cls.implementation_packages
    if (
        not isinstance(packages, tuple)
        or len(packages) != len(set(packages))
        or any(
            not isinstance(package, str)
            or not _DISTRIBUTION_NAME.fullmatch(package)
            for package in packages
        )
    ):
        raise ProviderRegistrationError(
            f"{capability} provider implementation_packages must be unique "
            "distribution names"
        )
    options_model = cls.profile_options_model()
    if options_model is not None:
        if not (
            isinstance(options_model, type) and issubclass(options_model, BaseModel)
        ):
            raise ProviderRegistrationError(
                f"{capability} provider profile_options_model() must return a "
                "Pydantic model class or None"
            )
        validate_profile_options_model(name, options_model)


def _validate_inference_metadata(cls: type[InferenceProvider]) -> None:
    if cls.default_model is not None and (
        not isinstance(cls.default_model, str) or not cls.default_model
    ):
        raise ProviderRegistrationError(
            "inference provider default_model must be a non-empty string"
        )
    if not isinstance(cls.supports_native_batch, bool):
        raise ProviderRegistrationError(
            "inference provider supports_native_batch must be boolean"
        )
    if not isinstance(cls.supports_custom_base_url, bool):
        raise ProviderRegistrationError(
            "inference provider supports_custom_base_url must be boolean"
        )
    if not isinstance(cls.requires_base_url, bool):
        raise ProviderRegistrationError(
            "inference provider requires_base_url must be boolean"
        )
    if cls.requires_base_url and not cls.supports_custom_base_url:
        raise ProviderRegistrationError(
            "inference provider requiring base_url must support custom base URLs"
        )
    if cls.default_base_url is not None:
        if not cls.supports_custom_base_url:
            raise ProviderRegistrationError(
                "inference provider with default_base_url must support custom base URLs"
            )
        try:
            OpenAICompatibleBaseUrl(cls.default_base_url)
        except EndpointUrlError as error:
            raise ProviderRegistrationError(
                "inference provider default_base_url is invalid"
            ) from error
    if cls.supports_native_batch:
        unimplemented = [
            name
            for name in (
                "submit_batch",
                "poll_batch",
                "fetch_batch_results",
                "cancel_batch",
            )
            if getattr(cls, name) is getattr(InferenceProvider, name)
        ]
        if unimplemented:
            raise ProviderRegistrationError(
                "inference provider advertising native batch support must "
                f"override {', '.join(unimplemented)}"
            )
    limit = cls.max_batch_requests
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
    ):
        raise ProviderRegistrationError(
            "inference provider max_batch_requests must be a positive integer"
        )
    multiplier = cls.batch_cost_multiplier
    if (
        isinstance(multiplier, bool)
        or not isinstance(multiplier, (int, float))
        or not math.isfinite(multiplier)
        or multiplier <= 0
    ):
        raise ProviderRegistrationError(
            "inference provider batch_cost_multiplier must be positive and finite"
        )


def _instantiate[ProviderType: BaseProvider](
    cls: type[ProviderType],
    *,
    name: str,
    capability: str,
    profile_options: Mapping[str, Any] | None = None,
) -> ProviderType:
    parsed: BaseModel | None = parse_profile_options(cls, profile_options)
    failure: ProviderConfigurationError | None = None
    try:
        if parsed is None:
            return cls()
        return cls(profile_options=parsed)
    except Exception as error:
        failure = ProviderConfigurationError(
            f"{capability} provider '{name}' could not be initialized "
            f"[{type(error).__name__}]"
        )
    raise failure


def _is_provider_name(value: object) -> bool:
    return isinstance(value, str) and bool(_PROVIDER_NAME.fullmatch(value))
