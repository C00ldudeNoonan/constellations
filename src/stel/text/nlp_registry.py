"""The NLP provider registry: registration contract and entry-point discovery.

A provider (spaCy today, anything packaged separately tomorrow) registers one
`NLPProviderRegistration`. Transform execution only ever asks the registry for
a provider by name, so adding a vendor never edits `_nlp.py` or the runner
(issue #222, ADR-0064).

Registration is declarative on purpose. A provider states which capabilities it
can emit and, when it knows, which languages it supports, so a project that asks
for something the provider cannot do fails during `stel compile` rather than
after documents have been read. Capabilities are not erased into a lowest common
denominator: a transform names the ones it needs and the registry says which
providers have them.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import EntryPoint, distributions
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .nlp import NLPProvider

# The entry-point group suffix is the contract major version. A plugin picks the
# group it was built against and never loads under another.
NLP_PROVIDER_CONTRACT_VERSION = 1
NLP_PROVIDER_GROUP_BASE = "stel.nlp_providers"

NLPCapability = Literal["tokens", "lemmas", "pos", "entities", "sentences", "confidence"]
NLP_CAPABILITIES: frozenset[str] = frozenset(
    ("tokens", "lemmas", "pos", "entities", "sentences", "confidence")
)

_PROVIDER_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_LANGUAGE_CODE = re.compile(r"^[a-z]{2,3}(-[a-z0-9]{2,8})*$")
_GROUP_VERSION = re.compile(r"\.v(\d+)$")


class NLPProviderRegistryError(ValueError):
    """A provider is unknown, invalid, or conflicts with another registration.

    A `ValueError` so that option validation reports it as a configuration
    error rather than letting it escape a Pydantic validator."""


def normalize_language_code(code: str) -> str:
    """Lowercase a language tag and use `-` as the separator (`zh_CN` -> `zh-cn`)."""
    return code.strip().lower().replace("_", "-")


def primary_language(code: str) -> str:
    """The primary subtag of a normalized tag (`zh-cn` -> `zh`)."""
    return code.split("-", maxsplit=1)[0]


def is_language_code(code: str) -> bool:
    return bool(_LANGUAGE_CODE.fullmatch(code))


@dataclass(frozen=True)
class NLPModelSpec:
    """What a provider loader is asked to build: one model for one language."""

    model: str
    language: str
    disable: tuple[str, ...]


@dataclass(frozen=True)
class NLPProviderRegistration:
    """One provider's declaration.

    `languages` is `None` when the provider cannot know the languages ahead of
    loading a model (spaCy: the installed model decides), in which case the
    language is checked at load time. A provider that does know lists them and
    gets a preflight failure for an unsupported language. `loader` must import
    its optional dependency lazily and raise `NLPError` with an install command
    when the extra or model is absent; it must not download anything."""

    name: str
    loader: Callable[[NLPModelSpec], NLPProvider]
    capabilities: frozenset[str]
    languages: frozenset[str] | None

    def supports_language(self, language: str) -> bool:
        if self.languages is None:
            return True
        normalized = normalize_language_code(language)
        return (
            normalized in self.languages
            or primary_language(normalized) in self.languages
        )


_REGISTRY: dict[str, NLPProviderRegistration] = {}
# provider name -> distribution, for diagnostics only
_PLUGIN_DISTRIBUTIONS: dict[str, str] = {}
# provider name -> (distribution, advertised contract version)
_INCOMPATIBLE_PLUGINS: dict[str, tuple[str, str]] = {}
_DISCOVERY_COMPLETE = False


def entry_point_group() -> str:
    return f"{NLP_PROVIDER_GROUP_BASE}.v{NLP_PROVIDER_CONTRACT_VERSION}"


def register_nlp_provider(registration: NLPProviderRegistration) -> None:
    """Add a provider. Validated here, at registration, not at first use."""
    if not isinstance(registration, NLPProviderRegistration):
        raise NLPProviderRegistryError(
            "NLP provider registration must be an NLPProviderRegistration"
        )
    name = registration.name
    if not isinstance(name, str) or not _PROVIDER_NAME.fullmatch(name):
        raise NLPProviderRegistryError(
            f"NLP provider name must match {_PROVIDER_NAME.pattern}"
        )
    if not callable(registration.loader):
        raise NLPProviderRegistryError(
            f"NLP provider '{name}' loader must be callable"
        )
    unknown = sorted(set(registration.capabilities) - NLP_CAPABILITIES)
    if unknown:
        raise NLPProviderRegistryError(
            f"NLP provider '{name}' declares unknown capabilities {unknown}; "
            f"choose from {sorted(NLP_CAPABILITIES)}"
        )
    if not registration.capabilities:
        raise NLPProviderRegistryError(
            f"NLP provider '{name}' must declare at least one capability"
        )
    if registration.languages is not None:
        bad = sorted(
            code
            for code in registration.languages
            if code != normalize_language_code(code) or not is_language_code(code)
        )
        if bad or not registration.languages:
            raise NLPProviderRegistryError(
                f"NLP provider '{name}' languages must be a non-empty set of "
                f"normalized codes such as 'en' or 'zh-cn'; got {bad}"
            )
    existing = _REGISTRY.get(name)
    if existing is not None and existing != registration:
        raise NLPProviderRegistryError(
            f"NLP provider '{name}' is already registered"
        )
    _REGISTRY[name] = registration


def discover_nlp_providers(*, force: bool = False) -> None:
    """Load separately packaged providers from the versioned entry-point group.

    Runs once, deterministically (entry-point name, then distribution). Duplicate
    names, a plugin claiming a built-in name, a name mismatch, and a load failure
    all raise rather than being skipped, and a plugin that only advertises a
    different contract version is remembered so a lookup can say so instead of
    "not registered"."""
    global _DISCOVERY_COMPLETE
    if _DISCOVERY_COMPLETE and not force:
        return
    current: list[EntryPoint] = []
    other_versions: list[tuple[EntryPoint, str]] = []
    # Scan distributions directly: the stdlib entry_points() view collapses
    # same-named entry points across distributions, which would let one plugin
    # silently shadow another instead of failing the duplicate.
    for dist in distributions():
        for item in dist.entry_points:
            if item.group == entry_point_group():
                current.append(item)
            elif item.group.startswith(NLP_PROVIDER_GROUP_BASE + "."):
                match = _GROUP_VERSION.search(item.group)
                other_versions.append((item, match.group(1) if match else "unknown"))
    for item in sorted(current, key=lambda ep: (ep.name, _distribution(ep))):
        _load_plugin(item)
    for item, version in sorted(
        other_versions, key=lambda pair: (pair[0].name, _distribution(pair[0]))
    ):
        _INCOMPATIBLE_PLUGINS.setdefault(item.name, (_distribution(item), version))
    _DISCOVERY_COMPLETE = True


def get_nlp_registration(name: str) -> NLPProviderRegistration:
    discover_nlp_providers()
    registration = _REGISTRY.get(name)
    if registration is not None:
        return registration
    incompatible = _INCOMPATIBLE_PLUGINS.get(name)
    if incompatible is not None:
        distribution, version = incompatible
        raise NLPProviderRegistryError(
            f"NLP provider '{name}' from distribution '{distribution}' targets "
            f"contract v{version}, but this stel release requires "
            f"v{NLP_PROVIDER_CONTRACT_VERSION}; install a compatible plugin release"
        )
    raise NLPProviderRegistryError(
        f"NLP provider '{name}' is not registered. Available: {list_nlp_providers()}"
    )


def list_nlp_providers() -> list[str]:
    discover_nlp_providers()
    return sorted(_REGISTRY)


def _distribution(item: EntryPoint) -> str:
    name = getattr(getattr(item, "dist", None), "name", None)
    return name if isinstance(name, str) and name else "<unknown distribution>"


def _load_plugin(item: EntryPoint) -> None:
    distribution = _distribution(item)
    try:
        loaded = item.load()
    except Exception as error:
        # Only the exception type: its text can carry paths or secrets.
        raise NLPProviderRegistryError(
            f"NLP provider entry point '{item.name}' from distribution "
            f"'{distribution}' failed to load [{type(error).__name__}]"
        ) from None
    if not isinstance(loaded, NLPProviderRegistration) or loaded.name != item.name:
        raise NLPProviderRegistryError(
            f"NLP provider entry point '{item.name}' from distribution "
            f"'{distribution}' must load an NLPProviderRegistration whose name "
            "is the entry-point name"
        )
    if item.name in _REGISTRY and _REGISTRY[item.name] != loaded:
        other = _PLUGIN_DISTRIBUTIONS.get(item.name, "built-in")
        raise NLPProviderRegistryError(
            f"NLP provider '{item.name}' is claimed by both '{other}' and "
            f"'{distribution}'; provider names must be unique"
        )
    try:
        register_nlp_provider(loaded)
    except NLPProviderRegistryError as error:
        raise NLPProviderRegistryError(f"distribution '{distribution}': {error}") from None
    _PLUGIN_DISTRIBUTIONS[item.name] = distribution
