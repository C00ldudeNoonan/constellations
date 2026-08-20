"""Discovery + resolution of profiles.yml.

Lookup order for the profiles file:
  1. The directory passed via `--profiles-dir` (CLI flag).
  2. The directory named by the `STEL_PROFILES_DIR` env var.
  3. `<project_dir>/profiles.yml` (project-local; stel addition for portability).
  4. `~/.stel/profiles.yml` (dbt-style user-global location).

First hit wins. A project without a `profile:` uses an implicit local DuckDB
target instead (see `_implicit_local_profile`).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .adapters import (
    AdapterConfigError,
    AdapterError,
    parse_warehouse_config,
    prepare_warehouse_profile_input,
)
from .config.profile import (
    DEFAULT_LLM_PROVIDER,
    EmbeddingProfileConfig,
    LLMConfig,
    ProfileConfig,
    WarehouseConfig,
)
from .config.project import ProjectConfig
from .config.source import SourceConfig
from .config.yaml_diagnostics import (
    YamlDocument,
    format_yaml_parse_error,
    parse_yaml_document,
)
from .credentials import CredentialReference
from .env import PROFILES_DIR_ENV, read_env
from .providers import (
    ProviderConfigurationError,
    ProviderNotFoundError,
    ProviderRegistrationError,
    discover_providers,
    get_embedding_provider,
    get_inference_provider,
    parse_profile_options,
    resolve_provider_model,
)
from .retrieval import (
    RetrievalConfigError,
    RetrievalStoreConfig,
    absolutize_store_config,
    parse_store_config,
)

PROFILES_FILENAME = "profiles.yml"


class ProfileError(Exception):
    pass


@dataclass(frozen=True)
class ResolvedRetrievalConfig:
    default: str
    allow_public_indexes: bool
    stores: dict[str, RetrievalStoreConfig]


@dataclass(frozen=True)
class ResolvedProfile:
    """The single source of truth for warehouse + LLM config during a run."""

    profile_name: str
    target_name: str
    warehouse: WarehouseConfig
    llm: LLMConfig | None
    source_paths: dict[str, str]
    profiles_path: Path | None  # None when using inline-legacy fallback
    retrieval: ResolvedRetrievalConfig | None = None
    embedding: EmbeddingProfileConfig | None = None


@dataclass(frozen=True)
class ResolvedEmbeddingOptions:
    provider_options: dict[str, Any] = field(repr=False)
    api_key_env: CredentialReference | None
    timeout_seconds: float


def resolve_profile(
    project: ProjectConfig,
    project_dir: Path,
    *,
    target: str | None = None,
    profiles_dir: Path | None = None,
) -> ResolvedProfile:
    """Resolve the active profile + target for this invocation.

    A project without a `profile:` uses an implicit local DuckDB target from
    its inline `duckdb:` block. Raises `ProfileError` on any structured problem.
    """
    if not project.profile:
        return _implicit_local_profile(project)

    profiles_path = _discover_profiles_file(project_dir, profiles_dir)
    if profiles_path is None:
        raise ProfileError(
            f"Project '{project.name}' references profile '{project.profile}' "
            f"but no profiles.yml was found. Looked in: "
            f"--profiles-dir, ${PROFILES_DIR_ENV}, {project_dir}/profiles.yml, "
            f"~/.stel/profiles.yml."
        )

    profiles, profiles_document = _load_profiles_file(profiles_path)
    if project.profile not in profiles:
        raise ProfileError(
            f"Profile '{project.profile}' not in {profiles_path}. "
            f"Available: {sorted(profiles)}"
        )

    profile = profiles[project.profile]
    target_name = target or profile.target
    if target_name not in profile.outputs:
        raise ProfileError(
            f"Target '{target_name}' not in profile '{project.profile}' "
            f"({profiles_path}). Available: {sorted(profile.outputs)}"
        )

    selected = profile.outputs[target_name]
    warehouse_error: ProfileError | None = None
    try:
        warehouse = parse_warehouse_config(selected.warehouse)
    except AdapterConfigError as error:
        prefix = (
            project.profile,
            "outputs",
            target_name,
            "warehouse",
        )
        diagnostics = profiles_document.format_validation_details(
            profiles_path,
            error.validation_details,
            prefix=prefix,
        )
        warehouse_type = selected.warehouse.get("type", "duckdb")
        warehouse_error = ProfileError(
            f"Invalid {warehouse_type} warehouse YAML at {profiles_path}:\n"
            f"{diagnostics}"
        )
    except AdapterError as error:
        warehouse_error = ProfileError(
            f"{profiles_path}: profile '{project.profile}' target '{target_name}': "
            f"{error}"
        )
    if warehouse_error is not None:
        del selected, profile
        profiles = {}
        raise warehouse_error from None
    # Third-party providers load exactly once, here, before any source,
    # credential, or provider I/O; discovery failures are profile errors.
    try:
        discover_providers()
    except ProviderRegistrationError as e:
        raise ProfileError(f"Provider plugin discovery failed: {e}") from e
    llm = _absolutize_llm(selected.llm, project_dir)
    if llm is not None:
        try:
            inference_provider = get_inference_provider(llm.provider)
            # Validate operator-supplied provider_options against the
            # selected provider's published model; the raw mapping stays on
            # the resolved profile and is re-parsed at the provider boundary.
            parse_profile_options(type(inference_provider), llm.provider_options)
            # Reject an api_key_env a provider cannot accept (e.g. Vertex, which
            # is ADC-only) at preflight, matching the embedding path below.
            inference_provider.validate_credential_reference(llm.api_key_env)
            llm = llm.model_copy(
                update={
                    "model": resolve_provider_model(inference_provider, llm.model),
                    "base_url": inference_provider.resolve_base_url(llm.base_url),
                }
            )
        except (ProviderNotFoundError, ProviderConfigurationError) as e:
            raise ProfileError(
                f"{profiles_path}: profile '{project.profile}' target "
                f"'{target_name}' selects {e}"
            ) from e
    embedding = selected.embedding
    if embedding is not None:
        try:
            embedding_provider = get_embedding_provider(
                embedding.provider,
                profile_options=embedding.provider_options,
            )
            embedding_provider.validate_credential_reference(embedding.api_key_env)
        except (ProviderNotFoundError, ProviderConfigurationError) as e:
            raise ProfileError(
                f"{profiles_path}: profile '{project.profile}' target "
                f"'{target_name}' selects {e}"
            ) from e
    retrieval = _resolve_retrieval(selected.retrieval, project_dir, profiles_path)
    warehouse = warehouse.absolutize(project_dir)
    warehouse.bind_target_name(target_name)
    return ResolvedProfile(
        profile_name=project.profile,
        target_name=target_name,
        warehouse=warehouse,
        llm=llm,
        embedding=embedding,
        retrieval=retrieval,
        source_paths=selected.source_paths,
        profiles_path=profiles_path,
    )


def _absolutize_llm(llm: LLMConfig | None, project_dir: Path) -> LLMConfig | None:
    if llm is None or llm.cache_path is None:
        return llm
    return llm.model_copy(
        update={"cache_path": (project_dir / llm.cache_path).resolve()}
    )


def _implicit_local_profile(project: ProjectConfig) -> ResolvedProfile:
    """Zero-config local target: a project with no `profile:` runs against the
    DuckDB database named by its inline `duckdb:` block. This is a supported
    convenience for local and test projects; declare a `profile:` + profiles.yml
    for warehouse targets, credentials, retrieval, or LLM configuration."""
    raw: dict[str, Any] = {"type": "duckdb", "path": project.duckdb.path}
    # Forward the inline block's *defaultedness*, not just its value: passing
    # `schema` unconditionally would make every inline project look like it
    # had named its schema, and the #313 legacy-schema guard would never fire
    # for the zero-config path that needs it most.
    if "schema_name" in project.duckdb.model_fields_set:
        raw["schema"] = project.duckdb.schema_name
    warehouse = parse_warehouse_config(raw)
    warehouse.bind_target_name("<inline>")
    return ResolvedProfile(
        profile_name="<inline>",
        target_name="<inline>",
        warehouse=warehouse,
        llm=None,
        embedding=None,
        retrieval=None,
        source_paths={},
        profiles_path=None,
    )


def resolve_embedding_options(
    provider_name: str,
    resolved: ResolvedProfile | None,
) -> ResolvedEmbeddingOptions:
    """Bind a model's embedding provider to operator-owned target settings."""
    profile = resolved.embedding if resolved is not None else None
    if profile is None:
        try:
            get_embedding_provider(provider_name)
        except (ProviderNotFoundError, ProviderConfigurationError) as error:
            raise ProfileError(str(error)) from error
        return ResolvedEmbeddingOptions({}, None, 60.0)
    if profile.provider != provider_name:
        raise ProfileError(
            f"Embed model provider '{provider_name}' cannot override profile "
            f"embedding provider '{profile.provider}'; select the provider under "
            "`embedding:` in profiles.yml"
        )
    try:
        provider = get_embedding_provider(
            provider_name,
            profile_options=profile.provider_options,
        )
        provider.validate_credential_reference(profile.api_key_env)
    except (ProviderNotFoundError, ProviderConfigurationError) as error:
        raise ProfileError(str(error)) from error
    return ResolvedEmbeddingOptions(
        dict(profile.provider_options),
        profile.api_key_env,
        profile.timeout_seconds,
    )


def _resolve_retrieval(
    retrieval: Any,
    project_dir: Path,
    profiles_path: Path,
) -> ResolvedRetrievalConfig | None:
    if retrieval is None:
        return None
    if retrieval.default not in retrieval.stores:
        raise ProfileError(
            f"{profiles_path}: retrieval.default '{retrieval.default}' is not in "
            f"retrieval.stores. Available: {sorted(retrieval.stores)}"
        )
    stores: dict[str, RetrievalStoreConfig] = {}
    for alias, raw in retrieval.stores.items():
        if not alias:
            raise ProfileError(f"{profiles_path}: retrieval store aliases must not be empty")
        try:
            parsed = parse_store_config(raw)
            stores[alias] = absolutize_store_config(parsed, project_dir)
        except RetrievalConfigError as error:
            raise ProfileError(
                f"{profiles_path}: retrieval store '{alias}': {error}"
            ) from None
    return ResolvedRetrievalConfig(
        default=retrieval.default,
        allow_public_indexes=retrieval.allow_public_indexes,
        stores=stores,
    )


def apply_source_path_overrides(
    sources: list[SourceConfig], resolved: ResolvedProfile
) -> list[SourceConfig]:
    """Apply target-specific source roots from profiles.yml.

    Project YAML remains the reviewed declaration of sources; profiles.yml can
    override only their roots per target so dev/staging/prod can point at
    different buckets or local directories without editing the project.
    """
    if not resolved.source_paths:
        return sources

    source_names = {source.name for source in sources}
    unknown = sorted(set(resolved.source_paths) - source_names)
    if unknown:
        raise ProfileError(
            f"Profile target '{resolved.target_name}' defines source_paths for "
            f"unknown source(s): {', '.join(unknown)}. Available sources: "
            f"{', '.join(sorted(source_names)) or '<none>'}."
        )

    out: list[SourceConfig] = []
    for source in sources:
        override = resolved.source_paths.get(source.name)
        if override is None:
            out.append(source)
            continue
        out.append(
            source.model_copy(
                update={"path": override, "external": source.external or _is_local_path(override)}
            )
        )
    return out


def _is_local_path(path: str) -> bool:
    return "://" not in path


def _discover_profiles_file(
    project_dir: Path, profiles_dir: Path | None
) -> Path | None:
    trusted_candidates: list[Path] = []
    if profiles_dir is not None:
        trusted_candidates.append(profiles_dir / PROFILES_FILENAME)
    env_dir = read_env(PROFILES_DIR_ENV)
    if env_dir:
        trusted_candidates.append(Path(env_dir) / PROFILES_FILENAME)

    for path in trusted_candidates:
        if path.exists():
            return path

    project_local = project_dir / PROFILES_FILENAME
    if project_local.is_symlink():
        raise ProfileError(
            f"Refusing to load symlinked project-local profiles file "
            f"{project_local}. Use a regular file in the project, or pass "
            "--profiles-dir for an operator-trusted profile."
        )
    if project_local.exists():
        if not project_local.is_file():
            raise ProfileError(
                f"Project-local profiles path is not a regular file: {project_local}"
            )
        return project_local

    global_profile = Path.home() / ".stel" / PROFILES_FILENAME
    if global_profile.exists():
        return global_profile
    return None


# `{{ env_var('NAME') }}` or `{{ env_var('NAME', 'default') }}` — the one piece
# of dbt's Jinja grammar profiles need for routing values. Adapter credential
# hooks replace protected references before this interpolation runs.
_ENV_VAR_RE = re.compile(
    r"\{\{\s*env_var\(\s*(['\"])(?P<name>[A-Za-z_][A-Za-z0-9_]*)\1"
    r"(?:\s*,\s*(['\"])(?P<default>.*?)\3)?\s*\)\s*\}\}"
)
_PROTECTED_PROFILE_REFERENCE_FIELDS = frozenset(
    {
        "client_secret",
        "keyfile",
        "keyfile_json",
        "refresh_token",
        "token",
        "token_uri",
    }
)


def _interpolate_env_vars(value: Any, path: Path) -> Any:
    if isinstance(value, str):

        def _sub(match: re.Match[str]) -> str:
            name = match.group("name")
            env_value = os.environ.get(name)
            if env_value is not None:
                return env_value
            default = match.group("default")
            if default is not None:
                return default
            raise ProfileError(
                f"{path}: env_var('{name}') is not set and has no default"
            )

        return _ENV_VAR_RE.sub(_sub, value)
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for key, item in value.items():
            if (
                key == "api_key_env"
                and isinstance(item, str)
                and _ENV_VAR_RE.search(item)
            ):
                raise ProfileError(
                    f"{path}: llm.api_key_env must name an environment variable "
                    "directly; do not interpolate its secret value with env_var()."
                )
            out[key] = _interpolate_env_vars(item, path)
        return out
    if isinstance(value, list):
        return [_interpolate_env_vars(v, path) for v in value]
    return value


def _prepare_profile_warehouses(data: dict[Any, Any], path: Path) -> None:
    for profile in data.values():
        if not isinstance(profile, dict):
            continue
        outputs = profile.get("outputs")
        if not isinstance(outputs, dict):
            continue
        for target in outputs.values():
            if not isinstance(target, dict):
                continue
            warehouse = target.get("warehouse")
            if not isinstance(warehouse, dict):
                continue
            prepared = dict(warehouse)
            warehouse_type = prepared.get("type")
            if isinstance(warehouse_type, str):
                prepared["type"] = _interpolate_env_vars(warehouse_type, path)
            try:
                protected = prepare_warehouse_profile_input(prepared)
            except AdapterError as error:
                raise ProfileError(f"{path}: {error}") from None
            for key in _PROTECTED_PROFILE_REFERENCE_FIELDS:
                item = protected.get(key)
                if isinstance(item, str) and _ENV_VAR_RE.search(item):
                    raise ProfileError(
                        f"{path}: credential field `{key}` was not protected by "
                        "a registered adapter; refusing generic env_var() "
                        "interpolation"
                    )
            target["warehouse"] = protected


def _load_profiles_file(
    path: Path,
) -> tuple[dict[str, ProfileConfig], YamlDocument]:
    document: YamlDocument | None = None
    load_error: ProfileError | None = None
    try:
        with path.open() as f:
            document = parse_yaml_document(f.read())
    except yaml.YAMLError as e:
        load_error = ProfileError(format_yaml_parse_error(path, e))
    except OSError as e:
        load_error = ProfileError(f"Could not read profiles file {path}: {e}")
    if load_error is not None:
        raise load_error
    assert document is not None
    data: Any = document.data if document.data is not None else {}
    document = document.without_data()
    if not isinstance(data, dict):
        diagnostic = document.format_message(
            path,
            (),
            "top-level must be a mapping of profile names",
        )
        data = None
        load_error = ProfileError(diagnostic)
        raise load_error
    profile_input_error: ProfileError | None = None
    try:
        _prepare_profile_warehouses(data, path)
        data = _interpolate_env_vars(data, path)
    except ProfileError as error:
        profile_input_error = ProfileError(str(error))
    if profile_input_error is not None:
        data = {}
        raise profile_input_error

    out: dict[str, ProfileConfig] = {}
    for name, body in data.items():
        validation_failure: ProfileError | None = None
        try:
            out[name] = ProfileConfig.model_validate(body)
        except ValidationError as e:
            diagnostics = document.format_validation_errors(
                path,
                e,
                prefix=(name,),
            )
            validation_failure = ProfileError(
                f"Invalid profile YAML at {path}:\n{diagnostics}"
            )
        if validation_failure is not None:
            body = {}
            data = {}
            out = {}
            raise validation_failure
    return out, document


def resolve_llm_options(
    options: dict[str, Any], resolved: ResolvedProfile
) -> dict[str, Any]:
    """Merge profile.llm defaults into model-level extraction options.

    Model-level execution options win; the credential variable remains
    operator-owned profile configuration.
    """
    if "api_key_env" in options:
        credential_ownership_error = ProfileError(
            "llm option 'api_key_env' is operator-owned configuration; set it "
            "under `llm:` in profiles.yml, not in model extraction options"
        )
        options = {}
        raise credential_ownership_error
    if "base_url" in options:
        endpoint_ownership_error = ProfileError(
            "llm option 'base_url' is operator-owned configuration; set it "
            "under `llm:` in profiles.yml, not in model extraction options"
        )
        options = {}
        raise endpoint_ownership_error
    if "provider_options" in options:
        provider_options_ownership_error = ProfileError(
            "llm option 'provider_options' is operator-owned configuration; "
            "set it under `llm:` in profiles.yml, not in model extraction options"
        )
        options = {}
        raise provider_options_ownership_error
    merged = dict(options)
    profile_provider = (
        resolved.llm.provider if resolved.llm is not None else DEFAULT_LLM_PROVIDER
    )
    if "provider" in merged and merged["provider"] != profile_provider:
        raise ProfileError(
            "llm option 'provider' cannot override the profile provider because "
            "credentials are operator-owned; select the provider under `llm:` "
            "in profiles.yml"
        )
    provider_name = str(merged.get("provider") or profile_provider)
    try:
        provider = get_inference_provider(provider_name)
    except (ProviderNotFoundError, ProviderConfigurationError) as e:
        raise ProfileError(str(e)) from e
    merged["provider"] = provider_name
    requested_model = merged.get("model")
    if requested_model is None and resolved.llm is not None:
        requested_model = resolved.llm.model
    try:
        merged["model"] = resolve_provider_model(provider, requested_model)
    except ProviderConfigurationError as e:
        raise ProfileError(str(e)) from e
    profile_base_url = (
        resolved.llm.base_url if resolved.llm is not None else None
    )
    try:
        base_url = provider.resolve_base_url(profile_base_url)
    except ProviderConfigurationError as e:
        raise ProfileError(str(e)) from e
    if base_url is not None:
        merged["base_url"] = base_url
    profile_provider_options = (
        resolved.llm.provider_options if resolved.llm is not None else {}
    )
    if profile_provider_options:
        try:
            parse_profile_options(type(provider), profile_provider_options)
        except ProviderConfigurationError as e:
            raise ProfileError(str(e)) from e
        merged["provider_options"] = dict(profile_provider_options)
    if merged.get("batch") and not provider.supports_native_batch:
        raise ProfileError(
            f"Inference provider '{provider_name}' does not support native "
            "batch execution"
        )
    api_key_env = (
        resolved.llm.api_key_env
        if resolved.llm is not None and resolved.llm.api_key_env is not None
        else provider.default_credential_env
    )
    if api_key_env is not None:
        merged.setdefault("api_key_env", api_key_env)
    if resolved.llm is None:
        return merged

    if resolved.llm.cache_path is not None:
        merged.setdefault("cache_path", str(resolved.llm.cache_path))
    if "timeout_seconds" in resolved.llm.model_fields_set:
        merged.setdefault("timeout_seconds", resolved.llm.timeout_seconds)
    if resolved.llm.system_prompt is not None:
        merged.setdefault("system_prompt", resolved.llm.system_prompt)
    return merged
