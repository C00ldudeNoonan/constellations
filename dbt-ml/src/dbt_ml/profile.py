"""Discovery + resolution of profiles.yml.

Lookup order for the profiles file:
  1. The directory passed via `--profiles-dir` (CLI flag).
  2. The directory named by the `DBT_ML_PROFILES_DIR` env var
     (`DOCBT_PROFILES_DIR` is honored as a deprecated alias).
  3. `<project_dir>/profiles.yml` (project-local; dbt-ml addition for portability).
  4. `~/.dbt_ml/profiles.yml` (dbt-style user-global location).

First hit wins.
"""
from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .adapters import AdapterError, parse_warehouse_config
from .config.profile import LLMConfig, ProfileConfig, WarehouseConfig
from .config.project import ProjectConfig
from .config.source import SourceConfig

PROFILES_FILENAME = "profiles.yml"
PROFILES_DIR_ENV = "DBT_ML_PROFILES_DIR"
LEGACY_PROFILES_DIR_ENV = "DOCBT_PROFILES_DIR"


class ProfileError(Exception):
    pass


@dataclass(frozen=True)
class ResolvedProfile:
    """The single source of truth for warehouse + LLM config during a run."""

    profile_name: str
    target_name: str
    warehouse: WarehouseConfig
    llm: LLMConfig | None
    source_paths: dict[str, str]
    profiles_path: Path | None  # None when using inline-legacy fallback


def resolve_profile(
    project: ProjectConfig,
    project_dir: Path,
    *,
    target: str | None = None,
    profiles_dir: Path | None = None,
) -> ResolvedProfile:
    """Resolve the active profile + target for this invocation.

    Falls back to the legacy inline `duckdb:` block when no `profile:` is set
    in the project file. Raises `ProfileError` on any structured problem.
    """
    if not project.profile:
        return _legacy_resolved(project)

    profiles_path = _discover_profiles_file(project_dir, profiles_dir)
    if profiles_path is None:
        raise ProfileError(
            f"Project '{project.name}' references profile '{project.profile}' "
            f"but no profiles.yml was found. Looked in: "
            f"--profiles-dir, ${PROFILES_DIR_ENV}, {project_dir}/profiles.yml, "
            f"~/.dbt_ml/profiles.yml."
        )

    profiles = _load_profiles_file(profiles_path)
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
    try:
        warehouse = parse_warehouse_config(selected.warehouse)
    except AdapterError as e:
        raise ProfileError(
            f"{profiles_path}: profile '{project.profile}' target '{target_name}': {e}"
        ) from e
    return ResolvedProfile(
        profile_name=project.profile,
        target_name=target_name,
        warehouse=warehouse.absolutize(project_dir),
        llm=_absolutize_llm(selected.llm, project_dir),
        source_paths=selected.source_paths,
        profiles_path=profiles_path,
    )


def _absolutize_llm(llm: LLMConfig | None, project_dir: Path) -> LLMConfig | None:
    if llm is None or llm.cache_path is None:
        return llm
    return llm.model_copy(
        update={"cache_path": (project_dir / llm.cache_path).resolve()}
    )


def _legacy_resolved(project: ProjectConfig) -> ResolvedProfile:
    warnings.warn(
        f"Project '{project.name}' has no `profile:`, so dbt-ml is falling back to "
        "the inline `duckdb:` block. This path is deprecated and will be removed; "
        "declare a `profile:` and a profiles.yml `warehouse:` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    warehouse = parse_warehouse_config(
        {
            "type": "duckdb",
            "path": project.duckdb.path,
            "schema": project.duckdb.schema_name,
        }
    )
    return ResolvedProfile(
        profile_name="<inline>",
        target_name="<inline>",
        warehouse=warehouse,
        llm=None,
        source_paths={},
        profiles_path=None,
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


def _legacy_env_dir() -> str | None:
    value = os.environ.get(LEGACY_PROFILES_DIR_ENV)
    if value:
        warnings.warn(
            f"${LEGACY_PROFILES_DIR_ENV} is deprecated; use ${PROFILES_DIR_ENV}.",
            DeprecationWarning,
            stacklevel=3,
        )
    return value


def _discover_profiles_file(
    project_dir: Path, profiles_dir: Path | None
) -> Path | None:
    candidates: list[Path] = []
    if profiles_dir is not None:
        candidates.append(profiles_dir / PROFILES_FILENAME)
    env_dir = os.environ.get(PROFILES_DIR_ENV) or _legacy_env_dir()
    if env_dir:
        candidates.append(Path(env_dir) / PROFILES_FILENAME)
    candidates.append(project_dir / PROFILES_FILENAME)
    candidates.append(Path.home() / ".dbt_ml" / PROFILES_FILENAME)

    for p in candidates:
        if p.exists():
            return p
    return None


# `{{ env_var('NAME') }}` or `{{ env_var('NAME', 'default') }}` — the one piece
# of dbt's Jinja grammar profiles need, so secrets stay out of the file.
_ENV_VAR_RE = re.compile(
    r"\{\{\s*env_var\(\s*(['\"])(?P<name>[A-Za-z_][A-Za-z0-9_]*)\1"
    r"(?:\s*,\s*(['\"])(?P<default>.*?)\3)?\s*\)\s*\}\}"
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
        return {k: _interpolate_env_vars(v, path) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env_vars(v, path) for v in value]
    return value


def _load_profiles_file(path: Path) -> dict[str, ProfileConfig]:
    with path.open() as f:
        data: Any = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ProfileError(f"{path}: top-level must be a mapping of profile names")
    data = _interpolate_env_vars(data, path)

    out: dict[str, ProfileConfig] = {}
    for name, body in data.items():
        try:
            out[name] = ProfileConfig.model_validate(body)
        except ValidationError as e:
            raise ProfileError(f"{path}: profile '{name}' invalid:\n{e}") from e
    return out


def resolve_llm_options(
    options: dict[str, Any], resolved: ResolvedProfile
) -> dict[str, Any]:
    """Merge profile.llm defaults into model-level extraction options.

    Model-level options always win; profile.llm fills in what's missing.
    """
    if resolved.llm is None:
        return options

    merged = dict(options)
    merged.setdefault("model", resolved.llm.model)
    if resolved.llm.cache_path is not None:
        merged.setdefault("cache_path", str(resolved.llm.cache_path))
    if resolved.llm.system_prompt is not None:
        merged.setdefault("system_prompt", resolved.llm.system_prompt)
    return merged
