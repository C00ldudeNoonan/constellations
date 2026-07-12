from __future__ import annotations

from pathlib import Path

import pytest

from dbt_ml.config import load_project
from dbt_ml.config.source import SourceConfig
from dbt_ml.profile import (
    ProfileError,
    ResolvedProfile,
    apply_source_path_overrides,
    resolve_llm_options,
    resolve_profile,
)


def _write_project(
    tmp_path: Path,
    name: str = "test_proj",
    *,
    profile: str | None = None,
    inline_duckdb: bool = False,
) -> Path:
    lines = [f"name: {name}", 'version: "0.1.0"']
    if profile:
        lines.append(f"profile: {profile}")
    if inline_duckdb:
        lines += [
            "duckdb:",
            "  path: ./inline/db.duckdb",
            "  schema: inline_schema",
        ]
    (tmp_path / "dbt_ml_project.yml").write_text("\n".join(lines) + "\n")
    return tmp_path


def _write_profiles(
    tmp_path: Path,
    name: str = "test_proj",
    *,
    default_target: str = "dev",
    targets: dict[str, dict] | None = None,
) -> Path:
    targets = targets or {
        "dev": {
            "warehouse": {
                "type": "duckdb",
                "path": "./target/dev.duckdb",
                "schema": "dev_schema",
            }
        }
    }
    lines = [f"{name}:", f"  target: {default_target}", "  outputs:"]
    for tname, tcfg in targets.items():
        lines.append(f"    {tname}:")
        wh = tcfg["warehouse"]
        lines += [
            "      warehouse:",
            f"        type: {wh['type']}",
            f"        path: {wh['path']}",
            f"        schema: {wh['schema']}",
        ]
        if "llm" in tcfg:
            llm = tcfg["llm"]
            lines.append("      llm:")
            for k, v in llm.items():
                lines.append(f"        {k}: {v}")
        if "source_paths" in tcfg:
            lines.append("      source_paths:")
            for source_name, source_path in tcfg["source_paths"].items():
                lines.append(f"        {source_name}: {source_path}")
    path = tmp_path / "profiles.yml"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_legacy_fallback_when_no_profile(tmp_path: Path) -> None:
    _write_project(tmp_path, inline_duckdb=True)
    project, _, _ = load_project(tmp_path)
    with pytest.warns(DeprecationWarning, match="no `profile:`"):
        resolved = resolve_profile(project, tmp_path)
    assert resolved.profile_name == "<inline>"
    assert resolved.warehouse.schema_name == "inline_schema"


def test_profile_resolves_warehouse(tmp_path: Path) -> None:
    _write_project(tmp_path, profile="test_proj")
    _write_profiles(tmp_path)
    project, _, _ = load_project(tmp_path)
    resolved = resolve_profile(project, tmp_path)
    assert resolved.profile_name == "test_proj"
    assert resolved.target_name == "dev"
    assert resolved.warehouse.schema_name == "dev_schema"


def test_target_override(tmp_path: Path) -> None:
    _write_project(tmp_path, profile="test_proj")
    _write_profiles(
        tmp_path,
        default_target="dev",
        targets={
            "dev": {
                "warehouse": {"type": "duckdb", "path": "./d.duckdb", "schema": "d"}
            },
            "prod": {
                "warehouse": {"type": "duckdb", "path": "./p.duckdb", "schema": "p"}
            },
        },
    )
    project, _, _ = load_project(tmp_path)
    resolved = resolve_profile(project, tmp_path, target="prod")
    assert resolved.target_name == "prod"
    assert resolved.warehouse.schema_name == "p"


def test_unknown_target_raises(tmp_path: Path) -> None:
    _write_project(tmp_path, profile="test_proj")
    _write_profiles(tmp_path)
    project, _, _ = load_project(tmp_path)
    with pytest.raises(ProfileError, match="Target 'nope'"):
        resolve_profile(project, tmp_path, target="nope")


def test_missing_profiles_file_raises(tmp_path: Path) -> None:
    _write_project(tmp_path, profile="test_proj")
    project, _, _ = load_project(tmp_path)
    with pytest.raises(ProfileError, match=r"no profiles\.yml was found"):
        resolve_profile(project, tmp_path)


def test_unknown_profile_raises(tmp_path: Path) -> None:
    _write_project(tmp_path, profile="not_there")
    _write_profiles(tmp_path, name="something_else")
    project, _, _ = load_project(tmp_path)
    with pytest.raises(ProfileError, match="Profile 'not_there' not in"):
        resolve_profile(project, tmp_path)


def test_profiles_dir_override(tmp_path: Path) -> None:
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    _write_project(project_dir, profile="test_proj")

    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    _write_profiles(other_dir)

    project, _, _ = load_project(project_dir)
    resolved = resolve_profile(project, project_dir, profiles_dir=other_dir)
    assert resolved.warehouse.schema_name == "dev_schema"


def test_env_var_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    _write_project(project_dir, profile="test_proj")

    other_dir = tmp_path / "via_env"
    other_dir.mkdir()
    _write_profiles(other_dir)

    monkeypatch.setenv("DBT_ML_PROFILES_DIR", str(other_dir))
    project, _, _ = load_project(project_dir)
    resolved = resolve_profile(project, project_dir)
    assert resolved.warehouse.schema_name == "dev_schema"


def test_legacy_env_var_still_works_with_deprecation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    _write_project(project_dir, profile="test_proj")

    other_dir = tmp_path / "via_legacy_env"
    other_dir.mkdir()
    _write_profiles(other_dir)

    monkeypatch.delenv("DBT_ML_PROFILES_DIR", raising=False)
    monkeypatch.setenv("DOCBT_PROFILES_DIR", str(other_dir))
    project, _, _ = load_project(project_dir)
    with pytest.warns(DeprecationWarning, match="DOCBT_PROFILES_DIR"):
        resolved = resolve_profile(project, project_dir)
    assert resolved.warehouse.schema_name == "dev_schema"


def test_llm_options_merged_from_profile(tmp_path: Path) -> None:
    _write_project(tmp_path, profile="test_proj")
    _write_profiles(
        tmp_path,
        targets={
            "dev": {
                "warehouse": {"type": "duckdb", "path": "./d.duckdb", "schema": "d"},
                "llm": {
                    "provider": "anthropic",
                    "model": "claude-haiku-4-5",
                    "api_key_env": "DBT_ML_ANTHROPIC_KEY",
                    "cache_path": "./target/cache.duckdb",
                },
            }
        },
    )
    project, _, _ = load_project(tmp_path)
    resolved = resolve_profile(project, tmp_path)
    options = resolve_llm_options({"fields": [{"name": "x"}]}, resolved)
    assert options["model"] == "claude-haiku-4-5"
    assert options["api_key_env"] == "DBT_ML_ANTHROPIC_KEY"
    assert options["cache_path"].endswith("cache.duckdb")
    assert options["fields"] == [{"name": "x"}]


def test_model_option_overrides_profile(tmp_path: Path) -> None:
    _write_project(tmp_path, profile="test_proj")
    _write_profiles(
        tmp_path,
        targets={
            "dev": {
                "warehouse": {"type": "duckdb", "path": "./d.duckdb", "schema": "d"},
                "llm": {"provider": "anthropic", "model": "claude-haiku-4-5"},
            }
        },
    )
    project, _, _ = load_project(tmp_path)
    resolved = resolve_profile(project, tmp_path)
    options = resolve_llm_options(
        {"model": "claude-sonnet-4-6", "fields": []}, resolved
    )
    assert options["model"] == "claude-sonnet-4-6"


def test_model_api_key_env_cannot_override_profile(tmp_path: Path) -> None:
    _write_project(tmp_path, profile="test_proj")
    _write_profiles(
        tmp_path,
        targets={
            "dev": {
                "warehouse": {"type": "duckdb", "path": "./d.duckdb", "schema": "d"},
                "llm": {
                    "provider": "anthropic",
                    "api_key_env": "PROFILE_ANTHROPIC_KEY",
                },
            }
        },
    )
    project, _, _ = load_project(tmp_path)
    resolved = resolve_profile(project, tmp_path)
    with pytest.raises(ProfileError, match="operator-owned"):
        resolve_llm_options(
            {"api_key_env": "MODEL_ANTHROPIC_KEY", "fields": []}, resolved
        )


def test_resolved_profile_is_frozen_dataclass(tmp_path: Path) -> None:
    import dataclasses

    _write_project(tmp_path, profile="test_proj")
    _write_profiles(tmp_path)
    project, _, _ = load_project(tmp_path)
    resolved = resolve_profile(project, tmp_path)
    assert isinstance(resolved, ResolvedProfile)
    with pytest.raises(dataclasses.FrozenInstanceError):
        resolved.target_name = "other"  # type: ignore[misc]


def test_target_source_path_overrides(tmp_path: Path) -> None:
    _write_project(tmp_path, profile="test_proj")
    _write_profiles(
        tmp_path,
        targets={
            "dev": {
                "warehouse": {"type": "duckdb", "path": "./d.duckdb", "schema": "d"},
                "source_paths": {"filings": "gs://bucket-dev/sec_filings"},
            },
            "prod": {
                "warehouse": {"type": "duckdb", "path": "./p.duckdb", "schema": "p"},
                "source_paths": {"filings": "gs://bucket-prod/sec_filings"},
            },
        },
    )
    project, _, _ = load_project(tmp_path)
    source = SourceConfig(name="filings", path="gs://bucket-prod/sec_filings")

    dev = resolve_profile(project, tmp_path, target="dev")
    prod = resolve_profile(project, tmp_path, target="prod")

    assert apply_source_path_overrides([source], dev)[0].path == (
        "gs://bucket-dev/sec_filings"
    )
    assert apply_source_path_overrides([source], prod)[0].path == (
        "gs://bucket-prod/sec_filings"
    )


def test_target_local_source_path_override_is_profile_trusted(tmp_path: Path) -> None:
    _write_project(tmp_path, profile="test_proj")
    outside = tmp_path.parent / "operator_docs"
    _write_profiles(
        tmp_path,
        targets={
            "dev": {
                "warehouse": {"type": "duckdb", "path": "./d.duckdb", "schema": "d"},
                "source_paths": {"filings": outside.as_posix()},
            }
        },
    )
    project, _, _ = load_project(tmp_path)
    source = SourceConfig(name="filings", path="data/prod", external=False)

    resolved = resolve_profile(project, tmp_path)
    overridden = apply_source_path_overrides([source], resolved)[0]

    assert overridden.path == outside.as_posix()
    assert overridden.external is True


def test_unknown_target_source_path_override_raises(tmp_path: Path) -> None:
    _write_project(tmp_path, profile="test_proj")
    _write_profiles(
        tmp_path,
        targets={
            "dev": {
                "warehouse": {"type": "duckdb", "path": "./d.duckdb", "schema": "d"},
                "source_paths": {"typo": "data/dev"},
            }
        },
    )
    project, _, _ = load_project(tmp_path)
    resolved = resolve_profile(project, tmp_path)

    with pytest.raises(ProfileError, match="unknown source"):
        apply_source_path_overrides(
            [SourceConfig(name="filings", path="data/prod")], resolved
        )


# ─── env_var interpolation (issue #73) ──────────────────────────────────────


def _write_profiles_raw(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "profiles.yml"
    path.write_text(body)
    return path


def test_env_var_interpolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_project(tmp_path, profile="test_proj")
    _write_profiles_raw(
        tmp_path,
        "\n".join(
            [
                "test_proj:",
                "  target: dev",
                "  outputs:",
                "    dev:",
                "      warehouse:",
                "        type: duckdb",
                "        path: \"./target/{{ env_var('DBT_ML_DB_NAME') }}.duckdb\"",
                "        schema: \"{{ env_var('DBT_ML_SCHEMA', 'fallback_schema') }}\"",
            ]
        )
        + "\n",
    )
    monkeypatch.setenv("DBT_ML_DB_NAME", "from_env")
    monkeypatch.delenv("DBT_ML_SCHEMA", raising=False)

    project, _, _ = load_project(tmp_path)
    resolved = resolve_profile(project, tmp_path)
    assert resolved.warehouse.storage_location().endswith("from_env.duckdb")
    assert resolved.warehouse.schema_name == "fallback_schema"


def test_env_var_missing_without_default_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_project(tmp_path, profile="test_proj")
    _write_profiles_raw(
        tmp_path,
        "\n".join(
            [
                "test_proj:",
                "  target: dev",
                "  outputs:",
                "    dev:",
                "      warehouse:",
                "        type: duckdb",
                "        path: \"{{ env_var('DBT_ML_NOT_SET_ANYWHERE') }}\"",
            ]
        )
        + "\n",
    )
    monkeypatch.delenv("DBT_ML_NOT_SET_ANYWHERE", raising=False)
    project, _, _ = load_project(tmp_path)
    with pytest.raises(ProfileError, match="DBT_ML_NOT_SET_ANYWHERE"):
        resolve_profile(project, tmp_path)


def test_api_key_env_rejects_secret_interpolation_without_leaking_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_project(tmp_path, profile="test_proj")
    secret = "sk-ant-distinctive-secret-value"
    monkeypatch.setenv("DBT_ML_SECRET_KEY", secret)
    _write_profiles_raw(
        tmp_path,
        "\n".join(
            [
                "test_proj:",
                "  outputs:",
                "    dev:",
                "      warehouse:",
                "        type: duckdb",
                "        path: ./target/db.duckdb",
                "      llm:",
                '        api_key_env: "{{ env_var(\'DBT_ML_SECRET_KEY\') }}"',
            ]
        )
        + "\n",
    )

    project, _, _ = load_project(tmp_path)
    with pytest.raises(ProfileError, match="name an environment variable") as exc_info:
        resolve_profile(project, tmp_path)
    assert secret not in str(exc_info.value)


# ─── per-adapter warehouse config validation (issue #73) ────────────────────


def test_unknown_warehouse_type_raises(tmp_path: Path) -> None:
    _write_project(tmp_path, profile="test_proj")
    _write_profiles(
        tmp_path,
        targets={
            "dev": {
                "warehouse": {"type": "snowflake", "path": "./x", "schema": "s"}
            }
        },
    )
    project, _, _ = load_project(tmp_path)
    with pytest.raises(ProfileError, match="snowflake"):
        resolve_profile(project, tmp_path)


def test_invalid_duckdb_config_names_adapter(tmp_path: Path) -> None:
    _write_project(tmp_path, profile="test_proj")
    _write_profiles_raw(
        tmp_path,
        "\n".join(
            [
                "test_proj:",
                "  target: dev",
                "  outputs:",
                "    dev:",
                "      warehouse:",
                "        type: duckdb",
                "        schema: ok_but_no_path",
            ]
        )
        + "\n",
    )
    project, _, _ = load_project(tmp_path)
    with pytest.raises(ProfileError, match="duckdb"):
        resolve_profile(project, tmp_path)


def test_unknown_warehouse_field_rejected(tmp_path: Path) -> None:
    """Typo'd keys fail loudly instead of being silently ignored."""
    _write_project(tmp_path, profile="test_proj")
    _write_profiles_raw(
        tmp_path,
        "\n".join(
            [
                "test_proj:",
                "  target: dev",
                "  outputs:",
                "    dev:",
                "      warehouse:",
                "        type: duckdb",
                "        path: ./target/db.duckdb",
                "        pth_typo: ./oops",
            ]
        )
        + "\n",
    )
    project, _, _ = load_project(tmp_path)
    with pytest.raises(ProfileError, match="pth_typo") as exc_info:
        resolve_profile(project, tmp_path)
    message = str(exc_info.value)
    assert "profiles.yml:8:9" in message
    assert "test_proj.outputs.dev.warehouse.pth_typo" in message


@pytest.mark.parametrize("bad_key", ["source_path", "source-path"])
def test_unknown_target_field_rejected(tmp_path: Path, bad_key: str) -> None:
    _write_project(tmp_path, profile="test_proj")
    _write_profiles_raw(
        tmp_path,
        "\n".join(
            [
                "test_proj:",
                "  target: dev",
                "  outputs:",
                "    dev:",
                "      warehouse:",
                "        type: duckdb",
                "        path: ./target/db.duckdb",
                f"      {bad_key}:",
                "        filings: data/dev",
            ]
        )
        + "\n",
    )
    project, _, _ = load_project(tmp_path)

    with pytest.raises(ProfileError, match=bad_key) as exc_info:
        resolve_profile(project, tmp_path)
    message = str(exc_info.value)
    assert "profiles.yml:8:7" in message
    assert f"test_proj.outputs.dev.{bad_key}" in message


def test_profile_validation_diagnostic_does_not_echo_secret_input(
    tmp_path: Path,
) -> None:
    _write_project(tmp_path, profile="test_proj")
    secret = "sk-distinctive-invalid-profile-secret"
    _write_profiles_raw(
        tmp_path,
        "\n".join(
            [
                "test_proj:",
                "  outputs:",
                "    dev:",
                "      warehouse:",
                "        type: duckdb",
                "        path: ./target/db.duckdb",
                "      llm:",
                f"        api_key: {secret}",
            ]
        )
        + "\n",
    )

    project, _, _ = load_project(tmp_path)
    with pytest.raises(ProfileError) as exc_info:
        resolve_profile(project, tmp_path)

    message = str(exc_info.value)
    assert "profiles.yml:8:9" in message
    assert "test_proj.outputs.dev.llm.api_key" in message
    assert "Extra inputs are not permitted" in message
    assert secret not in message


def test_duplicate_profile_key_is_rejected_at_second_key_without_value(
    tmp_path: Path,
) -> None:
    _write_project(tmp_path, profile="test_proj")
    secret = "distinctive-duplicate-profile-secret"
    _write_profiles_raw(
        tmp_path,
        "\n".join(
            [
                "test_proj:",
                "  target: dev",
                f"  target: {secret}",
                "  outputs:",
                "    dev:",
                "      warehouse:",
                "        type: duckdb",
                "        path: ./target/db.duckdb",
            ]
        )
        + "\n",
    )

    project, _, _ = load_project(tmp_path)
    with pytest.raises(ProfileError) as exc_info:
        resolve_profile(project, tmp_path)

    message = str(exc_info.value)
    assert "profiles.yml:3:3 [test_proj.target]" in message
    assert "duplicate mapping key" in message
    assert secret not in message


@pytest.mark.parametrize("contents", ["[]\n", "0\n", "false\n"])
def test_falsy_profile_document_is_not_treated_as_empty_mapping(
    tmp_path: Path,
    contents: str,
) -> None:
    _write_project(tmp_path, profile="test_proj")
    _write_profiles_raw(tmp_path, contents)

    project, _, _ = load_project(tmp_path)
    with pytest.raises(ProfileError) as exc_info:
        resolve_profile(project, tmp_path)

    message = str(exc_info.value)
    assert "profiles.yml:1:1 [<root>]" in message
    assert "top-level must be a mapping of profile names" in message
