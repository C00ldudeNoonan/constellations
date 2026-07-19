from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from dbt_ml.adapters import AdapterError, StateScope, create_adapter
from dbt_ml.config import load_project
from dbt_ml.config.source import SourceConfig
from dbt_ml.credentials import CredentialReference
from dbt_ml.hashing import canonical_fingerprint
from dbt_ml.profile import (
    ProfileError,
    ResolvedProfile,
    _load_profiles_file,
    apply_source_path_overrides,
    resolve_llm_options,
    resolve_profile,
)


def _assert_error_does_not_retain(
    error: BaseException,
    *sentinels: str,
) -> None:
    assert error.__cause__ is None
    assert error.__context__ is None
    traceback = error.__traceback__
    while traceback is not None:
        if "/src/dbt_ml/" in traceback.tb_frame.f_code.co_filename:
            rendered = repr(traceback.tb_frame.f_locals)
            for sentinel in sentinels:
                assert sentinel not in rendered
        traceback = traceback.tb_next


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
    assert options["provider"] == "anthropic"
    assert options["model"] == "claude-haiku-4-5"
    reference = options["api_key_env"]
    assert isinstance(reference, CredentialReference)
    assert "DBT_ML_ANTHROPIC_KEY" not in repr(reference)
    assert "DBT_ML_ANTHROPIC_KEY" not in repr(options)
    assert options["cache_path"].endswith("cache.duckdb")
    assert options["fields"] == [{"name": "x"}]


def test_vllm_endpoint_options_are_resolved_from_profile(tmp_path: Path) -> None:
    _write_project(tmp_path, profile="test_proj")
    _write_profiles(
        tmp_path,
        targets={
            "dev": {
                "warehouse": {
                    "type": "duckdb",
                    "path": "./d.duckdb",
                    "schema": "d",
                },
                "llm": {
                    "provider": "vllm",
                    "model": "invoice-extractor",
                    "base_url": "HTTPS://INFERENCE.EXAMPLE.TEST:443/v1/",
                    "timeout_seconds": 120,
                },
            }
        },
    )

    project, _, _ = load_project(tmp_path)
    resolved = resolve_profile(project, tmp_path)
    assert resolved.llm is not None
    assert resolved.llm.base_url == "https://inference.example.test/v1"
    options = resolve_llm_options({"fields": [{"name": "x"}]}, resolved)

    assert options["provider"] == "vllm"
    assert options["model"] == "invoice-extractor"
    assert options["base_url"] == "https://inference.example.test/v1"
    assert options["timeout_seconds"] == 120
    assert "api_key_env" not in options


def test_model_base_url_cannot_override_profile(tmp_path: Path) -> None:
    _write_project(tmp_path, profile="test_proj")
    _write_profiles(
        tmp_path,
        targets={
            "dev": {
                "warehouse": {
                    "type": "duckdb",
                    "path": "./d.duckdb",
                    "schema": "d",
                },
                "llm": {
                    "provider": "vllm",
                    "model": "invoice-extractor",
                    "base_url": "https://profile.example.test/v1",
                },
            }
        },
    )
    project, _, _ = load_project(tmp_path)
    resolved = resolve_profile(project, tmp_path)

    with pytest.raises(ProfileError, match=r"base_url.*operator-owned"):
        resolve_llm_options(
            {
                "base_url": "https://model.example.test/v1",
                "fields": [{"name": "x"}],
            },
            resolved,
        )


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


def test_model_provider_cannot_override_profile_credentials(tmp_path: Path) -> None:
    _write_project(tmp_path, profile="test_proj")
    _write_profiles(
        tmp_path,
        targets={
            "dev": {
                "warehouse": {
                    "type": "duckdb",
                    "path": "./d.duckdb",
                    "schema": "d",
                },
                "llm": {
                    "provider": "anthropic",
                    "api_key_env": "PROFILE_ANTHROPIC_KEY",
                },
            }
        },
    )
    project, _, _ = load_project(tmp_path)
    resolved = resolve_profile(project, tmp_path)

    with pytest.raises(ProfileError, match="cannot override the profile provider"):
        resolve_llm_options(
            {"provider": "another-provider", "fields": [{"name": "x"}]},
            resolved,
        )


def test_model_provider_requires_profile_selection_without_llm_block(
    tmp_path: Path,
) -> None:
    """Provider selection is operator-owned even when the profile has no
    `llm:` block — model YAML cannot opt into a non-default provider."""
    _write_project(tmp_path, profile="test_proj")
    _write_profiles(
        tmp_path,
        targets={
            "dev": {
                "warehouse": {
                    "type": "duckdb",
                    "path": "./d.duckdb",
                    "schema": "d",
                },
            }
        },
    )
    project, _, _ = load_project(tmp_path)
    resolved = resolve_profile(project, tmp_path)
    assert resolved.llm is None

    with pytest.raises(ProfileError, match="cannot override the profile provider"):
        resolve_llm_options(
            {"provider": "another-provider", "fields": [{"name": "x"}]},
            resolved,
        )

    options = resolve_llm_options(
        {"provider": "anthropic", "fields": [{"name": "x"}]}, resolved
    )
    assert options["provider"] == "anthropic"


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
    model_reference = "MODEL_ANTHROPIC_KEY"
    with pytest.raises(ProfileError, match="operator-owned") as exc_info:
        resolve_llm_options(
            {"api_key_env": model_reference, "fields": []}, resolved
        )

    _assert_error_does_not_retain(exc_info.value, model_reference)


def test_profile_selected_provider_enforces_native_batch_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = SimpleNamespace(
        default_credential_env=None,
        supports_native_batch=False,
        resolve_model=lambda model: model or "sync-model",
        resolve_base_url=lambda base_url: base_url,
    )
    monkeypatch.setattr("dbt_ml.profile.get_inference_provider", lambda _name: provider)
    _write_project(tmp_path, profile="test_proj")
    _write_profiles(
        tmp_path,
        targets={
            "dev": {
                "warehouse": {
                    "type": "duckdb",
                    "path": "./d.duckdb",
                    "schema": "main",
                },
                "llm": {"provider": "sync-only-profile-test"},
            }
        },
    )
    project, _, _ = load_project(tmp_path)
    resolved = resolve_profile(project, tmp_path)

    with pytest.raises(ProfileError, match="does not support native batch"):
        resolve_llm_options(
            {"batch": True, "fields": [{"name": "x"}]},
            resolved,
        )


def test_provider_initialization_failure_is_a_profile_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dbt_ml.providers import ProviderConfigurationError

    def fail_provider(_name: str) -> None:
        raise ProviderConfigurationError("provider initialization failed safely")

    monkeypatch.setattr("dbt_ml.profile.get_inference_provider", fail_provider)
    _write_project(tmp_path, profile="test_proj")
    _write_profiles(
        tmp_path,
        targets={
            "dev": {
                "warehouse": {
                    "type": "duckdb",
                    "path": "./d.duckdb",
                    "schema": "main",
                },
                "llm": {"provider": "failing-profile-provider", "model": "m"},
            }
        },
    )
    project, _, _ = load_project(tmp_path)

    with pytest.raises(ProfileError, match="initialization failed safely"):
        resolve_profile(project, tmp_path)


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
    _assert_error_does_not_retain(
        exc_info.value,
        secret,
        "DBT_ML_SECRET_KEY",
    )


def _bigquery_profile(credential_lines: list[str]) -> str:
    return (
        "\n".join(
            [
                "test_proj:",
                "  target: prod",
                "  outputs:",
                "    prod:",
                "      warehouse:",
                "        type: bigquery",
                "        project: example-project",
                *[f"        {line}" for line in credential_lines],
            ]
        )
        + "\n"
    )


def test_bigquery_references_remain_opaque_until_sdk_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_project(tmp_path, profile="test_proj")
    env_names = (
        "DISTINCTIVE_REFRESH_REFERENCE",
        "DISTINCTIVE_CLIENT_SECRET_REFERENCE",
        "DISTINCTIVE_TOKEN_URI_REFERENCE",
    )
    _write_profiles_raw(
        tmp_path,
        _bigquery_profile(
            [
                'refresh_token: "{{ env_var(\'DISTINCTIVE_REFRESH_REFERENCE\') }}"',
                "client_id: public-client-id",
                'client_secret: "{{ env_var(\'DISTINCTIVE_CLIENT_SECRET_REFERENCE\') }}"',
                'token_uri: "{{ env_var(\'DISTINCTIVE_TOKEN_URI_REFERENCE\') }}"',
            ]
        ),
    )
    for env_name in env_names:
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setattr(
        CredentialReference,
        "resolve",
        lambda self: pytest.fail("profile resolution accessed a credential"),
    )

    project, _, _ = load_project(tmp_path)
    resolved = resolve_profile(project, tmp_path)
    profiles, _ = _load_profiles_file(tmp_path / "profiles.yml")

    rendered: list[str] = [
        repr(resolved),
        repr(resolved.warehouse),
        repr(resolved.warehouse.model_dump()),
        resolved.warehouse.model_dump_json(),
        repr(profiles),
        repr(profiles["test_proj"].model_dump()),
        profiles["test_proj"].model_dump_json(),
    ]
    for env_name in env_names:
        assert all(env_name not in value for value in rendered)


def test_missing_inactive_bigquery_reference_does_not_break_active_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_project(tmp_path, profile="test_proj")
    env_name = "DISTINCTIVE_INACTIVE_BQ_TOKEN"
    monkeypatch.delenv(env_name, raising=False)
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
                "        path: ./target/dev.duckdb",
                "    prod:",
                "      warehouse:",
                "        type: bigquery",
                "        project: example-project",
                f'        token: "{{{{ env_var(\'{env_name}\') }}}}"',
            ]
        )
        + "\n",
    )
    project, _, _ = load_project(tmp_path)

    assert resolve_profile(project, tmp_path).warehouse.type == "duckdb"
    prod = resolve_profile(project, tmp_path, target="prod")
    adapter = create_adapter(prod.warehouse, project_dir=tmp_path)
    with pytest.raises(AdapterError) as exc_info:
        adapter._credentials()  # type: ignore[attr-defined]

    message = str(exc_info.value)
    assert "not set or is empty" in message
    assert env_name not in message


def test_environment_selected_bigquery_type_protects_credentials_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_project(tmp_path, profile="test_proj")
    env_name = "DISTINCTIVE_TYPE_SELECTED_BQ_TOKEN"
    monkeypatch.setenv("DBT_ML_WAREHOUSE_TYPE", "bigquery")
    monkeypatch.delenv(env_name, raising=False)
    _write_profiles_raw(
        tmp_path,
        _bigquery_profile(
            [f'token: "{{{{ env_var(\'{env_name}\') }}}}"']
        ).replace("type: bigquery", 'type: "{{ env_var(\'DBT_ML_WAREHOUSE_TYPE\') }}"'),
    )
    project, _, _ = load_project(tmp_path)

    resolved = resolve_profile(project, tmp_path)
    adapter = create_adapter(resolved.warehouse, project_dir=tmp_path)

    with pytest.raises(AdapterError) as exc_info:
        adapter._credentials()  # type: ignore[attr-defined]

    assert env_name not in str(exc_info.value)


def test_unknown_adapter_protects_references_without_resolving_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_project(tmp_path, profile="test_proj")
    env_name = "DISTINCTIVE_UNKNOWN_ADAPTER_TOKEN"
    secret = "distinctive-unknown-adapter-secret"
    monkeypatch.setenv(env_name, secret)
    _write_profiles_raw(
        tmp_path,
        _bigquery_profile(
            [f'token: "{{{{ env_var(\'{env_name}\') }}}}"']
        ).replace("type: bigquery", "type: future-warehouse"),
    )
    project, _, _ = load_project(tmp_path)

    with pytest.raises(ProfileError) as exc_info:
        resolve_profile(project, tmp_path)

    message = str(exc_info.value)
    assert "No adapter registered" in message
    assert env_name not in message
    assert secret not in message


def test_inactive_unknown_adapter_keeps_credential_reference_opaque(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_project(tmp_path, profile="test_proj")
    env_name = "DISTINCTIVE_INACTIVE_FUTURE_TOKEN"
    monkeypatch.delenv(env_name, raising=False)
    _write_profiles(
        tmp_path,
        targets={
            "dev": {
                "warehouse": {
                    "type": "duckdb",
                    "path": "./target/dev.duckdb",
                    "schema": "dbt_ml",
                }
            }
        },
    )
    profiles_path = tmp_path / "profiles.yml"
    profiles_path.write_text(
        profiles_path.read_text()
        + "    future:\n"
        + "      warehouse:\n"
        + "        type: future-warehouse\n"
        + f'        token: "{{{{ env_var(\'{env_name}\') }}}}"\n'
    )
    profiles, _ = _load_profiles_file(profiles_path)

    future = profiles["test_proj"].outputs["future"]
    rendered = repr(future) + repr(future.model_dump()) + future.model_dump_json()
    assert env_name not in rendered
    assert isinstance(future.warehouse["token"], CredentialReference)


def test_protected_field_names_do_not_affect_nonwarehouse_interpolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_project(tmp_path, profile="test_proj")
    monkeypatch.setenv("TOKEN_SOURCE_PATH", "data/token-source")
    _write_profiles_raw(
        tmp_path,
        "\n".join(
            [
                "test_proj:",
                "  outputs:",
                "    dev:",
                "      warehouse:",
                "        type: duckdb",
                "        path: ./target/dev.duckdb",
                "      source_paths:",
                '        token: "{{ env_var(\'TOKEN_SOURCE_PATH\') }}"',
            ]
        )
        + "\n",
    )
    project, _, _ = load_project(tmp_path)

    resolved = resolve_profile(project, tmp_path)

    assert resolved.source_paths == {"token": "data/token-source"}


@pytest.mark.parametrize(
    ("field_name", "unsafe_value"),
    [
        ("token", "literal-secret-sentinel"),
        ("keyfile_json", '{"private_key":"literal-secret-sentinel"}'),
        ("keyfile_json", "{private_key: literal-secret-sentinel}"),
        (
            "client_secret",
            '"{{ env_var(\'DISTINCTIVE_UNSAFE_REFERENCE\', \'fallback-secret\') }}"',
        ),
        (
            "refresh_token",
            '"prefix-{{ env_var(\'DISTINCTIVE_UNSAFE_REFERENCE\') }}"',
        ),
    ],
)
def test_bigquery_legacy_secret_forms_fail_without_echoing_input(
    tmp_path: Path,
    field_name: str,
    unsafe_value: str,
) -> None:
    _write_project(tmp_path, profile="test_proj")
    _write_profiles_raw(
        tmp_path,
        _bigquery_profile([f"{field_name}: {unsafe_value}"]),
    )
    project, _, _ = load_project(tmp_path)

    with pytest.raises(ProfileError) as exc_info:
        resolve_profile(project, tmp_path)

    message = str(exc_info.value)
    assert field_name in message
    assert "literal-secret-sentinel" not in message
    assert "DISTINCTIVE_UNSAFE_REFERENCE" not in message
    assert "fallback-secret" not in message


def test_bigquery_credential_reference_rotation_does_not_change_identity() -> None:
    from dbt_ml.adapters import parse_warehouse_config

    first = parse_warehouse_config(
        {
            "type": "bigquery",
            "project": "example-project",
            "token": "{{ env_var('FIRST_PRIVATE_TOKEN_REFERENCE') }}",
        }
    )
    second = parse_warehouse_config(
        {
            "type": "bigquery",
            "project": "example-project",
            "token": "{{ env_var('SECOND_PRIVATE_TOKEN_REFERENCE') }}",
        }
    )

    assert first == second
    assert canonical_fingerprint(
        first, domain="profile-test"
    ) == canonical_fingerprint(second, domain="profile-test")
    assert StateScope.for_target_descriptor(
        "model",
        stage="retrieval_publish",
        descriptor={"warehouse": first},
    ) == StateScope.for_target_descriptor(
        "model",
        stage="retrieval_publish",
        descriptor={"warehouse": second},
    )
    rendered = repr(first.model_dump()) + first.model_dump_json()
    assert "FIRST_PRIVATE_TOKEN_REFERENCE" not in rendered


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
    sentinel = "distinctive-invalid-warehouse-secret"
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
                f"        pth_typo: {sentinel}",
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
    assert sentinel not in message
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    _assert_error_does_not_retain(exc_info.value, sentinel)


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
    _assert_error_does_not_retain(exc_info.value, secret)


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
    _assert_error_does_not_retain(exc_info.value, secret)


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
