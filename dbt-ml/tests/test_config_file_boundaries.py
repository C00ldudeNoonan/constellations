from __future__ import annotations

from pathlib import Path

import pytest

from dbt_ml.config import ConfigError, load_project
from dbt_ml.profile import (
    LEGACY_PROFILES_DIR_ENV,
    PROFILES_DIR_ENV,
    ProfileError,
    resolve_profile,
)


def _symlink_or_skip(
    link: Path, target: Path, *, target_is_directory: bool = False
) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are unavailable on this platform")


def _write_project(project: Path, *, profile: bool = False) -> None:
    project.mkdir()
    body = "name: config_boundaries\n"
    if profile:
        body += "profile: secure\n"
    (project / "dbt_ml_project.yml").write_text(body)


def _profile_yaml() -> str:
    return (
        "secure:\n"
        "  target: dev\n"
        "  outputs:\n"
        "    dev:\n"
        "      warehouse:\n"
        "        type: duckdb\n"
        "        path: ./target/db.duckdb\n"
        "        schema: secure\n"
    )


def test_project_file_symlink_is_rejected(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside_project.yml"
    outside.write_text("name: escaped\n")
    project_file = project / "dbt_ml_project.yml"
    _symlink_or_skip(project_file, outside)

    with pytest.raises(ConfigError, match=r"dbt_ml_project\.yml.*symlink"):
        load_project(project)


def test_project_file_must_be_regular(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "dbt_ml_project.yml").mkdir()

    with pytest.raises(ConfigError, match="regular non-symlink file"):
        load_project(project)


@pytest.mark.parametrize(
    ("directory", "body", "description"),
    [
        (
            "sources",
            "version: 2\nsources:\n  - name: escaped\n    path: data\n",
            "source configuration",
        ),
        (
            "models",
            "version: 2\nmodels:\n  - name: escaped\n"
            "    extraction:\n      backend: json\n",
            "model configuration",
        ),
    ],
)
def test_discovered_yaml_symlink_is_rejected(
    tmp_path: Path, directory: str, body: str, description: str
) -> None:
    project = tmp_path / "project"
    _write_project(project)
    config_root = project / directory
    config_root.mkdir()
    outside = tmp_path / f"outside_{directory}.yml"
    outside.write_text(body)
    linked = config_root / "linked.yml"
    _symlink_or_skip(linked, outside)

    with pytest.raises(ConfigError, match=rf"{description} symlink.*linked\.yml"):
        load_project(project)


@pytest.mark.parametrize("directory", ["sources", "models"])
def test_symlinked_yaml_subdirectory_is_rejected(
    tmp_path: Path, directory: str
) -> None:
    project = tmp_path / "project"
    _write_project(project)
    config_root = project / directory
    config_root.mkdir()
    outside = tmp_path / f"outside_{directory}"
    outside.mkdir()
    (outside / "hidden.yml").write_text("version: 2\n")
    linked = config_root / "linked"
    _symlink_or_skip(linked, outside, target_is_directory=True)

    with pytest.raises(ConfigError, match=rf"symlinked {directory[:-1]} configuration"):
        load_project(project)


def test_nested_regular_yaml_files_load(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write_project(project)
    source_root = project / "sources" / "nested"
    source_root.mkdir(parents=True)
    (source_root / "docs.yml").write_text(
        "version: 2\nsources:\n  - name: docs\n    path: data\n"
    )
    model_root = project / "models" / "nested"
    model_root.mkdir(parents=True)
    (model_root / "raw.yml").write_text(
        "version: 2\nmodels:\n  - name: raw_docs\n"
        "    source: ref('docs')\n"
        "    extraction:\n      backend: json\n"
    )

    _, sources, models = load_project(project)

    assert [source.name for source in sources] == ["docs"]
    assert [model.name for model in models] == ["raw_docs"]


def test_implicit_project_profile_symlink_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(PROFILES_DIR_ENV, raising=False)
    monkeypatch.delenv(LEGACY_PROFILES_DIR_ENV, raising=False)
    project = tmp_path / "project"
    _write_project(project, profile=True)
    outside = tmp_path / "outside_profiles.yml"
    outside.write_text(_profile_yaml())
    linked = project / "profiles.yml"
    _symlink_or_skip(linked, outside)
    config, _, _ = load_project(project)

    with pytest.raises(ProfileError, match="symlinked project-local profiles"):
        resolve_profile(config, project)


def test_explicit_profiles_symlink_remains_operator_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(PROFILES_DIR_ENV, raising=False)
    monkeypatch.delenv(LEGACY_PROFILES_DIR_ENV, raising=False)
    project = tmp_path / "project"
    _write_project(project, profile=True)
    untrusted = tmp_path / "untrusted_profiles.yml"
    untrusted.write_text("not: the selected profile\n")
    _symlink_or_skip(project / "profiles.yml", untrusted)

    trusted_file = tmp_path / "trusted_profiles.yml"
    trusted_file.write_text(_profile_yaml())
    profiles_dir = tmp_path / "operator_profiles"
    profiles_dir.mkdir()
    explicit = profiles_dir / "profiles.yml"
    _symlink_or_skip(explicit, trusted_file)
    config, _, _ = load_project(project)

    resolved = resolve_profile(config, project, profiles_dir=profiles_dir)

    assert resolved.profile_name == "secure"
    assert resolved.profiles_path == explicit


def test_global_profiles_symlink_remains_operator_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(PROFILES_DIR_ENV, raising=False)
    monkeypatch.delenv(LEGACY_PROFILES_DIR_ENV, raising=False)
    project = tmp_path / "project"
    _write_project(project, profile=True)
    trusted_file = tmp_path / "trusted_profiles.yml"
    trusted_file.write_text(_profile_yaml())
    home = tmp_path / "home"
    global_dir = home / ".dbt_ml"
    global_dir.mkdir(parents=True)
    global_profile = global_dir / "profiles.yml"
    _symlink_or_skip(global_profile, trusted_file)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    config, _, _ = load_project(project)

    resolved = resolve_profile(config, project)

    assert resolved.profile_name == "secure"
    assert resolved.profiles_path == global_profile
