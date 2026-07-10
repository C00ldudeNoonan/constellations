"""Filesystem boundary policy (issue #65).

Project-YAML paths are confined to the project directory (ConfigError / exit 2
on escape) with explicit per-path opt-ins; profiles.yml paths are trusted, but
`clean` needs --force to delete a warehouse file outside the project.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from dbt_ml.cli import cli
from dbt_ml.config import ConfigError, load_project
from dbt_ml.config.model import MLArtifactConfig, MLConfig, ModelConfig
from dbt_ml.config.project import ProjectConfig
from dbt_ml.paths import resolve_within_project
from dbt_ml.runner import run_project
from dbt_ml.synth import generate_invoices
from dbt_ml.versioning import compute_code_version

# ─── helper ──────────────────────────────────────────────────────────────────


def test_relative_inside_ok(tmp_path: Path) -> None:
    resolved = resolve_within_project("data/docs", tmp_path, surface="test")
    assert resolved == (tmp_path / "data" / "docs").resolve()


def test_dotdot_escape_rejected(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    with pytest.raises(ConfigError, match="outside the project directory"):
        resolve_within_project("../outside", project, surface="test")


def test_absolute_escape_rejected(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    with pytest.raises(ConfigError, match="outside the project directory"):
        resolve_within_project(tmp_path / "elsewhere", project, surface="test")


def test_external_allows_escape(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    resolved = resolve_within_project(
        tmp_path / "elsewhere", project, surface="test", external=True
    )
    assert resolved == (tmp_path / "elsewhere").resolve()


def test_symlink_escape_rejected(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = project / "data"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable (Windows without privilege)")
    with pytest.raises(ConfigError, match="outside the project directory"):
        resolve_within_project("data", project, surface="test")


# ─── layout paths ────────────────────────────────────────────────────────────


def _minimal_project(project_dir: Path, extra: str = "") -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "dbt_ml_project.yml").write_text(
        f"name: p\nversion: '0.1.0'\n{extra}"
    )


def test_layout_target_path_escape_rejected(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _minimal_project(project, "target-path: ../t\n")
    with pytest.raises(ConfigError, match="target-path"):
        load_project(project)


def test_layout_source_paths_escape_rejected(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _minimal_project(project, "source-paths: ['../models']\n")
    with pytest.raises(ConfigError, match="source-paths"):
        load_project(project)


# ─── source path (read + seed) ───────────────────────────────────────────────


@pytest.fixture
def escaping_project(tmp_path: Path, example_project_dir: Path) -> Path:
    """invoice_pipeline copy whose source points one level above the project."""
    dst = tmp_path / "project"
    shutil.copytree(
        example_project_dir,
        dst,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    src_yml = dst / "sources" / "invoices.yml"
    src_yml.write_text(
        src_yml.read_text().replace('"./data/invoices/"', '"../outside_data/"')
    )
    generate_invoices(3, tmp_path / "outside_data", seed=1)
    return dst


def test_run_rejects_escaping_source(escaping_project: Path) -> None:
    with pytest.raises(ConfigError, match="external: true"):
        run_project(escaping_project, select="raw_invoices")


def test_run_allows_external_source_with_flag(escaping_project: Path) -> None:
    src_yml = escaping_project / "sources" / "invoices.yml"
    src_yml.write_text(
        src_yml.read_text().replace(
            'path: "../outside_data/"',
            'path: "../outside_data/"\n    external: true',
        )
    )
    results = run_project(escaping_project, select="raw_invoices")
    assert results[0].rows_written == 3


def test_seed_rejects_escaping_source(escaping_project: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--project-dir", str(escaping_project), "seed", "--count", "2"]
    )
    assert result.exit_code == 2, result.output
    assert "outside the project directory" in result.output


def test_seed_allows_external_source_with_flag(escaping_project: Path) -> None:
    src_yml = escaping_project / "sources" / "invoices.yml"
    src_yml.write_text(
        src_yml.read_text().replace(
            'path: "../outside_data/"',
            'path: "../outside_data/"\n    external: true',
        )
    )
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--project-dir", str(escaping_project), "seed", "--count", "2"]
    )
    assert result.exit_code == 0, result.output


# ─── ml.artifact.path ────────────────────────────────────────────────────────


def _ml(path: str, external: bool = False) -> MLConfig:
    return MLConfig(
        task="features",
        provider="builtin.tfidf",
        text_field="body",
        artifact=MLArtifactConfig(path=Path(path), external=external),
    )


def test_artifact_path_escape_rejected(tmp_path: Path) -> None:
    from dbt_ml.classic_ml import _artifact_path

    project = tmp_path / "project"
    project.mkdir()
    model = ModelConfig(name="m", ml=_ml("../artifacts"))
    with pytest.raises(ConfigError, match=r"ml\.artifact\.path"):
        _artifact_path(model.ml, model, ProjectConfig(name="p"), project)


def test_artifact_path_external_allowed(tmp_path: Path) -> None:
    from dbt_ml.classic_ml import _artifact_path

    project = tmp_path / "project"
    project.mkdir()
    model = ModelConfig(name="m", ml=_ml("../artifacts", external=True))
    resolved = _artifact_path(model.ml, model, ProjectConfig(name="p"), project)
    assert resolved == (tmp_path / "artifacts").resolve()


# ─── model-level llm cache_path ──────────────────────────────────────────────


def test_model_level_llm_cache_path_confined(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    dst = tmp_path / "llm_proj"
    shutil.copytree(
        repo / "examples" / "llm_invoice_pipeline",
        dst,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    model = dst / "models" / "raw_invoices_llm.yml"
    model.write_text(
        model.read_text().replace(
            "      options:",
            "      options:\n        cache_path: ../outside_cache.duckdb",
            1,
        )
    )
    (dst / "data" / "invoices_text").mkdir(parents=True)
    (dst / "data" / "invoices_text" / "doc.txt").write_text("INVOICE")

    with pytest.raises(ConfigError, match=r"profiles\.yml"):
        run_project(dst)


# ─── clean --force ───────────────────────────────────────────────────────────


def test_clean_in_project_needs_no_force(
    tmp_path: Path, example_project_dir: Path
) -> None:
    dst = tmp_path / "project"
    shutil.copytree(
        example_project_dir,
        dst,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    generate_invoices(2, dst / "data" / "invoices", seed=1)
    run_project(dst, select="raw_invoices")

    runner = CliRunner()
    result = runner.invoke(cli, ["--project-dir", str(dst), "clean"])
    assert result.exit_code == 0, result.output
    assert not (dst / "target" / "dbt_ml.duckdb").exists()


def test_clean_outside_project_requires_force(
    tmp_path: Path, example_project_dir: Path
) -> None:
    dst = tmp_path / "project"
    shutil.copytree(
        example_project_dir,
        dst,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    wh = tmp_path / "elsewhere" / "wh.duckdb"
    wh.parent.mkdir(parents=True)
    wh.write_bytes(b"")
    profiles = dst / "profiles.yml"
    profiles.write_text(
        profiles.read_text().replace(
            "path: ./target/dbt_ml.duckdb", f"path: {wh.as_posix()}"
        )
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["--project-dir", str(dst), "clean"])
    assert result.exit_code == 2, result.output
    assert "--force" in result.output
    assert wh.exists()

    forced = runner.invoke(cli, ["--project-dir", str(dst), "clean", "--force"])
    assert forced.exit_code == 0, forced.output
    assert not wh.exists()


# ─── code_version hygiene ────────────────────────────────────────────────────


def test_code_version_ignores_artifact_external(tmp_path: Path) -> None:
    a = compute_code_version(
        extraction=None, transform=None, ml=_ml("target/a"), project_dir=tmp_path
    )
    b = compute_code_version(
        extraction=None,
        transform=None,
        ml=_ml("target/a", external=True),
        project_dir=tmp_path,
    )
    assert a == b
