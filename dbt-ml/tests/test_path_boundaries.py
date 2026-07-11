"""Filesystem boundary policy (issue #65).

Project-YAML paths are confined to the project directory (ConfigError / exit 2
on escape) with explicit per-path opt-ins; profiles.yml paths are trusted and
`clean` never invokes warehouse deletion.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from dbt_ml.cli import cli
from dbt_ml.config import ConfigError, SourceConfig, load_project
from dbt_ml.config.model import MLArtifactConfig, MLConfig, ModelConfig
from dbt_ml.config.project import ProjectConfig
from dbt_ml.paths import resolve_within_project
from dbt_ml.runner import run_project
from dbt_ml.sources import LocalDocumentSource, SourceError
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


def test_legacy_inline_duckdb_path_escape_rejected(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _minimal_project(project, "duckdb:\n  path: ../outside.duckdb\n")
    with pytest.raises(ConfigError, match=r"duckdb\.path"):
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


@pytest.mark.parametrize(
    ("pattern", "message"),
    [
        ("/tmp/*.json", "relative path pattern"),
        (r"C:\outside\*.json", "relative path pattern"),
        ("../outside/*.json", "parent traversal"),
        (r"..\outside\*.json", "parent traversal"),
    ],
)
def test_source_file_pattern_rejects_escape_syntax(
    pattern: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SourceConfig(name="docs", path="data", file_pattern=pattern)


def test_local_source_revalidates_file_pattern_before_globbing(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (tmp_path / "secret.json").write_text("{}")
    source = SourceConfig(name="docs", path="data")
    source.file_pattern = "../secret.json"

    with pytest.raises(ConfigError, match="parent traversal"):
        LocalDocumentSource().discover(source, tmp_path)


@pytest.mark.parametrize("operation", ["discover", "scan"])
def test_local_source_rejects_leaf_symlink(
    tmp_path: Path, operation: str
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    target = data / "target.json"
    target.write_text("{}")
    link = data / "link.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable (Windows without privilege)")

    source = SourceConfig(name="docs", path="data", file_pattern="link.json")
    with pytest.raises(ConfigError, match="matched symlink"):
        getattr(LocalDocumentSource(), operation)(source, tmp_path)


@pytest.mark.parametrize("operation", ["discover", "scan"])
def test_local_source_rejects_match_through_escaping_directory_symlink(
    tmp_path: Path, operation: str
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.json").write_text("{}")
    try:
        (data / "linked").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable (Windows without privilege)")

    source = SourceConfig(
        name="docs",
        path="data",
        file_pattern="linked/*.json",
        recursive=False,
    )
    with pytest.raises(ConfigError, match="outside the source root"):
        getattr(LocalDocumentSource(), operation)(source, tmp_path)


@pytest.mark.parametrize("operation", ["discover", "scan"])
def test_local_source_rejects_match_through_internal_directory_symlink(
    tmp_path: Path, operation: str
) -> None:
    data = tmp_path / "data"
    target = data / "target"
    target.mkdir(parents=True)
    (target / "doc.json").write_text("{}")
    try:
        (data / "linked").symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable (Windows without privilege)")

    source = SourceConfig(
        name="docs",
        path="data",
        file_pattern="linked/*.json",
        recursive=False,
    )
    with pytest.raises(ConfigError, match="traverse symlink"):
        getattr(LocalDocumentSource(), operation)(source, tmp_path)


def test_local_source_preserves_nested_pattern_semantics(tmp_path: Path) -> None:
    data = tmp_path / "data"
    (data / "2026").mkdir(parents=True)
    (data / "archive" / "2026").mkdir(parents=True)
    (data / "2026" / "current.json").write_text("{}")
    (data / "archive" / "2026" / "old.json").write_text("{}")
    (data / "root.json").write_text("{}")
    source = SourceConfig(
        name="docs",
        path="data",
        file_pattern="2026/*.json",
        recursive=False,
    )

    direct = LocalDocumentSource().discover(source, tmp_path)
    source.recursive = True
    recursive = LocalDocumentSource().discover(source, tmp_path)

    assert [ref.relative_path for ref in direct] == ["2026/current.json"]
    assert [ref.relative_path for ref in recursive] == [
        "2026/current.json",
        "archive/2026/old.json",
    ]


def test_local_source_fetch_rejects_directory_symlink_swap(tmp_path: Path) -> None:
    data = tmp_path / "data"
    nested = data / "nested"
    nested.mkdir(parents=True)
    (nested / "doc.json").write_text('{"scope": "inside"}')
    source = SourceConfig(
        name="docs", path="data", file_pattern="nested/*.json"
    )
    local_source = LocalDocumentSource()
    ref = local_source.discover(source, tmp_path)[0]

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "doc.json").write_text('{"scope": "outside"}')
    nested.rename(data / "original_nested")
    try:
        nested.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable (Windows without privilege)")
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    with pytest.raises(SourceError, match="symlink before fetch"):
        local_source.fetch(ref, work_dir)
    assert list(work_dir.iterdir()) == []


def test_local_source_fetch_snapshots_verified_file(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    original = data / "doc.json"
    original.write_text('{"scope": "inside"}')
    source = SourceConfig(name="docs", path="data", file_pattern="*.json")
    local_source = LocalDocumentSource()
    ref = local_source.discover(source, tmp_path)[0]
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    fetched = local_source.fetch(ref, work_dir)

    assert fetched != original
    assert fetched.name == original.name
    assert fetched.read_text() == original.read_text()
    assert fetched.is_relative_to(work_dir)


def test_external_source_matches_remain_confined_to_external_root(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (external / "allowed.json").write_text("{}")
    source = SourceConfig(
        name="docs", path=str(external), file_pattern="*.json", external=True
    )

    refs = LocalDocumentSource().discover(source, project)

    assert [ref.relative_path for ref in refs] == ["allowed.json"]

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.json").write_text("{}")
    try:
        (external / "linked").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable (Windows without privilege)")
    source.file_pattern = "linked/*.json"
    source.recursive = False
    with pytest.raises(ConfigError, match="outside the source root"):
        LocalDocumentSource().discover(source, project)


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


# ─── clean ───────────────────────────────────────────────────────────────────


def test_clean_removes_only_known_artifacts_and_preserves_warehouse(
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
    target = dst / "target"
    warehouse = target / "dbt_ml.duckdb"
    (target / "manifest.json").write_text("{}")
    (target / "run_results.json").write_text("{}")
    (target / "sources.yml").write_text("version: 2\n")
    (target / "docs").mkdir()
    (target / "docs" / "index.html").write_text("generated")
    (target / "artifacts").mkdir()
    (target / "artifacts" / "model.bin").write_bytes(b"generated")
    unknown = target / "keep.txt"
    unknown.write_text("not owned by dbt-ml")

    runner = CliRunner()
    result = runner.invoke(cli, ["--project-dir", str(dst), "clean"])
    assert result.exit_code == 0, result.output
    assert warehouse.exists()
    assert unknown.read_text() == "not owned by dbt-ml"
    assert not (target / "manifest.json").exists()
    assert not (target / "run_results.json").exists()
    assert not (target / "sources.yml").exists()
    assert not (target / "docs").exists()
    assert not (target / "artifacts").exists()


def test_clean_never_deletes_external_warehouse(
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
    wh.write_bytes(b"external warehouse sentinel")
    profiles = dst / "profiles.yml"
    profiles.write_text(
        profiles.read_text().replace(
            "path: ./target/dbt_ml.duckdb", f"path: {wh.as_posix()}"
        )
    )

    target = dst / "target"
    target.mkdir()
    (target / "manifest.json").write_text("{}")

    runner = CliRunner()
    result = runner.invoke(cli, ["--project-dir", str(dst), "clean"])
    assert result.exit_code == 0, result.output
    assert wh.read_bytes() == b"external warehouse sentinel"
    assert not target.exists()


def test_clean_refuses_symlinked_target(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _minimal_project(project)
    actual = project / "actual_target"
    actual.mkdir()
    sentinel = actual / "keep.txt"
    sentinel.write_text("keep")
    target = project / "target"
    try:
        target.symlink_to(actual, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable (Windows without privilege)")

    result = CliRunner().invoke(cli, ["--project-dir", str(project), "clean"])

    assert result.exit_code == 2, result.output
    assert "symlink component" in result.output
    assert target.is_symlink()
    assert sentinel.read_text() == "keep"


@pytest.mark.parametrize("target_path", [".", "target/.."])
def test_clean_refuses_project_root(tmp_path: Path, target_path: str) -> None:
    project = tmp_path / "project"
    _minimal_project(project, f"target-path: {target_path}\n")
    sentinel = project / "keep.txt"
    sentinel.write_text("keep")

    result = CliRunner().invoke(cli, ["--project-dir", str(project), "clean"])

    assert result.exit_code == 2, result.output
    assert "project root" in result.output
    assert sentinel.read_text() == "keep"


def test_clean_refuses_layout_overlap(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _minimal_project(project, "target-path: models\n")
    models = project / "models"
    models.mkdir()
    sentinel = models / "keep.yml"
    sentinel.write_text("version: 2\nmodels: []\n")

    result = CliRunner().invoke(cli, ["--project-dir", str(project), "clean"])

    assert result.exit_code == 2, result.output
    assert "overlaps" in result.output
    assert sentinel.exists()


def test_clean_refuses_intermediate_symlink(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _minimal_project(project, "target-path: linked/target\n")
    actual = project / "actual"
    target = actual / "target"
    target.mkdir(parents=True)
    sentinel = target / "manifest.json"
    sentinel.write_text("keep")
    try:
        (project / "linked").symlink_to(actual, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable (Windows without privilege)")

    result = CliRunner().invoke(cli, ["--project-dir", str(project), "clean"])

    assert result.exit_code == 2, result.output
    assert "symlink component" in result.output
    assert sentinel.read_text() == "keep"


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
