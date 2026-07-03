"""state:modified selection (issue #76): manifest-diff CI workflows."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from dbt_ml.config import load_project
from dbt_ml.config.model import ModelConfig
from dbt_ml.config.source import SourceConfig
from dbt_ml.dag import ProjectDAG, SelectionError
from dbt_ml.manifest import (
    StateError,
    compute_modified_models,
    read_state_code_versions,
    write_manifest,
)
from dbt_ml.runner import run_project
from dbt_ml.synth import generate_invoices


@pytest.fixture
def dag() -> ProjectDAG:
    sources = [SourceConfig(name="src", path="data")]
    models = [
        ModelConfig(name="a", source="ref('src')"),
        ModelConfig(name="b", depends_on=["ref('a')"]),
        ModelConfig(name="c", depends_on=["ref('b')"]),
    ]
    return ProjectDAG(sources, models)


# ─── selector grammar ───────────────────────────────────────────────────────


def test_state_modified_requires_state(dag: ProjectDAG) -> None:
    with pytest.raises(SelectionError, match="--state"):
        dag.select_models(select="state:modified")


def test_state_modified_selects_modified(dag: ProjectDAG) -> None:
    assert dag.select_models(select="state:modified", modified={"b"}) == ["b"]


def test_state_modified_plus_includes_descendants(dag: ProjectDAG) -> None:
    assert dag.select_models(select="state:modified+", modified={"a"}) == [
        "a",
        "b",
        "c",
    ]


def test_state_modified_empty_set_selects_nothing(dag: ProjectDAG) -> None:
    assert dag.select_models(select="state:modified+", modified=set()) == []


def test_unknown_state_selector_rejected(dag: ProjectDAG) -> None:
    with pytest.raises(SelectionError, match="state:modified"):
        dag.select_models(select="state:new", modified=set())


# ─── manifest state helpers ─────────────────────────────────────────────────


def test_read_state_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(StateError, match="No manifest found"):
        read_state_code_versions(tmp_path)


def test_read_state_invalid_json_raises(tmp_path: Path) -> None:
    bad = tmp_path / "manifest.json"
    bad.write_text("{not json")
    with pytest.raises(StateError, match="not valid JSON"):
        read_state_code_versions(bad)


def test_read_state_accepts_file_or_directory(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"models": [{"name": "m", "code_version": "abc"}]})
    )
    assert read_state_code_versions(manifest) == {"m": "abc"}
    assert read_state_code_versions(tmp_path) == {"m": "abc"}


# ─── end to end: the CI recipe ──────────────────────────────────────────────


@pytest.fixture
def fresh_project(tmp_path: Path, example_project_dir: Path) -> Path:
    dst = tmp_path / "project"
    shutil.copytree(
        example_project_dir,
        dst,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    generate_invoices(5, dst / "data" / "invoices", seed=1)
    return dst


def _snapshot_manifest(project: Path, tmp_path: Path) -> Path:
    """The CI pattern: store the manifest from a main-branch compile/run."""
    state_dir = tmp_path / "main-manifest"
    state_dir.mkdir()
    write_manifest(project)
    shutil.copy(project / "target" / "manifest.json", state_dir / "manifest.json")
    return state_dir


def test_unchanged_project_selects_nothing(fresh_project: Path, tmp_path: Path) -> None:
    state_dir = _snapshot_manifest(fresh_project, tmp_path)
    results = run_project(
        fresh_project, select="state:modified+", state=state_dir
    )
    assert results == []


def test_changed_transform_module_is_modified(
    fresh_project: Path, tmp_path: Path
) -> None:
    state_dir = _snapshot_manifest(fresh_project, tmp_path)
    run_project(fresh_project)

    transform = fresh_project / "transforms" / "summarize.py"
    transform.write_text(transform.read_text() + "\n# tweaked\n")

    results = run_project(fresh_project, select="state:modified+", state=state_dir)
    assert [r.model_name for r in results] == ["invoice_summary"]


def test_state_modified_composes_with_ancestors(
    fresh_project: Path, tmp_path: Path
) -> None:
    """+state:modified pulls in upstream models so the changed one can run."""
    state_dir = _snapshot_manifest(fresh_project, tmp_path)

    transform = fresh_project / "transforms" / "monthly_totals.py"
    transform.write_text(transform.read_text() + "\n# tweaked\n")

    results = run_project(fresh_project, select="+state:modified", state=state_dir)
    assert [r.model_name for r in results] == ["raw_invoices", "monthly_totals"]


def test_new_model_counts_as_modified(fresh_project: Path, tmp_path: Path) -> None:
    _, _, models = load_project(fresh_project)
    state_dir = _snapshot_manifest(fresh_project, tmp_path)
    # Simulate a manifest that predates one of the models
    manifest_path = state_dir / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["models"] = [m for m in data["models"] if m["name"] != "monthly_totals"]
    manifest_path.write_text(json.dumps(data))

    modified = compute_modified_models(models, fresh_project, state_dir)
    assert modified == {"monthly_totals"}
