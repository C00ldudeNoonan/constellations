from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from click.testing import CliRunner

from stel.cli import cli
from stel.config.source import SourceConfig
from stel.manifest import write_run_results
from stel.runner import build_project, run_project
from stel.sources import DocumentRef, GCSDocumentSource, LocalDocumentSource


@pytest.fixture
def mixed_source_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "stel_project.yml").write_text(
        "name: mixed_sources\nversion: '0.1.0'\nprofile: mixed_sources\n"
    )
    (project / "profiles.yml").write_text(
        "mixed_sources:\n"
        "  target: dev\n"
        "  outputs:\n"
        "    dev:\n"
        "      warehouse:\n"
        "        type: duckdb\n"
        "        path: ./target/db.duckdb\n"
        "        schema: mixed\n"
    )
    sources = project / "sources"
    sources.mkdir()
    (sources / "sources.yml").write_text(
        "version: 2\n"
        "sources:\n"
        "  - name: local_docs\n"
        "    path: ./data/local\n"
        "    file_pattern: '*.json'\n"
        "  - name: remote_docs\n"
        "    path: gs://unrelated-bucket/raw\n"
        "    file_pattern: '*.json'\n"
    )
    models = project / "models"
    models.mkdir()
    (models / "models.yml").write_text(
        "version: 2\n"
        "models:\n"
        "  - name: local_raw\n"
        "    source: ref('local_docs')\n"
        "    extraction:\n"
        "      backend: json\n"
        "      options:\n"
        "        fields: [id, value]\n"
        "    materialization: incremental\n"
        "  - name: local_downstream\n"
        "    depends_on: [ref('local_raw')]\n"
        "    transform:\n"
        "      type: python\n"
        "      module: transforms.passthrough\n"
        "    materialization: full\n"
        "  - name: remote_raw\n"
        "    source: ref('remote_docs')\n"
        "    extraction:\n"
        "      backend: json\n"
        "      options:\n"
        "        fields: [id, value]\n"
        "    materialization: incremental\n"
    )
    transforms = project / "transforms"
    transforms.mkdir()
    (transforms / "passthrough.py").write_text(
        "from __future__ import annotations\n\n"
        "import polars as pl\n\n"
        "def run(deps: dict[str, pl.DataFrame]) -> pl.DataFrame:\n"
        "    return deps['local_raw'].select('id', 'value')\n"
    )
    local_data = project / "data" / "local"
    local_data.mkdir(parents=True)
    (local_data / "one.json").write_text('{"id": "local-1", "value": 42}')
    return project


@pytest.fixture
def forbidden_gcs_client(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    calls: list[str | None] = []

    def _forbidden(self: GCSDocumentSource, project: str | None = None) -> None:
        calls.append(project)
        raise AssertionError("unrelated GCS client was constructed")

    monkeypatch.setattr(GCSDocumentSource, "_make_client", _forbidden)
    return calls


def test_run_select_does_not_touch_unrelated_gcs_and_records_sources(
    mixed_source_project: Path,
    forbidden_gcs_client: list[str | None],
) -> None:
    results = run_project(mixed_source_project, select="local_raw")

    assert [result.model_name for result in results] == ["local_raw"]
    assert results[0].rows_written == 1
    assert forbidden_gcs_client == []

    payload = json.loads(write_run_results(mixed_source_project, results).read_text())
    assert payload["metadata"]["sources_considered"] == ["local_docs"]


def test_downstream_selection_discovers_exact_source_ancestors(
    mixed_source_project: Path,
    forbidden_gcs_client: list[str | None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovered: list[str] = []
    original_discover = LocalDocumentSource.discover

    def _tracked_discover(
        self: LocalDocumentSource,
        source: SourceConfig,
        project_dir: Path,
        *,
        source_filter: Sequence[str] = (),
    ) -> list[DocumentRef]:
        discovered.append(source.name)
        return original_discover(self, source, project_dir, source_filter=source_filter)

    monkeypatch.setattr(LocalDocumentSource, "discover", _tracked_discover)
    run_project(mixed_source_project, select="local_raw")
    discovered.clear()

    results = run_project(mixed_source_project, select="local_downstream")

    assert [result.model_name for result in results] == ["local_downstream"]
    assert discovered == ["local_docs"]
    assert forbidden_gcs_client == []


def test_build_exclude_prevents_excluded_branch_source_discovery(
    mixed_source_project: Path,
    forbidden_gcs_client: list[str | None],
) -> None:
    result = build_project(mixed_source_project, exclude="remote_raw")

    assert [run.model_name for run in result.run_results] == [
        "local_raw",
        "local_downstream",
    ]
    assert not any(run.errors for run in result.run_results)
    assert forbidden_gcs_client == []


def test_watch_observes_only_selected_graph_source_paths(
    mixed_source_project: Path,
    forbidden_gcs_client: list[str | None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import watchfiles

    watched: list[tuple[Path, ...]] = []

    def _watch(*paths: Path, **kwargs: object) -> tuple[()]:
        watched.append(paths)
        return ()

    monkeypatch.setattr(watchfiles, "watch", _watch)
    result = CliRunner().invoke(
        cli,
        [
            "--project-dir",
            str(mixed_source_project),
            "run",
            "--watch",
            "--select",
            "local_raw+",
        ],
    )

    assert result.exit_code == 0, result.output
    assert watched == [((mixed_source_project / "data" / "local").resolve(),)]
    assert forbidden_gcs_client == []


def test_watch_with_empty_tag_selection_is_a_successful_noop(
    mixed_source_project: Path,
    forbidden_gcs_client: list[str | None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import watchfiles

    watch_called = False

    def _watch(*paths: Path, **kwargs: object) -> tuple[()]:
        nonlocal watch_called
        watch_called = True
        return ()

    monkeypatch.setattr(watchfiles, "watch", _watch)
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "--project-dir",
            str(mixed_source_project),
            "--watch",
            "--select",
            "tag:not_present",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "No models selected." in result.output
    assert not watch_called
    assert forbidden_gcs_client == []
