"""CLI-services seam (issue #190, Workstream D).

The logic behind the Click commands is now importable without invoking the CLI.
These tests exercise the shared bootstrap and the watch service directly —
coverage that previously required spawning the `run --watch` command — and pin
that `cli.py` still re-exports the moved symbols so command call sites and
existing imports are unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import dbt_ml.cli as cli
from dbt_ml.cli_services.context import (
    CONFIG_ERRORS,
    ConfigClickError,
    build_dag_or_click,
    load_project_or_click,
)
from dbt_ml.cli_services.watch import run_watch


def test_cli_reexports_point_at_services() -> None:
    assert cli._run_watch is run_watch
    assert cli._load is load_project_or_click
    assert cli._build_dag is build_dag_or_click
    assert cli.ConfigClickError is ConfigClickError
    assert cli._CONFIG_ERRORS is CONFIG_ERRORS


def test_config_click_error_exits_two() -> None:
    # The exit-code contract (#87): setup failures exit 2, distinct from a run
    # that started and had a model fail (exit 1).
    assert ConfigClickError.exit_code == 2


def test_load_project_or_click_wraps_config_error(tmp_path: Path) -> None:
    # An empty dir has no project file; the bootstrap surfaces it as the
    # exit-2 error rather than a raw ConfigError.
    with pytest.raises(ConfigClickError):
        load_project_or_click(tmp_path)


def test_watch_reports_no_source_paths(
    tmp_path: Path, example_project_dir: Path
) -> None:
    import shutil

    project = tmp_path / "proj"
    shutil.copytree(
        example_project_dir,
        project,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    # No source files exist on disk yet, so the watch service refuses to start
    # its loop — reachable now without invoking the CLI or watchfiles.
    with pytest.raises(Exception, match="No source paths exist on disk"):
        run_watch(
            project,
            profiles_dir=None,
            target=None,
            full_refresh=False,
            select=None,
            exclude=None,
        )
