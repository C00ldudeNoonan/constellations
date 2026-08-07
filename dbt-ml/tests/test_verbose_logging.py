"""Verbose progress feedback for long-running builds (issue #268)."""
from __future__ import annotations

import io
import logging
import os
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from dbt_ml import progress as progress_module
from dbt_ml.cli import cli
from dbt_ml.logging_setup import configure_verbose_logging, resolve_verbosity
from dbt_ml.progress import (
    _NullReporter,
    _TerminalReporter,
    configure_progress,
    get_reporter,
    set_reporter,
)
from dbt_ml.synth import generate_invoices


@pytest.fixture(autouse=True)
def _reset_dbt_ml_logger():
    """Every test starts with a clean handler set so ordering is deterministic."""
    yield
    configure_verbose_logging(0)
    set_reporter(None)


def _copy_example(tmp_path: Path, example_project_dir: Path) -> Path:
    dst = tmp_path / "project"
    shutil.copytree(
        example_project_dir,
        dst,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    return dst


def test_resolve_verbosity_prefers_cli_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DBT_ML_VERBOSE", "2")
    assert resolve_verbosity(1) == 1


def test_resolve_verbosity_reads_env_when_no_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DBT_ML_VERBOSE", "2")
    assert resolve_verbosity(0) == 2


def test_resolve_verbosity_unset_env_is_quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DBT_ML_VERBOSE", raising=False)
    assert resolve_verbosity(0) == 0


def test_resolve_verbosity_non_integer_env_defaults_to_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DBT_ML_VERBOSE", "yes")
    assert resolve_verbosity(0) == 1


def test_configure_verbose_logging_is_idempotent() -> None:
    logger = logging.getLogger("dbt_ml")
    configure_verbose_logging(1)
    handler_after_first = getattr(logger, "_dbt_ml_verbose_handler", None)
    assert handler_after_first is not None
    configure_verbose_logging(1)
    handler_after_second = getattr(logger, "_dbt_ml_verbose_handler", None)
    assert handler_after_second is not None
    # The stored handler is replaced, not appended, so we never stack duplicates.
    handlers_on_logger = [
        h for h in logger.handlers if h is handler_after_first
    ]
    assert not handlers_on_logger


def test_configure_verbose_logging_zero_removes_handler() -> None:
    logger = logging.getLogger("dbt_ml")
    configure_verbose_logging(1)
    assert getattr(logger, "_dbt_ml_verbose_handler", None) is not None
    configure_verbose_logging(0)
    assert getattr(logger, "_dbt_ml_verbose_handler", None) is None


def test_verbose_logging_debug_level_at_double_v() -> None:
    logger = logging.getLogger("dbt_ml")
    configure_verbose_logging(2)
    assert logger.level == logging.DEBUG


def test_configure_progress_non_tty_stays_null() -> None:
    buffer = io.StringIO()  # not a TTY
    configure_progress(1, stream=buffer)
    assert isinstance(get_reporter(), _NullReporter)


def test_configure_progress_tty_installs_terminal_reporter() -> None:
    class _FakeTTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    stream = _FakeTTY()
    configure_progress(1, stream=stream)
    assert isinstance(get_reporter(), _TerminalReporter)


def test_terminal_reporter_emits_source_and_finish_lines() -> None:
    class _FakeTTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    stream = _FakeTTY()
    reporter = _TerminalReporter(stream)
    reporter.source_discovered("raw_invoices", 12345)
    reporter.model_finished("raw_invoices", 12345, 3.5, None)
    text = stream.getvalue()
    assert "raw_invoices" in text
    assert "12,345" in text


def test_null_reporter_task_advances_no_op() -> None:
    task = _NullReporter().model_task("m", "extraction", 5)
    with task:
        task.advance(3)  # must not raise


def test_verbose_build_emits_progress_and_summary_to_stderr(
    tmp_path: Path, example_project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DBT_ML_VERBOSE", raising=False)
    dst = _copy_example(tmp_path, example_project_dir)
    generate_invoices(3, dst / "data" / "invoices", seed=1)

    runner = CliRunner()
    result = runner.invoke(
        cli, ["--project-dir", str(dst), "run", "-v"]
    )
    assert result.exit_code == 0, (result.stdout, result.stderr)
    # Discovery + per-model markers land on stderr; the run summary keeps
    # stdout clean for the results table.
    assert "raw_invoices" in result.stderr
    assert "starting" in result.stderr
    assert "finished" in result.stderr


def test_dbt_ml_verbose_env_var_enables_logging_without_flag(
    tmp_path: Path, example_project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DBT_ML_VERBOSE", "1")
    dst = _copy_example(tmp_path, example_project_dir)
    generate_invoices(2, dst / "data" / "invoices", seed=1)

    runner = CliRunner()
    result = runner.invoke(cli, ["--project-dir", str(dst), "run"])
    assert result.exit_code == 0, (result.stdout, result.stderr)
    assert "starting" in result.stderr


def test_default_run_stays_silent_on_stderr(
    tmp_path: Path, example_project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DBT_ML_VERBOSE", raising=False)
    dst = _copy_example(tmp_path, example_project_dir)
    generate_invoices(2, dst / "data" / "invoices", seed=1)

    runner = CliRunner()
    result = runner.invoke(cli, ["--project-dir", str(dst), "run"])
    assert result.exit_code == 0, (result.stdout, result.stderr)
    # No progress lines from dbt-ml when verbose is off — a bare "starting"
    # or per-model discovery line would prove the default output regressed.
    assert "starting" not in result.stderr
    assert "discovered" not in result.stderr


# Silence unused-import warnings from linting: os/progress_module are used to
# hint the reader that these fixtures reach into module state.
_ = (os, progress_module)
