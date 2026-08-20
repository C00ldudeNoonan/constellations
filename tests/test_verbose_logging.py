"""Verbose progress feedback for long-running builds (issue #268)."""
from __future__ import annotations

import io
import logging
import os
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from stel import progress as progress_module
from stel.cli import _enable_verbose_output, cli
from stel.logging_setup import configure_verbose_logging, resolve_verbosity
from stel.progress import (
    _NullReporter,
    _TerminalReporter,
    configure_progress,
    get_reporter,
    set_reporter,
)
from stel.synth import generate_invoices


@pytest.fixture(autouse=True)
def _reset_stel_logger():
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
    monkeypatch.setenv("STEL_VERBOSE", "0")
    assert resolve_verbosity(1) == 1


def test_resolve_verbosity_caps_repeated_flag_at_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`-vv` (and higher) must not escalate past the safe INFO ceiling; the DEBUG
    log sites carry unsanitized exception text that we deliberately keep off
    the verbose channel."""
    monkeypatch.delenv("STEL_VERBOSE", raising=False)
    assert resolve_verbosity(2) == 1
    assert resolve_verbosity(5) == 1


def test_resolve_verbosity_reads_env_when_no_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STEL_VERBOSE", "1")
    assert resolve_verbosity(0) == 1


def test_resolve_verbosity_caps_env_at_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STEL_VERBOSE", "2")
    assert resolve_verbosity(0) == 1


def test_resolve_verbosity_unset_env_is_quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STEL_VERBOSE", raising=False)
    assert resolve_verbosity(0) == 0


def test_resolve_verbosity_non_integer_env_defaults_to_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STEL_VERBOSE", "yes")
    assert resolve_verbosity(0) == 1


def test_configure_verbose_logging_is_idempotent() -> None:
    logger = logging.getLogger("stel")
    configure_verbose_logging(1)
    handler_after_first = getattr(logger, "_stel_verbose_handler", None)
    assert handler_after_first is not None
    configure_verbose_logging(1)
    handler_after_second = getattr(logger, "_stel_verbose_handler", None)
    assert handler_after_second is not None
    # The stored handler is replaced, not appended, so we never stack duplicates.
    handlers_on_logger = [
        h for h in logger.handlers if h is handler_after_first
    ]
    assert not handlers_on_logger


def test_configure_verbose_logging_zero_removes_handler() -> None:
    logger = logging.getLogger("stel")
    configure_verbose_logging(1)
    assert getattr(logger, "_stel_verbose_handler", None) is not None
    configure_verbose_logging(0)
    assert getattr(logger, "_stel_verbose_handler", None) is None


def test_verbose_logging_caps_at_info_no_debug() -> None:
    """DEBUG must never be enabled via the verbose flag: the debug log sites
    (transform failures, provider errors) carry unsanitized exception text
    that AGENTS.md keeps out of logs. Regression guard against re-adding a
    `-vv` DEBUG path."""
    logger = logging.getLogger("stel")
    configure_verbose_logging(1)
    assert logger.level == logging.INFO
    handler = getattr(logger, "_stel_verbose_handler", None)
    assert handler is not None
    assert handler.level == logging.INFO


def test_configure_progress_non_tty_stays_null() -> None:
    buffer = io.StringIO()  # not a TTY
    installed = configure_progress(1, stream=buffer)
    assert installed is False
    assert isinstance(get_reporter(), _NullReporter)


def test_configure_progress_tty_installs_terminal_reporter() -> None:
    class _FakeTTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    stream = _FakeTTY()
    installed = configure_progress(1, stream=stream)
    assert installed is True
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


def test_terminal_reporter_defers_publication_during_active_bar() -> None:
    # issue #292 review: publication telemetry fires per flush inside the live
    # progress bar; echoing then would smear it. It must be buffered while the
    # bar is active and flushed once the task exits.
    class _FakeTTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    stream = _FakeTTY()
    reporter = _TerminalReporter(stream)
    task = reporter.model_task("m", "extraction", 3)
    with task:
        reporter.publication("job_id=1")
        reporter.publication("job_id=2")
        # Nothing published to the stream while the bar is live.
        assert "[publish]" not in stream.getvalue()
    text = stream.getvalue()
    # Flushed below the completed bar, in order.
    assert "[publish] job_id=1" in text
    assert "[publish] job_id=2" in text
    assert text.index("job_id=1") < text.index("job_id=2")


def test_terminal_reporter_publication_echoes_without_active_bar() -> None:
    class _FakeTTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    stream = _FakeTTY()
    reporter = _TerminalReporter(stream)
    reporter.publication("job_id=9")
    assert "[publish] job_id=9" in stream.getvalue()


def test_null_reporter_task_advances_no_op() -> None:
    task = _NullReporter().model_task("m", "extraction", 5)
    with task:
        task.advance(3)  # must not raise


def test_enable_verbose_output_tty_uses_only_reporter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When stderr is a TTY, the reporter owns output — the INFO log handler
    must not attach or per-flush lines would collide with the progress-bar
    redraws and every discovery/model event would double-print."""
    class _FakeTTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr("sys.stderr", _FakeTTY())
    _enable_verbose_output(1)
    logger = logging.getLogger("stel")
    assert getattr(logger, "_stel_verbose_handler", None) is None
    assert isinstance(get_reporter(), _TerminalReporter)


def test_enable_verbose_output_non_tty_uses_only_log_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.stderr", io.StringIO())
    _enable_verbose_output(1)
    logger = logging.getLogger("stel")
    assert getattr(logger, "_stel_verbose_handler", None) is not None
    assert isinstance(get_reporter(), _NullReporter)


def test_enable_verbose_output_zero_installs_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.stderr", io.StringIO())
    _enable_verbose_output(0)
    logger = logging.getLogger("stel")
    assert getattr(logger, "_stel_verbose_handler", None) is None
    assert isinstance(get_reporter(), _NullReporter)


def test_enable_verbose_output_bars_unsafe_forces_log_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`run --threads N` over multiple models runs them concurrently, each with
    its own progress bar on one stderr. bars_safe=False must fall back to the
    (atomic, interleave-safe) log handler even on a TTY, and leave no reporter."""
    class _FakeTTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr("sys.stderr", _FakeTTY())
    _enable_verbose_output(1, bars_safe=False)
    logger = logging.getLogger("stel")
    assert getattr(logger, "_stel_verbose_handler", None) is not None
    assert isinstance(get_reporter(), _NullReporter)


def test_source_filter_selected_count_is_reported_on_log_channel(
    tmp_path: Path, example_project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-TTY runs must report the post-filter selected count, not the larger
    pre-filter discovery count, so an orchestrated run can't show hundreds
    discovered while processing a filtered handful."""
    monkeypatch.delenv("STEL_VERBOSE", raising=False)
    dst = _copy_example(tmp_path, example_project_dir)
    generate_invoices(5, dst / "data" / "invoices", seed=1)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--project-dir", str(dst), "run", "-v",
            "--source-filter", "invoice_0000[0-1].json",
        ],
    )
    assert result.exit_code == 0, (result.stdout, result.stderr)
    # 2 of 5 match the glob; the selected line must show the filtered count.
    assert "2 document(s) selected" in result.stderr


def test_verbose_build_emits_progress_and_summary_to_stderr(
    tmp_path: Path, example_project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("STEL_VERBOSE", raising=False)
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
    # Under a captured (non-TTY) stderr the log handler is the sole channel,
    # so the terminal reporter's `[done]` / `[source]` echoes must NOT also
    # appear — that would mean both channels ran and every event would
    # double-print.
    assert "[done]" not in result.stderr
    assert "[source]" not in result.stderr


def test_stel_verbose_env_var_enables_logging_without_flag(
    tmp_path: Path, example_project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STEL_VERBOSE", "1")
    dst = _copy_example(tmp_path, example_project_dir)
    generate_invoices(2, dst / "data" / "invoices", seed=1)

    runner = CliRunner()
    result = runner.invoke(cli, ["--project-dir", str(dst), "run"])
    assert result.exit_code == 0, (result.stdout, result.stderr)
    assert "starting" in result.stderr


def test_default_run_stays_silent_on_stderr(
    tmp_path: Path, example_project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("STEL_VERBOSE", raising=False)
    dst = _copy_example(tmp_path, example_project_dir)
    generate_invoices(2, dst / "data" / "invoices", seed=1)

    runner = CliRunner()
    result = runner.invoke(cli, ["--project-dir", str(dst), "run"])
    assert result.exit_code == 0, (result.stdout, result.stderr)
    # No progress lines from stel when verbose is off — a bare "starting"
    # or per-model discovery line would prove the default output regressed.
    assert "starting" not in result.stderr
    assert "discovered" not in result.stderr


# Silence unused-import warnings from linting: os/progress_module are used to
# hint the reader that these fixtures reach into module state.
_ = (os, progress_module)
