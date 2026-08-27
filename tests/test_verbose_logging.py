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
from stel.logging_setup import (
    REPORTER_ECHO_EXTRA,
    _ReporterHandler,
    configure_verbose_logging,
    resolve_verbosity,
)
from stel.progress import (
    _MAX_DEFERRED_LINES,
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


def test_enable_verbose_output_tty_routes_log_through_reporter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a TTY the reporter owns the terminal, but the log handler still
    attaches — routed through the reporter so records defer past a live bar
    instead of smearing it (issue #403). Before that fix the handler was torn
    down here, so every log.info the reporter does not itself render was lost
    on a terminal and visible only when stderr was redirected."""
    class _FakeTTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr("sys.stderr", _FakeTTY())
    _enable_verbose_output(1)
    logger = logging.getLogger("stel")
    handler = getattr(logger, "_stel_verbose_handler", None)
    assert isinstance(handler, _ReporterHandler)
    assert isinstance(get_reporter(), _TerminalReporter)


def test_tty_verbose_forwards_plain_log_records_to_the_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regression #403 names: a provider batch poll or a source-scan line
    has no reporter callback, so on a TTY it used to vanish entirely."""
    class _FakeTTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    stream = _FakeTTY()
    monkeypatch.setattr("sys.stderr", stream)
    _enable_verbose_output(1)
    logging.getLogger("stel.providers.anthropic").info(
        "batch %s: in_progress (processing=%d)", "batch_abc", 42
    )
    assert "batch_abc" in stream.getvalue()
    assert "processing=42" in stream.getvalue()


def test_tty_verbose_drops_records_the_reporter_already_renders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both channels are live on a TTY, so an event with a reporter callback
    must print once. The marker travels on the record, not at the call site:
    the emitting module does not know which channel is installed."""
    class _FakeTTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    stream = _FakeTTY()
    monkeypatch.setattr("sys.stderr", stream)
    _enable_verbose_output(1)
    log = logging.getLogger("stel.runner")
    log.info("finished raw_invoices", extra=REPORTER_ECHO_EXTRA)
    assert "finished raw_invoices" not in stream.getvalue()
    # The reporter's own callback is what renders it.
    get_reporter().model_finished("raw_invoices", 3, 1.0, None)
    assert "raw_invoices" in stream.getvalue()


def test_non_tty_verbose_keeps_reporter_echo_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A captured run has no reporter, so the marked record is the only copy of
    the event and must survive."""
    stream = io.StringIO()
    monkeypatch.setattr("sys.stderr", stream)
    _enable_verbose_output(1)
    logging.getLogger("stel.runner").info(
        "finished raw_invoices", extra=REPORTER_ECHO_EXTRA
    )
    assert "finished raw_invoices" in stream.getvalue()


def test_forwarded_log_records_defer_past_a_live_bar() -> None:
    """A record arriving mid-bar must not be written straight to the stream —
    click.echo does not clear/redraw the bar, so it would smear."""
    class _FakeTTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    stream = _FakeTTY()
    reporter = _TerminalReporter(stream)
    task = reporter.model_task("m", "extraction", 3)
    with task:
        reporter.detail("polled batch batch_abc")
        assert "batch_abc" not in stream.getvalue()
    assert "polled batch batch_abc" in stream.getvalue()


def test_deferred_detail_lines_are_capped_and_overflow_is_reported() -> None:
    """An hours-long bar must not hold every line it produced. Oldest lines are
    dropped, and the drop is stated rather than silently swallowed."""
    class _FakeTTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    stream = _FakeTTY()
    reporter = _TerminalReporter(stream)
    overflow = 5
    with reporter.model_task("m", "extraction", 3):
        for i in range(_MAX_DEFERRED_LINES + overflow):
            reporter.detail(f"line-{i}")
    text = stream.getvalue()
    assert f"{overflow} earlier detail line(s) dropped" in text
    assert "line-0" not in text
    assert f"line-{_MAX_DEFERRED_LINES + overflow - 1}" in text


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


def test_tty_verbose_run_shows_both_reporter_and_log_lines_once(
    tmp_path: Path, example_project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end on the reporter channel (issue #403). Forcing the TTY branch
    rather than faking sys.stderr, because CliRunner owns that stream.

    Three things at once: a line only the reporter renders, a line only the log
    channel carries, and an event both know about appearing exactly once."""
    monkeypatch.delenv("STEL_VERBOSE", raising=False)
    monkeypatch.setattr("stel.progress._is_tty", lambda _stream: True)
    dst = _copy_example(tmp_path, example_project_dir)
    generate_invoices(3, dst / "data" / "invoices", seed=1)

    runner = CliRunner()
    result = runner.invoke(cli, ["--project-dir", str(dst), "run", "-v"])
    assert result.exit_code == 0, (result.stdout, result.stderr)
    err = result.stderr

    # Reporter-only rendering: no log call produces these prefixes.
    assert "[source]" in err
    assert "[done]" in err
    # Log-only: `Source '...': scanning <dir>` has no reporter callback, and was
    # invisible on a terminal before this fix.
    assert "scanning" in err
    assert "starting raw_invoices" in err
    # Both channels know about model completion; it must render once.
    assert err.count("raw_invoices") >= 1
    assert "finished raw_invoices" not in err  # suppressed in favor of [done]
    assert err.count("[done]   raw_invoices") == 1
    # Likewise source selection: the reporter's [source] line, not the log's.
    assert "document(s) selected" not in err
    assert err.count("[source] vendor_invoices") == 1


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
