"""Verbose progress feedback for long-running builds (issue #268)."""
from __future__ import annotations

import io
import json
import logging
import os
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from stel import progress as progress_module
from stel.cli import _configure_output, cli
from stel.logging_setup import (
    REPORTER_ECHO_EXTRA,
    _ReporterHandler,
    configure_verbose_logging,
    resolve_verbosity,
)
from stel.progress import (
    _MAX_DEFERRED_LINES,
    OutputLevel,
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


def test_configure_progress_non_tty_installs_ledger_without_bars() -> None:
    """The ledger is not TTY-gated — a CI log wants it too (issue #404) — but
    bars are, since a carriage-return redraw is noise in a captured stream."""
    buffer = io.StringIO()  # not a TTY
    bars = configure_progress(OutputLevel.VERBOSE, stream=buffer)
    assert bars is False
    assert isinstance(get_reporter(), _TerminalReporter)


def test_configure_progress_tty_verbose_enables_bars() -> None:
    class _FakeTTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    stream = _FakeTTY()
    bars = configure_progress(OutputLevel.VERBOSE, stream=stream)
    assert bars is True
    assert isinstance(get_reporter(), _TerminalReporter)


def test_configure_progress_normal_has_no_bars_even_on_a_tty() -> None:
    """The default level is a ledger, not a progress bar: bars are detail that
    `-v` buys."""
    class _FakeTTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    stream = _FakeTTY()
    bars = configure_progress(OutputLevel.NORMAL, stream=stream)
    assert bars is False
    assert isinstance(get_reporter(), _TerminalReporter)


def test_configure_progress_quiet_stays_null() -> None:
    class _FakeTTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    stream = _FakeTTY()
    assert configure_progress(OutputLevel.QUIET, stream=stream) is False
    assert isinstance(get_reporter(), _NullReporter)


def test_terminal_reporter_emits_source_and_finish_lines() -> None:
    class _FakeTTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    stream = _FakeTTY()
    reporter = _TerminalReporter(stream, level=OutputLevel.VERBOSE, bars=True)
    reporter.source_discovered("raw_invoices", 12345)
    reporter.model_finished(
        "raw_invoices", "extraction", 12345, 3.5, None, failed=False
    )
    text = stream.getvalue()
    assert "raw_invoices" in text
    assert "12,345" in text


def test_ledger_renders_header_counter_outcomes_and_footer() -> None:
    """The whole default-mode shape in one place: header, a numbered line per
    model with its outcome, and a footer that agrees with them."""
    stream = io.StringIO()
    reporter = _TerminalReporter(stream, level=OutputLevel.NORMAL, bars=False)
    reporter.run_started(3, target="dev", warehouse="duckdb", project_total=7)
    reporter.model_finished("raw", "extraction", 1204, 12.4, None, failed=False)
    reporter.model_finished("summary", "transform", 0, 0.2, None, failed=True)
    reporter.model_skipped("totals", "upstream failed")
    reporter.run_finished(ok=1, errored=1, skipped=1)
    lines = stream.getvalue().splitlines()

    # "3 of 7" only because a selector narrowed the run.
    assert lines[0] == "Running 3 of 7 models (target: dev, duckdb)"
    assert lines[1].startswith("[1/3] raw") and lines[1].endswith("OK")
    assert lines[2].startswith("[2/3] summary") and lines[2].endswith("ERROR")
    assert lines[3].startswith("[3/3] totals")
    assert "SKIPPED (upstream failed)" in lines[3]
    assert "1,204 rows" in lines[1]
    assert lines[4].startswith("Completed in ")
    assert lines[4].endswith(": 1 ok, 1 error, 1 skipped")


def test_ledger_header_omits_the_total_when_nothing_was_narrowed() -> None:
    """A bare `Running 7 models` — "7 of 7" reads as though something had been
    dropped."""
    stream = io.StringIO()
    reporter = _TerminalReporter(stream, level=OutputLevel.NORMAL, bars=False)
    reporter.run_started(7, target="prod", warehouse="bigquery", project_total=7)
    assert stream.getvalue().startswith("Running 7 models (target: prod, bigquery)")


def test_ledger_status_beats_the_failed_flag() -> None:
    """A budget-exceeded model is neither plain OK nor a bare ERROR; the status
    the runner set is what the operator needs to see."""
    stream = io.StringIO()
    reporter = _TerminalReporter(stream, level=OutputLevel.NORMAL, bars=False)
    reporter.run_started(1, target="dev", warehouse="duckdb", project_total=1)
    reporter.model_finished(
        "m", "llm", 0, 1.0, "budget_exceeded", failed=True
    )
    assert "BUDGET_EXCEEDED" in stream.getvalue()


def test_normal_level_suppresses_verbose_only_detail() -> None:
    """`[source]`, `[publish]` and bars are what `-v` buys; the default level
    must render none of them."""
    stream = io.StringIO()
    reporter = _TerminalReporter(stream, level=OutputLevel.NORMAL, bars=False)
    reporter.source_discovered("s", 10)
    reporter.publication("job_id=1")
    with reporter.model_task("m", "extraction", 5) as task:
        task.advance(5)
    assert stream.getvalue() == ""


def test_terminal_reporter_defers_publication_during_active_bar() -> None:
    # issue #292 review: publication telemetry fires per flush inside the live
    # progress bar; echoing then would smear it. It must be buffered while the
    # bar is active and flushed once the task exits.
    class _FakeTTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    stream = _FakeTTY()
    reporter = _TerminalReporter(stream, level=OutputLevel.VERBOSE, bars=True)
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
    reporter = _TerminalReporter(stream, level=OutputLevel.VERBOSE, bars=True)
    reporter.publication("job_id=9")
    assert "[publish] job_id=9" in stream.getvalue()


def test_null_reporter_task_advances_no_op() -> None:
    task = _NullReporter().model_task("m", "extraction", 5)
    with task:
        task.advance(3)  # must not raise


def test_configure_output_tty_routes_log_through_reporter(
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
    _configure_output(1)
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
    _configure_output(1)
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
    _configure_output(1)
    log = logging.getLogger("stel.runner")
    log.info("finished raw_invoices", extra=REPORTER_ECHO_EXTRA)
    assert "finished raw_invoices" not in stream.getvalue()
    # The reporter's own callback is what renders it.
    get_reporter().model_finished(
        "raw_invoices", "extraction", 3, 1.0, None, failed=False
    )
    assert "raw_invoices" in stream.getvalue()


def test_reporter_echo_records_survive_when_nothing_renders_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no reporter installed (`-v --json`) the marked record is the event's
    only copy, so the filter must let it through. The check is on the live
    reporter rather than on which handler is attached, because the reporter can
    be swapped after install."""
    stream = io.StringIO()
    monkeypatch.setattr("sys.stderr", stream)
    _configure_output(1, json_output=True)
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
    reporter = _TerminalReporter(stream, level=OutputLevel.VERBOSE, bars=True)
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
    reporter = _TerminalReporter(stream, level=OutputLevel.VERBOSE, bars=True)
    overflow = 5
    with reporter.model_task("m", "extraction", 3):
        for i in range(_MAX_DEFERRED_LINES + overflow):
            reporter.detail(f"line-{i}")
    text = stream.getvalue()
    assert f"{overflow} earlier detail line(s) dropped" in text
    assert "line-0" not in text
    assert f"line-{_MAX_DEFERRED_LINES + overflow - 1}" in text


def test_configure_output_non_tty_uses_the_plain_log_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A captured run gets the ledger and a plain stderr handler — not the
    reporter-routed one, since with no bar there is nothing to defer past."""
    monkeypatch.setattr("sys.stderr", io.StringIO())
    _configure_output(1)
    logger = logging.getLogger("stel")
    handler = getattr(logger, "_stel_verbose_handler", None)
    assert isinstance(handler, logging.StreamHandler)
    assert not isinstance(handler, _ReporterHandler)
    assert isinstance(get_reporter(), _TerminalReporter)


def test_configure_output_default_installs_ledger_but_no_log_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default is the ledger, not silence (issue #404) and not the log
    channel — `-v` is still what buys the INFO lines."""
    monkeypatch.setattr("sys.stderr", io.StringIO())
    _configure_output(0)
    logger = logging.getLogger("stel")
    assert getattr(logger, "_stel_verbose_handler", None) is None
    assert isinstance(get_reporter(), _TerminalReporter)


def test_configure_output_json_is_quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--json` is the machine path: stdout carries the payload and stderr says
    nothing, so no reporter is installed."""
    monkeypatch.setattr("sys.stderr", io.StringIO())
    _configure_output(0, json_output=True)
    assert isinstance(get_reporter(), _NullReporter)


def test_configure_output_json_with_verbose_keeps_the_log_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`-v --json` still narrates on stderr — the quiet default is about the
    ledger, not about suppressing an explicit request for detail."""
    monkeypatch.setattr("sys.stderr", io.StringIO())
    _configure_output(1, json_output=True)
    logger = logging.getLogger("stel")
    assert getattr(logger, "_stel_verbose_handler", None) is not None
    assert isinstance(get_reporter(), _NullReporter)


def test_configure_output_bars_unsafe_forces_log_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`run --threads N` over multiple models runs them concurrently, each with
    its own progress bar on one stderr. bars_safe=False must fall back to the
    (atomic, interleave-safe) log handler even on a TTY, and leave no reporter."""
    class _FakeTTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr("sys.stderr", _FakeTTY())
    _configure_output(1, bars_safe=False)
    logger = logging.getLogger("stel")
    handler = getattr(logger, "_stel_verbose_handler", None)
    assert isinstance(handler, logging.StreamHandler)
    assert not isinstance(handler, _ReporterHandler)
    # The ledger survives — it is the bars that are unsafe here, not the
    # per-model lines, which are emitted whole under the logging lock.
    assert isinstance(get_reporter(), _TerminalReporter)


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
    # 2 of 5 match the glob; the selected line must show the filtered count,
    # whichever channel renders it (the reporter here, since it is installed).
    # The backend's own "discovered 5" line is the pre-filter count by design
    # (issue #348) and legitimately sits alongside it.
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
    # Discovery, per-model markers and the ledger land on stderr; the run
    # summary keeps stdout clean for the results table.
    err = result.stderr
    assert "raw_invoices" in err
    assert "starting" in err  # log channel: no reporter equivalent
    assert "[source] vendor_invoices" in err  # reporter channel
    assert "Completed" in err  # ledger footer
    # Model completion is rendered by the reporter, so the log record carrying
    # the same event is dropped — one line per event, not one per channel.
    assert "finished raw_invoices" not in err


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

    # Reporter-only rendering: no log call produces these.
    assert "[source] vendor_invoices" in err
    assert "Completed" in err
    # Log-only: `Source '...': scanning <dir>` has no reporter callback, and was
    # invisible on a terminal before this fix.
    assert "scanning" in err
    assert "starting raw_invoices" in err
    # Both channels know about model completion; it must render once — as the
    # ledger line, with the log record dropped.
    assert "finished raw_invoices" not in err
    ledger = [ln for ln in err.splitlines() if ln.startswith("[1/")]
    assert len(ledger) == 1 and "raw_invoices" in ledger[0]
    # The bar's own label is separate from the ledger line, not a duplicate.
    assert "[extraction] raw_invoices" in err
    # Likewise source selection: the reporter's line, not the log's.
    assert err.count("document(s) selected") == 1


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


def test_default_run_streams_a_ledger_and_keeps_stdout_for_the_table(
    tmp_path: Path, example_project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The #404 behavior change: a default run narrates as it goes instead of
    printing nothing until it returns."""
    monkeypatch.delenv("STEL_VERBOSE", raising=False)
    dst = _copy_example(tmp_path, example_project_dir)
    generate_invoices(2, dst / "data" / "invoices", seed=1)

    runner = CliRunner()
    result = runner.invoke(cli, ["--project-dir", str(dst), "run"])
    assert result.exit_code == 0, (result.stdout, result.stderr)
    err = result.stderr
    assert "Running " in err and "target: dev" in err
    assert "raw_invoices" in err
    assert "[1/" in err
    assert "Completed" in err and "ok" in err
    # Detail still costs `-v`: no INFO log lines, no discovery lines, no bars.
    assert "starting" not in err
    assert "[source]" not in err
    # The summary table stays on stdout, unmoved.
    assert "model" in result.stdout and "rows" in result.stdout


def test_default_run_ledger_reports_each_model_once(
    tmp_path: Path, example_project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("STEL_VERBOSE", raising=False)
    dst = _copy_example(tmp_path, example_project_dir)
    generate_invoices(2, dst / "data" / "invoices", seed=1)

    runner = CliRunner()
    result = runner.invoke(cli, ["--project-dir", str(dst), "run"])
    assert result.exit_code == 0, (result.stdout, result.stderr)
    err = result.stderr
    ledger = [ln for ln in err.splitlines() if ln.startswith("[")]
    # One line per selected model, numbered from 1 without gaps or repeats.
    assert len(ledger) == len({ln.split("]")[0] for ln in ledger})
    assert ledger[0].startswith("[1/")
    # Nothing but ledger lines carries a bracket prefix at the default level —
    # per-flush [publish] telemetry is detail that `-v` buys.
    assert "[publish]" not in err


def test_build_ledger_marks_skipped_models_and_the_footer_agrees(
    tmp_path: Path, example_project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model whose upstream errored gets its own ledger line rather than
    silently vanishing between the header and the footer."""
    monkeypatch.delenv("STEL_VERBOSE", raising=False)
    dst = _copy_example(tmp_path, example_project_dir)
    generate_invoices(2, dst / "data" / "invoices", seed=1)
    # Both transforms depend on raw_invoices directly in the example, so chain
    # them first — otherwise a failure blocks nothing and there is no skip to
    # assert on.
    totals_yml = dst / "models" / "monthly_totals.yml"
    totals_yml.write_text(
        totals_yml.read_text(encoding="utf-8").replace(
            "ref('raw_invoices')", "ref('invoice_summary')"
        ),
        encoding="utf-8",
    )
    # Now make the middle model raise, so the descendant is blocked.
    (dst / "transforms" / "summarize.py").write_text(
        '''import polars as pl


def run(deps: dict[str, pl.DataFrame], ctx: object = None) -> pl.DataFrame:
    raise RuntimeError("deliberate failure")
''',
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["--project-dir", str(dst), "build"])
    err = result.stderr
    assert result.exit_code == 1, (result.stdout, err)
    assert "Running 3 models" in err
    assert "invoice_summary" in err and "ERROR" in err
    assert "monthly_totals" in err and "SKIPPED (upstream failed)" in err
    # The footer agrees with the lines above it. Not the last stderr line —
    # the CLI's per-model ERROR detail follows the ledger.
    footer = next(ln for ln in err.splitlines() if ln.startswith("Completed"))
    assert footer.endswith("1 ok, 1 error, 1 skipped")


def test_json_run_says_nothing_on_stderr(
    tmp_path: Path, example_project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--json` keeps the machine contract: one payload on stdout, and the
    ledger suppressed rather than merely redirected."""
    monkeypatch.delenv("STEL_VERBOSE", raising=False)
    dst = _copy_example(tmp_path, example_project_dir)
    generate_invoices(2, dst / "data" / "invoices", seed=1)

    runner = CliRunner()
    result = runner.invoke(cli, ["--project-dir", str(dst), "run", "--json"])
    assert result.exit_code == 0, (result.stdout, result.stderr)
    assert result.stderr.strip() == ""
    json.loads(result.stdout)


# Silence unused-import warnings from linting: os/progress_module are used to
# hint the reader that these fixtures reach into module state.
_ = (os, progress_module)
