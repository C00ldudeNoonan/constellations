"""A sanitized failure's native detail reaches the file the operator named (issue #590).

`_operation_failed` keeps the operation, the step and the native exception's
type and drops the text; the full exception goes to a DEBUG record that `-v`
never enables. That left a CLI operator with no path to the cause. These tests
pin the replacement: `--diagnostics-file` writes the records carrying an
exception, and warnings, to one file and nowhere else, and every other channel
stays as sanitized as before (ADR-0012).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from stel.logging_setup import (
    configure_diagnostics_file,
    configure_verbose_logging,
    diagnostics_file_written,
    resolve_diagnostics_file,
)
from stel.retrieval import RetrievalError
from stel.retrieval.lancedb import _operation_failed

# Shaped like what LanceDB quotes back: an object-store URI carrying a
# credential-looking query string. It may reach the file and nothing else.
SENTINEL = "gs://distinctive-bucket/prefix?token=distinctive-native-secret"


def _native_error() -> RuntimeError:
    """A raised-and-caught error, so it carries a traceback like a real one."""
    try:
        raise RuntimeError(SENTINEL)
    except RuntimeError as error:
        return error


def _stel_logger() -> logging.Logger:
    return logging.getLogger("stel")


def test_resolve_prefers_the_flag_over_the_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("STEL_DIAGNOSTICS_FILE", str(tmp_path / "from-env.log"))
    assert resolve_diagnostics_file(tmp_path / "flag.log") == tmp_path / "flag.log"


def test_resolve_reads_the_env_when_no_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("STEL_DIAGNOSTICS_FILE", str(tmp_path / "from-env.log"))
    assert resolve_diagnostics_file(None) == tmp_path / "from-env.log"


def test_resolve_treats_an_unset_or_empty_env_as_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STEL_DIAGNOSTICS_FILE", raising=False)
    assert resolve_diagnostics_file(None) is None
    # `docker compose` forwards an unset variable as the empty string.
    monkeypatch.setenv("STEL_DIAGNOSTICS_FILE", "  ")
    assert resolve_diagnostics_file(None) is None


def test_records_carrying_an_exception_and_warnings_reach_the_file_and_nothing_else(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "diagnostics.log"
    configure_diagnostics_file(path)
    log = logging.getLogger("stel.retrieval.lancedb")

    log.debug("plain debug line")
    log.info("progress line")
    log.warning("retrying the index build")
    log.debug("native cause", exc_info=_native_error())

    text = path.read_text(encoding="utf-8")
    assert SENTINEL in text
    assert "Traceback (most recent call last)" in text
    assert "retrying the index build" in text
    assert "plain debug line" not in text
    assert "progress line" not in text
    # Without `-v` the warning still reaches stderr as it did through the
    # stdlib fallback, and the native text does not.
    err = capsys.readouterr().err
    assert "retrying the index build" in err
    assert SENTINEL not in err


def test_the_file_is_created_on_first_write_and_readable_by_its_owner_only(
    tmp_path: Path,
) -> None:
    path = tmp_path / "diagnostics.log"
    configure_diagnostics_file(path)
    assert not path.exists(), "a run that fails nowhere must leave no file behind"

    logging.getLogger("stel.execution.transform").debug("boom", exc_info=_native_error())

    assert path.exists()
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("verbose_first", [True, False])
def test_verbose_stays_capped_at_info_while_the_file_takes_debug(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], verbose_first: bool
) -> None:
    """The two channels compose in either order: the logger drops to DEBUG so
    the file can receive the native record, and the `-v` handler keeps its
    INFO level so stderr never sees it."""
    path = tmp_path / "diagnostics.log"
    if verbose_first:
        configure_verbose_logging(1)
        configure_diagnostics_file(path)
    else:
        configure_diagnostics_file(path)
        configure_verbose_logging(1)
    logger = _stel_logger()
    assert logger.level == logging.DEBUG
    verbose = getattr(logger, "_stel_verbose_handler", None)
    assert isinstance(verbose, logging.Handler)
    assert verbose.level == logging.INFO
    # `-v` owns stderr now, so no second handler stands in for the fallback.
    assert getattr(logger, "_stel_fallback_handler", None) is None

    logging.getLogger("stel.retrieval.lancedb").debug("native", exc_info=_native_error())

    assert SENTINEL in path.read_text(encoding="utf-8")
    assert SENTINEL not in capsys.readouterr().err


def test_nothing_propagates_to_a_parent_handler_while_the_file_is_configured(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The file is the sole disclosure surface: a DEBUG record that escaped to a
    root handler (an orchestrator's capture, say) would defeat the point.
    `caplog` listens on the root logger, so it is exactly that parent."""
    logger = _stel_logger()
    configure_diagnostics_file(tmp_path / "diagnostics.log")
    assert logger.propagate is False

    log = logging.getLogger("stel.retrieval.lancedb")
    log.debug("native", exc_info=_native_error())
    log.warning("a warning")
    assert caplog.records == []

    configure_diagnostics_file(None)
    assert logger.propagate is True
    assert logger.level == logging.NOTSET
    assert getattr(logger, "_stel_diagnostics_handler", None) is None
    assert getattr(logger, "_stel_fallback_handler", None) is None


def test_configuring_again_replaces_the_handler_rather_than_stacking(tmp_path: Path) -> None:
    logger = _stel_logger()
    configure_diagnostics_file(tmp_path / "first.log")
    first = getattr(logger, "_stel_diagnostics_handler", None)
    configure_diagnostics_file(tmp_path / "second.log")
    second = getattr(logger, "_stel_diagnostics_handler", None)
    assert isinstance(first, logging.Handler)
    assert isinstance(second, logging.Handler)
    assert second is not first
    assert first not in logger.handlers
    assert logger.handlers.count(second) == 1


def test_written_reports_the_path_only_once_a_record_reached_it(tmp_path: Path) -> None:
    path = tmp_path / "diagnostics.log"
    assert diagnostics_file_written() is None
    configure_diagnostics_file(path)
    assert diagnostics_file_written() is None, "configured is not written"
    logging.getLogger("stel.execution.transform").debug("boom", exc_info=_native_error())
    assert diagnostics_file_written() == path.resolve()
    configure_diagnostics_file(None)
    assert diagnostics_file_written() is None


def test_a_sanitized_lancedb_failure_lands_in_the_file_and_not_in_the_error(
    tmp_path: Path,
) -> None:
    """The store's own sanitizer is one of the sites the file exists for: the
    raised error carries the type and the code, the file carries the rest."""
    path = tmp_path / "diagnostics.log"
    configure_diagnostics_file(path)

    failure = _operation_failed("upsert", "lancedb_upsert_failed", _native_error())

    assert isinstance(failure, RetrievalError)
    assert "[RuntimeError] (code=lancedb_upsert_failed)" in str(failure)
    assert "distinctive" not in "".join((str(failure), repr(failure)))
    text = path.read_text(encoding="utf-8")
    assert SENTINEL in text
    assert "LanceDB operation 'upsert' failed" in text
