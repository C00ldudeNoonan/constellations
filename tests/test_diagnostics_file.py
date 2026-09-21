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
    PROVIDER_DIAGNOSTICS_EXTRA,
    configure_diagnostics_file,
    configure_verbose_logging,
    diagnostics_file_written,
    resolve_diagnostics_file,
)
from stel.providers.base import provider_error_debug_enabled
from stel.retrieval import RetrievalError
from stel.retrieval.lancedb import _operation_failed

# Shaped like what LanceDB quotes back: an object-store URI carrying a
# credential-looking query string. It may reach the file and nothing else.
SENTINEL = "gs://distinctive-bucket/prefix?token=distinctive-native-secret"
# What the provider switch discloses instead: `redacted_exception_text`'s
# allowlist, which names types and stel frames and quotes no native text.
ALLOWLIST = "builtins.RuntimeError\n  at stel.providers.fake:1"


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


@pytest.mark.skipif(os.name != "posix", reason="file modes are a POSIX property")
def test_an_existing_file_with_open_permissions_is_tightened_on_first_write(
    tmp_path: Path,
) -> None:
    """`O_CREAT`'s mode applies only to a file it creates. An orchestrator that
    pre-creates the destination `0644` would otherwise get native detail
    appended to a world-readable file (Codex review on #595)."""
    path = tmp_path / "diagnostics.log"
    path.write_text("", encoding="utf-8")
    path.chmod(0o644)
    configure_diagnostics_file(path)

    logging.getLogger("stel.execution.transform").debug("boom", exc_info=_native_error())

    assert path.stat().st_mode & 0o777 == 0o600


def test_a_destination_that_cannot_be_opened_costs_one_safe_line_and_not_the_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The first open happens while the native exception is being handled. The
    stdlib opens outside its emit guard and its error report prints the active
    exception chain, so an unwritable path would put the native text on stderr
    and replace the sanitized error with an OSError (Codex review on #595)."""
    path = tmp_path / "missing-directory" / "diagnostics.log"
    configure_diagnostics_file(path)
    log = logging.getLogger("stel.retrieval.lancedb")

    try:
        raise RuntimeError(SENTINEL)
    except RuntimeError:
        # Neither the OSError nor anything else may escape the log call.
        log.debug("native", exc_info=True)
        log.debug("native again", exc_info=True)

    err = capsys.readouterr().err
    assert err.count("diagnostics file") == 1, err
    assert "could not be written [FileNotFoundError]" in err
    assert SENTINEL not in err
    assert "Logging error" not in err
    assert not path.exists()
    assert diagnostics_file_written() is None


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


# ─── the provider-error switch's destination (issue #599) ───────────────────


def _provider_call_site(logger: logging.Logger) -> bool:
    """Shaped exactly like the nine guarded sites in `providers/` and
    `backends/llm_backend.py`. Returns whether the guard let it through."""
    if provider_error_debug_enabled() and logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "provider '%s' inference failed:\n%s",
            "fake",
            ALLOWLIST,
            extra=PROVIDER_DIAGNOSTICS_EXTRA,
        )
        return True
    return False


def test_the_provider_switch_alone_reaches_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`STEL_DEBUG_PROVIDER_ERRORS=1` emitted nothing under any combination of
    flags (issue #599): the call sites are gated on a DEBUG level `-v` caps at
    INFO, and the records carry no `exc_info`, so even the diagnostics file
    filtered them out. Its docstring promises local diagnosis, so the variable
    alone has to be enough."""
    monkeypatch.setenv("STEL_DEBUG_PROVIDER_ERRORS", "1")
    configure_verbose_logging(0)

    assert _provider_call_site(logging.getLogger("stel.providers.fake")), (
        "the guard rejected the call: the switch is on, so the logger must be "
        "at DEBUG by the time a provider error is handled"
    )
    assert ALLOWLIST in capsys.readouterr().err


def test_the_provider_switch_does_not_put_native_text_on_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The switch raises the `stel` logger to DEBUG, which is where every
    native-text record in the tree lives. What keeps those off stderr is no
    longer the logger's level but the channel's filter, so it is worth a test
    of its own: this is the security property the level change moves."""
    monkeypatch.setenv("STEL_DEBUG_PROVIDER_ERRORS", "1")
    configure_verbose_logging(0)

    logging.getLogger("stel.execution.transform").debug("boom", exc_info=_native_error())

    assert SENTINEL not in capsys.readouterr().err


def test_the_provider_switch_prefers_the_diagnostics_file_to_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """One destination, not two: with a file configured the allowlist goes
    there, so an operator collecting diagnostics gets one artifact."""
    monkeypatch.setenv("STEL_DEBUG_PROVIDER_ERRORS", "1")
    path = tmp_path / "diagnostics.log"
    configure_verbose_logging(0)
    configure_diagnostics_file(path)

    assert _provider_call_site(logging.getLogger("stel.providers.fake"))

    assert ALLOWLIST not in capsys.readouterr().err
    assert ALLOWLIST in path.read_text(encoding="utf-8")


def test_the_switch_being_off_emits_nothing_even_with_a_diagnostics_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The env var still decides whether the allowlist is produced at all; the
    file only decides where it lands. Off by default is the whole point."""
    path = tmp_path / "diagnostics.log"
    configure_verbose_logging(0)
    configure_diagnostics_file(path)

    assert not _provider_call_site(logging.getLogger("stel.providers.fake"))

    assert ALLOWLIST not in capsys.readouterr().err
    assert not path.exists() or ALLOWLIST not in path.read_text(encoding="utf-8")


def test_warnings_still_reach_stderr_while_the_provider_channel_is_installed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The provider channel turns propagation off, which is what used to send
    warnings to `logging.lastResort`. Without the fallback standing in, turning
    on a debug switch would silence every warning in the run."""
    monkeypatch.setenv("STEL_DEBUG_PROVIDER_ERRORS", "1")
    configure_verbose_logging(0)

    logging.getLogger("stel.runner").warning("a warning the operator needs")

    assert "a warning the operator needs" in capsys.readouterr().err
