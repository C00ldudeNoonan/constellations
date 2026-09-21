"""Optional verbose logging for long-running CLI commands (issue #268).

By default the CLI writes only its summary tables; when a build spans thousands
of documents over many minutes, that leaves callers with no way to tell whether
progress is being made. Callers opt in with ``-v`` on the CLI or the
``STEL_VERBOSE`` env var, which configures an INFO handler on the ``stel``
logger namespace so the ``log.info(...)`` calls already sprinkled through
discovery, extraction, and the runner become visible without changing default
output. The handler writes to stderr directly on a captured run, or hands
records to the progress reporter on a TTY so they coexist with a live progress
bar (issue #403) instead of the two channels excluding each other.

Deliberately capped at INFO. Enabling DEBUG through this flag would surface
the ``log.debug(..., exc_info=True)`` sites in ``execution/transform.py`` and
provider code (which carry raw exception text and traceback frames that
``artifact_error_text`` sanitizes for the user-facing error path); AGENTS.md
requires that sensitive exception text stay out of logs. Callers who need
DEBUG for troubleshooting should attach their own handler.

That hatch is reachable only from Python. Every orchestrated run invokes the
CLI as a subprocess, so for the operator who actually hits a sanitized store
failure the native cause was written to a logger nothing could receive
(issue #590). ``--diagnostics-file PATH`` (or ``STEL_DIAGNOSTICS_FILE``) is the
CLI-shaped version of the same hatch: :func:`configure_diagnostics_file`
attaches a DEBUG handler that writes only the records carrying an exception,
plus warnings, to one file the operator named. It is the sole disclosure
surface -- nothing changes on stderr, in ``run_results.json`` or in any
artifact, and while it is installed the ``stel`` logger stops propagating so a
parent handler cannot receive what only the file was meant to. See ADR-0012.

``STEL_DEBUG_PROVIDER_ERRORS`` is the third channel and discloses the least:
provider errors are sanitized before any logger sees them, so what it emits is
``redacted_exception_text``'s allowlist -- exception types, stel frame
locations, an external frame count -- and never native text. It had no
destination at all until issue #599: the call sites were gated on a DEBUG
level the CLI caps at INFO, and the records carry no ``exc_info``, so even
with a diagnostics file attached they fired and were then filtered out. The
switch now raises the level itself and, when no diagnostics file is
configured, installs a stderr channel scoped to those records alone.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from .env import (
    DIAGNOSTICS_FILE_ENV,
    PROVIDER_DEBUG_ENV,
    VERBOSE_ENV,
    env_flag_enabled,
    read_env,
)
from .progress import ProgressReporter, reporter_is_active

# Marks a record whose event the progress reporter also renders itself (source
# discovery, model completion, BigQuery publication telemetry). Since #403 both
# channels can be live at once, and since #404 the reporter is installed even
# without `-v`, so without this the operator would see such events twice — once
# from the emitting log call, once from the reporter. Dropped only while a
# reporter is actually rendering: on a `--json` run the log record is the event's
# only copy and must survive.
REPORTER_ECHO = "stel_reporter_echo"
REPORTER_ECHO_EXTRA = {REPORTER_ECHO: True}


def _drop_reporter_echoes(record: logging.LogRecord) -> bool:
    """``logging.Filter`` callable: False drops the record.

    Asked per record rather than decided at install time — the reporter can be
    swapped (nested commands, tests) after the handler is attached.
    """
    return not (getattr(record, REPORTER_ECHO, False) and reporter_is_active())


# Marks the redacted provider-error diagnostics `STEL_DEBUG_PROVIDER_ERRORS`
# opts into (issue #599). They are the one DEBUG record in the tree with no
# `exc_info`: `redacted_exception_text` exists so that native text never rides
# along, which is exactly what made them unroutable — `_carries_diagnostics`
# recognizes the others by their exception and the general cap dropped these.
# Hence a marker, the same mechanism as REPORTER_ECHO above.
PROVIDER_DIAGNOSTICS = "stel_provider_diagnostics"
PROVIDER_DIAGNOSTICS_EXTRA = {PROVIDER_DIAGNOSTICS: True}


def _is_provider_diagnostics(record: logging.LogRecord) -> bool:
    """``logging.Filter`` callable for the stderr channel the switch installs."""
    return bool(getattr(record, PROVIDER_DIAGNOSTICS, False))


_HANDLER_ATTR = "_stel_verbose_handler"
_DIAGNOSTICS_ATTR = "_stel_diagnostics_handler"
# The stderr channel for provider diagnostics when no diagnostics file is
# configured. Scoped to marked records, so raising the logger to DEBUG for it
# cannot put any other DEBUG record — the ones carrying native text — on stderr.
_PROVIDER_DIAGNOSTICS_ATTR = "_stel_provider_diagnostics_handler"
# Stands in for `logging.lastResort` while diagnostics turn propagation off and
# `-v` has installed nothing: warnings keep reaching stderr exactly as before.
_FALLBACK_ATTR = "_stel_fallback_handler"
# Must stay equal to the top-level package name: a handler attached to a
# namespace no module logs under silences `-v` without failing. Pinned in
# tests/test_frozen_names.py.
_ROOT_LOGGER = "stel"


def resolve_verbosity(cli_count: int) -> int:
    """CLI ``-v`` count wins over the env var; otherwise fall back to it.

    Repeated ``-v``s and env-var values greater than one both collapse to a
    single verbosity level so the DEBUG safety cap in
    :func:`configure_verbose_logging` can never be bypassed by shouting.
    """
    if cli_count > 0:
        return 1
    raw = read_env(VERBOSE_ENV, default="").strip()
    if not raw:
        return 0
    try:
        return 1 if int(raw) > 0 else 0
    except ValueError:
        return 1


class _ReporterHandler(logging.Handler):
    """Hands formatted records to a progress reporter instead of a stream.

    The reporter owns the terminal while a ``click.progressbar`` is live, so it
    is the only thing that can decide whether a line prints now or waits for the
    bar to finish. Records whose event the reporter already renders as a
    callback are dropped by ``_drop_reporter_echoes`` rather than at the call
    site: the emitting module should not have to know which channel is
    installed.
    """

    def __init__(self, reporter: ProgressReporter) -> None:
        super().__init__()
        self._reporter = reporter

    def emit(self, record: logging.LogRecord) -> None:
        # stdlib handler contract: a logging failure must never propagate into
        # the code being logged. handleError honors logging.raiseExceptions.
        try:
            self._reporter.detail(self.format(record))
        except Exception:
            self.handleError(record)


def configure_verbose_logging(
    verbosity: int, *, reporter: ProgressReporter | None = None
) -> None:
    """Attach a single INFO-level handler to the ``stel`` logger.

    Idempotent: repeated calls replace the previous handler rather than stacking
    duplicates, so re-invocation across nested commands or tests stays clean.
    The handler's level is fixed at INFO — see the module docstring for why;
    the logger's own level is derived by :func:`_apply_channel_policy`, since
    the diagnostics file may need DEBUG while this handler still filters at INFO.

    With ``reporter``, records are routed through it so they interleave safely
    with a live progress bar (issue #403); without one they go straight to
    stderr, which is what a captured/orchestrated run wants.
    """
    logger = logging.getLogger(_ROOT_LOGGER)
    existing = getattr(logger, _HANDLER_ATTR, None)
    if existing is not None:
        logger.removeHandler(existing)
        setattr(logger, _HANDLER_ATTR, None)

    if verbosity <= 0:
        # Fully restore the default so disabling verbose leaves no lingering
        # state: the policy helper turns propagation back on unless the
        # diagnostics file still needs it off.
        _apply_channel_policy(logger)
        return

    handler: logging.Handler = (
        logging.StreamHandler(sys.stderr)
        if reporter is None
        else _ReporterHandler(reporter)
    )
    handler.setLevel(logging.INFO)
    handler.addFilter(_drop_reporter_echoes)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    setattr(logger, _HANDLER_ATTR, handler)
    _apply_channel_policy(logger)


def resolve_diagnostics_file(cli_value: Path | None) -> Path | None:
    """The CLI option wins over ``STEL_DIAGNOSTICS_FILE``; an empty variable is unset."""
    if cli_value is not None:
        return cli_value
    raw = read_env(DIAGNOSTICS_FILE_ENV, default="").strip()
    return Path(raw) if raw else None


def _carries_diagnostics(record: logging.LogRecord) -> bool:
    """``logging.Filter`` callable: the records the diagnostics file exists for.

    Every sanitized failure in the tree logs its native exception once, at
    DEBUG with ``exc_info``, and nothing else in the tree does; warnings ride
    along so a retry sequence reads in order without the stderr capture.

    Provider errors are the exception and must be named rather than inferred:
    they are sanitized before any logger sees them, so their diagnostics carry
    a redacted allowlist and no ``exc_info`` at all, and would be dropped here
    by the very property that makes them safe (issue #599).
    """
    return (
        record.exc_info is not None
        or record.levelno >= logging.WARNING
        or _is_provider_diagnostics(record)
    )


class _OwnerOnlyFileHandler(logging.FileHandler):
    """Appends to a file readable by its owner alone, and fails closed.

    The file holds what every other channel sanitizes away -- object-store
    URIs with their query strings, provider response bodies -- so it gets the
    owner-only storage the cache directories already get, including a file an
    orchestrator pre-created with ordinary permissions. Opened lazily, so a
    run that fails nowhere leaves no file behind.

    Every failure of the handler itself is contained here. The stdlib opens a
    delayed file outside ``StreamHandler.emit``'s guard, and its
    ``handleError`` prints the active exception chain to stderr; both run
    while the native exception is being handled, so either would put on
    stderr exactly the text this file exists to keep off it.
    """

    def __init__(self, path: Path) -> None:
        super().__init__(path, mode="a", encoding="utf-8", delay=True)
        self.records_written = 0
        self._unusable = False

    def _open(self) -> Any:
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(self.baseFilename, flags, 0o600)
        # O_CREAT's mode applies only to a file it creates; an existing one
        # keeps whatever it had, so tighten it explicitly where the OS allows.
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        return open(fd, self.mode, encoding=self.encoding)

    def emit(self, record: logging.LogRecord) -> None:
        if self._unusable:
            return
        if self.stream is None:
            try:
                self.stream = self._open()
            except OSError:
                self.handleError(record)
                return
        super().emit(record)
        if not self._unusable:
            self.records_written += 1

    def handleError(self, record: logging.LogRecord) -> None:
        """One safe line on stderr, then stop trying.

        Never the stdlib report: it prints the exception being handled with
        its ``__context__``, which here is the native failure.
        """
        del record
        self._unusable = True
        error = sys.exc_info()[1]
        if sys.stderr is not None:
            sys.stderr.write(
                f"stel: diagnostics file {self.baseFilename} could not be written "
                f"[{type(error).__name__}]; native failure detail was not recorded\n"
            )


def configure_diagnostics_file(path: Path | None) -> None:
    """Attach, replace, or remove the DEBUG-level diagnostics file handler.

    Idempotent like :func:`configure_verbose_logging`, and independent of it:
    either can be configured first, and the logger's level and propagation are
    recomputed from whichever handlers are present. Passing ``None`` removes
    the handler and restores the default channel policy.
    """
    logger = logging.getLogger(_ROOT_LOGGER)
    existing = getattr(logger, _DIAGNOSTICS_ATTR, None)
    if existing is not None:
        logger.removeHandler(existing)
        existing.close()
        setattr(logger, _DIAGNOSTICS_ATTR, None)
    if path is not None:
        handler = _OwnerOnlyFileHandler(path)
        handler.setLevel(logging.DEBUG)
        handler.addFilter(_carries_diagnostics)
        handler.setFormatter(
            logging.Formatter(fmt="%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        setattr(logger, _DIAGNOSTICS_ATTR, handler)
    _apply_channel_policy(logger)


def diagnostics_file_written() -> Path | None:
    """The diagnostics file's path once at least one record has reached it.

    Lets a CLI failure boundary point the operator at the detail it could not
    print. ``None`` when no file is configured or nothing has been written.
    """
    handler = getattr(logging.getLogger(_ROOT_LOGGER), _DIAGNOSTICS_ATTR, None)
    if handler is None or handler.records_written == 0:
        return None
    return Path(handler.baseFilename)


def _apply_channel_policy(logger: logging.Logger) -> None:
    """Derive level, propagation and the stderr fallback from installed handlers.

    The logger's level is the lowest any handler needs: DEBUG while the
    diagnostics file is attached or ``STEL_DEBUG_PROVIDER_ERRORS`` is on, INFO
    under ``-v``, otherwise unset so the root's default applies as before.
    Propagation is off whenever a handler is installed -- progress lines are
    for the operator, not a parent handler that reformats records (a Dagster
    capture, say), and a DEBUG record that escaped to one would defeat the
    diagnostics file's whole point. With propagation off and no ``-v``
    handler, warnings would no longer reach ``logging.lastResort``, so a plain
    WARNING stderr handler stands in for it.

    Raising the level to DEBUG for the provider switch moves where native
    text is stopped, and that is worth being precise about. Every other DEBUG
    site in the tree carries it, and those records now pass the logger's level
    check; each of stel's own handlers then refuses them -- the provider
    channel filters to marked records, ``-v`` sits at INFO, the fallback at
    WARNING -- and propagation is off, so no parent handler sees them either.
    The level is only ever raised alongside one of those filtering handlers.
    What this does not cover is a handler the in-process caller attached to
    the ``stel`` logger themselves: that one will now receive DEBUG records it
    would previously have had to raise the level to see.
    """
    verbose = getattr(logger, _HANDLER_ATTR, None)
    diagnostics = getattr(logger, _DIAGNOSTICS_ATTR, None)
    fallback = getattr(logger, _FALLBACK_ATTR, None)
    if fallback is not None:
        logger.removeHandler(fallback)
        setattr(logger, _FALLBACK_ATTR, None)

    provider_channel = getattr(logger, _PROVIDER_DIAGNOSTICS_ATTR, None)
    if provider_channel is not None:
        logger.removeHandler(provider_channel)
        setattr(logger, _PROVIDER_DIAGNOSTICS_ATTR, None)
    # Read per call, not at import: the variable is part of the environment a
    # test or an embedding caller changes between runs, and the answer decides
    # both the level and the destination.
    provider_debug = env_flag_enabled(PROVIDER_DEBUG_ENV)
    if provider_debug and diagnostics is None:
        channel = logging.StreamHandler(sys.stderr)
        channel.setLevel(logging.DEBUG)
        channel.addFilter(_is_provider_diagnostics)
        channel.setFormatter(logging.Formatter(fmt="%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(channel)
        setattr(logger, _PROVIDER_DIAGNOSTICS_ATTR, channel)
        provider_channel = channel

    if diagnostics is not None or provider_debug:
        logger.setLevel(logging.DEBUG)
    elif verbose is not None:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.NOTSET)
    logger.propagate = verbose is None and diagnostics is None and provider_channel is None

    if not logger.propagate and verbose is None:
        stderr_fallback = logging.StreamHandler(sys.stderr)
        stderr_fallback.setLevel(logging.WARNING)
        logger.addHandler(stderr_fallback)
        setattr(logger, _FALLBACK_ATTR, stderr_fallback)
