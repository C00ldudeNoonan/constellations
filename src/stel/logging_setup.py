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
"""

from __future__ import annotations

import logging
import sys

from .env import VERBOSE_ENV, read_env
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

_HANDLER_ATTR = "_stel_verbose_handler"
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
    Level is fixed at INFO — see the module docstring for why.

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
        # state — otherwise `propagate = False` (set below) would persist and
        # silently drop `stel` records from any parent/root handler.
        logger.propagate = True
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
    logger.setLevel(logging.INFO)
    # Progress lines are for the operator, not any parent handler that may exist
    # (e.g. a Dagster capture that reformats records).
    logger.propagate = False
    setattr(logger, _HANDLER_ATTR, handler)
