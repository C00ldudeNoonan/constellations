"""Optional verbose logging for long-running CLI commands (issue #268).

By default the CLI writes only its summary tables; when a build spans thousands
of documents over many minutes, that leaves callers with no way to tell whether
progress is being made. Callers opt in with ``-v`` on the CLI or the
``STEL_VERBOSE`` env var, which configures a stderr handler on the ``stel``
logger namespace so the ``log.info(...)`` calls already sprinkled through
discovery, extraction, and the runner become visible without changing default
output.

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


def configure_verbose_logging(verbosity: int) -> None:
    """Attach a single INFO-level stderr handler to the ``stel`` logger.

    Idempotent: repeated calls replace the previous handler rather than stacking
    duplicates, so re-invocation across nested commands or tests stays clean.
    Level is fixed at INFO — see the module docstring for why.
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

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.INFO)
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
