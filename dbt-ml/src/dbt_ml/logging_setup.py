"""Optional verbose logging for long-running CLI commands (issue #268).

By default the CLI writes only its summary tables; when a build spans thousands
of documents over many minutes, that leaves callers with no way to tell whether
progress is being made. Callers opt in with ``-v``/``-vv`` on the CLI or the
``DBT_ML_VERBOSE`` env var (``1``/``2``), which configures a stderr handler on
the ``dbt_ml`` logger namespace so the ``log.info(...)`` calls already sprinkled
through discovery, extraction, and the runner become visible without changing
default output.
"""

from __future__ import annotations

import logging
import os
import sys

_VERBOSE_ENV_VAR = "DBT_ML_VERBOSE"
_HANDLER_ATTR = "_dbt_ml_verbose_handler"
_ROOT_LOGGER = "dbt_ml"


def resolve_verbosity(cli_count: int) -> int:
    """CLI ``-v`` count wins over the env var; otherwise fall back to it."""
    if cli_count > 0:
        return cli_count
    raw = os.environ.get(_VERBOSE_ENV_VAR, "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 1


def configure_verbose_logging(verbosity: int) -> None:
    """Attach a single stderr handler to the ``dbt_ml`` logger.

    Idempotent: repeated calls replace the previous handler rather than stacking
    duplicates, so re-invocation across nested commands or tests stays clean.
    """
    logger = logging.getLogger(_ROOT_LOGGER)
    existing = getattr(logger, _HANDLER_ATTR, None)
    if existing is not None:
        logger.removeHandler(existing)
        setattr(logger, _HANDLER_ATTR, None)

    if verbosity <= 0:
        return

    level = logging.DEBUG if verbosity >= 2 else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    # Progress lines are for the operator, not any parent handler that may exist
    # (e.g. a Dagster capture that reformats records).
    logger.propagate = False
    setattr(logger, _HANDLER_ATTR, handler)
