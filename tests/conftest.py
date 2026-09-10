from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def example_project_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "invoice_pipeline"


@pytest.fixture(autouse=True)
def _restore_stel_logging() -> Iterator[None]:
    """Undo whatever a test did to the `stel` logger (issue #518).

    `configure_verbose_logging` sets `propagate = False` so progress lines reach the
    operator rather than a parent handler, and it is process-global by nature:
    a CLI configures logging once for the run. A test that invokes the CLI with
    `-v` therefore leaves propagation off for every test that follows, and
    `caplog` listens on the *root* logger -- so any later assertion about a
    `stel` log record silently sees nothing.

    That was real: `tests/test_search.py::test_search_v_reports_timings_on_
    stderr_and_leaves_json_parseable` broke `tests/test_append_logs.py::
    test_a_failed_log_write_does_not_raise` whenever the two ran together. The
    full suite passed, because a third file happened to reconfigure logging in
    between, so **CI could not catch it** -- it only appeared when someone ran a
    subset, which is what you do while iterating.

    Restoring here rather than in each offending test, because the next such
    test would reintroduce it and the failure surfaces far from its cause.
    """
    logger = logging.getLogger("stel")
    handlers = list(logger.handlers)
    level, propagate = logger.level, logger.propagate
    try:
        yield
    finally:
        logger.handlers[:] = handlers
        logger.setLevel(level)
        logger.propagate = propagate


@pytest.fixture(autouse=True)
def _restore_adapter_registry() -> Iterator[None]:
    """Undo whatever a test did to the adapter registry (issue #518, #395).

    `_REGISTRY` is process-global and `register` is a plain assignment, so a
    test that registers a stub adapter leaves it there for every test that
    follows -- and one that registers a name an adapter already owns replaces
    the real adapter, which surfaces as an unrelated failure in some other
    file.

    Snapshotting here rather than in the test that happens to register one,
    because the next such test will not know to: a `finally` that deletes its
    own key restores nothing if the key was already taken.
    """
    from stel.adapters.registry import _REGISTRY

    snapshot = dict(_REGISTRY)
    try:
        yield
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(snapshot)
