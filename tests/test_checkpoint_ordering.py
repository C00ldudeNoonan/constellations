"""Ordering invariants every flushing stage depends on (issue #401 review).

These are the rules that are easy to get subtly wrong per stage and expensive
to get wrong at scale, so they are tested against `FlushPublisher` directly
rather than only through a pipeline: a stage-level test would need a warehouse
failure at exactly the wrong instant to reach any of them.
"""
from __future__ import annotations

from typing import Any

import pytest

from stel.adapters import AdapterError, StateRecord, StateScope
from stel.execution.checkpoint import FlushPublisher
from stel.execution.contracts import RunError


class _RecordingAdapter:
    """Records the order of state and write calls, and can fail on cue."""

    def __init__(self, *, fail_on: str | None = None, error: Exception | None = None) -> None:
        self.calls: list[str] = []
        self._fail_on = fail_on
        self._error = error or AdapterError("adapter said no")

    def _maybe_fail(self, name: str) -> None:
        self.calls.append(name)
        if self._fail_on == name:
            raise self._error

    def replace_state(self, scope: StateScope, records: list[StateRecord]) -> None:
        del scope
        self._maybe_fail("replace_state_empty" if not records else "replace_state")

    def upsert_state(self, scope: StateScope, records: list[StateRecord]) -> None:
        del scope, records
        self._maybe_fail("upsert_state")


def _publisher(adapter: Any, *, use_full: bool) -> FlushPublisher:
    return FlushPublisher(
        adapter,
        model_name="m",
        state_scope=StateScope("m"),
        use_full=use_full,
    )


def _records() -> list[StateRecord]:
    return [StateRecord("r1", "fp", "cv")]


# ─── rule 1: a rebuild clears state before it writes ────────────────────────


def test_full_rebuild_clears_state_before_the_first_write() -> None:
    """State that outlives its rows is the dangerous direction.

    If the opening flush replaced the target and the state reset then failed,
    the old fingerprints would survive against a table holding one flush — and
    the next run would read that state, decide those rows were done, and leave
    the target permanently partial while reporting success.
    """
    adapter = _RecordingAdapter()
    publisher = _publisher(adapter, use_full=True)

    publisher.publish(
        write_full=lambda: adapter.calls.append("write_full") or 1,
        write_incremental=lambda: adapter.calls.append("write_incremental") or 1,
        state_records=_records(),
    )

    assert adapter.calls.index("replace_state_empty") < adapter.calls.index("write_full")


def test_a_failed_state_reset_leaves_the_target_untouched() -> None:
    """Clearing first means a failure costs a re-run, never silent row loss."""
    adapter = _RecordingAdapter(fail_on="replace_state_empty")
    publisher = _publisher(adapter, use_full=True)

    with pytest.raises(RunError):
        publisher.publish(
            write_full=lambda: adapter.calls.append("write_full") or 1,
            write_incremental=lambda: 0,
            state_records=_records(),
        )

    assert "write_full" not in adapter.calls


def test_only_the_first_publication_of_a_rebuild_replaces() -> None:
    adapter = _RecordingAdapter()
    publisher = _publisher(adapter, use_full=True)
    for _ in range(3):
        publisher.publish(
            write_full=lambda: adapter.calls.append("write_full") or 1,
            write_incremental=lambda: adapter.calls.append("write_incremental") or 1,
            state_records=_records(),
        )

    assert adapter.calls.count("write_full") == 1
    assert adapter.calls.count("write_incremental") == 2
    # And the clear happens exactly once, not per flush.
    assert adapter.calls.count("replace_state_empty") == 1


def test_an_incremental_run_never_clears_state() -> None:
    adapter = _RecordingAdapter()
    publisher = _publisher(adapter, use_full=False)

    publisher.publish(
        write_full=lambda: 1,
        write_incremental=lambda: adapter.calls.append("write_incremental") or 1,
        state_records=_records(),
    )

    assert "replace_state_empty" not in adapter.calls


# ─── rule 2: state advances only after the write lands ──────────────────────


def test_state_advances_after_the_write() -> None:
    adapter = _RecordingAdapter()
    publisher = _publisher(adapter, use_full=False)

    publisher.publish(
        write_full=lambda: 1,
        write_incremental=lambda: adapter.calls.append("write_incremental") or 1,
        state_records=_records(),
    )

    assert adapter.calls.index("write_incremental") < adapter.calls.index("upsert_state")


def test_a_failed_write_never_records_state() -> None:
    adapter = _RecordingAdapter()
    publisher = _publisher(adapter, use_full=False)

    def failing_write() -> int:
        raise AdapterError("merge failed")

    with pytest.raises(RunError):
        publisher.publish(
            write_full=lambda: 0,
            write_incremental=failing_write,
            state_records=_records(),
        )

    assert "upsert_state" not in adapter.calls


def test_a_write_that_records_its_own_state_is_not_double_recorded() -> None:
    """`replace_children` applies rows and state in one transaction."""
    adapter = _RecordingAdapter()
    publisher = _publisher(adapter, use_full=False)

    publisher.publish(
        write_full=lambda: 0,
        write_incremental=lambda: adapter.calls.append("replace_children") or 1,
        state_records=_records(),
        advances_state_itself=True,
    )

    assert "upsert_state" not in adapter.calls


# ─── rule 3: warehouse text never reaches the run result ────────────────────


def test_a_non_adapter_publication_failure_is_sanitized() -> None:
    """A driver's message can quote the offending row and the statement that
    touched it, and a RunError is persisted into run_results.json."""

    class _ConversionException(Exception):
        pass

    secret = "SSN 123-45-6789 in INSERT INTO customers"
    adapter = _RecordingAdapter()
    publisher = _publisher(adapter, use_full=False)

    def leaky_write() -> int:
        raise _ConversionException(secret)

    with pytest.raises(RunError) as caught:
        publisher.publish(
            write_full=lambda: 0,
            write_incremental=leaky_write,
            state_records=_records(),
        )

    assert secret not in str(caught.value)
    assert "123-45-6789" not in str(caught.value)
    # It still says what failed and what survived.
    assert "_ConversionException" in str(caught.value)
    assert "earlier flushes are retained" in str(caught.value)


def test_an_adapter_error_keeps_its_own_sanitized_message() -> None:
    adapter = _RecordingAdapter()
    publisher = _publisher(adapter, use_full=False)

    def failing_write() -> int:
        raise AdapterError("incremental key column is missing")

    with pytest.raises(RunError, match="incremental key column is missing"):
        publisher.publish(
            write_full=lambda: 0,
            write_incremental=failing_write,
            state_records=_records(),
        )


# ─── accounting ─────────────────────────────────────────────────────────────


def test_rows_written_accumulates_across_flushes() -> None:
    adapter = _RecordingAdapter()
    publisher = _publisher(adapter, use_full=False)
    for count in (3, 5, 2):
        publisher.publish(
            write_full=lambda: 0,
            write_incremental=lambda count=count: count,
            state_records=_records(),
        )

    assert publisher.rows_written == 10
    assert publisher.published_any
