"""Wall-clock attribution across a model run (#432 item 1).

Both open performance issues begin by asking where the time goes and neither
can be answered from a run today. These tests are about the properties that
make an answer trustworthy: time is credited even when the work fails, phases
survive concurrency, and the numbers actually reach the operator.
"""
from __future__ import annotations

import threading
from typing import Any

import pytest

from stel.timing import PhaseTimings


def test_a_phase_accumulates_across_entries() -> None:
    timings = PhaseTimings()

    for _ in range(3):
        with timings.phase("provider"):
            pass

    assert set(timings.as_metrics()) == {"seconds_provider"}
    assert timings.as_metrics()["seconds_provider"] >= 0.0


def test_time_is_credited_even_when_the_block_raises() -> None:
    """A failed provider call still spent the time. Counting only successes
    would flatter exactly the slow path worth finding."""
    timings = PhaseTimings()

    with pytest.raises(RuntimeError):
        with timings.phase("provider"):
            raise RuntimeError("provider exploded")

    assert "seconds_provider" in timings.as_metrics()


def test_concurrent_phases_sum_exactly() -> None:
    """Provider batches overlap (#434), so the accumulator is written from
    several threads at once.

    Note what this does *not* prove: under the GIL a read-modify-write this
    short almost never interleaves, so an unsynchronized version passes it
    too — verified by mutation. The lock is pinned by the test below instead;
    this one guards the arithmetic.
    """
    timings = PhaseTimings()

    def work() -> None:
        for _ in range(50):
            timings.add("provider", 0.001)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert timings.as_metrics()["seconds_provider"] == pytest.approx(0.4, abs=1e-6)


def test_the_accumulator_is_guarded_by_its_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asserted directly rather than raced for. Mutual exclusion is not
    reliably observable from the outside in CPython, so a test that tried to
    provoke a lost update would pass whether or not the lock existed — which
    is worse than no test, because it reads like coverage."""
    timings = PhaseTimings()
    acquisitions = 0
    real_lock = timings._lock

    class _CountingLock:
        def __enter__(self) -> Any:
            nonlocal acquisitions
            acquisitions += 1
            return real_lock.__enter__()

        def __exit__(self, *args: Any) -> Any:
            return real_lock.__exit__(*args)

    monkeypatch.setattr(timings, "_lock", _CountingLock())

    timings.add("provider", 1.0)
    assert acquisitions == 1, "add() mutated the totals without taking the lock"

    timings.as_metrics()
    assert acquisitions == 2, "as_metrics() read the totals without the lock"


def test_metrics_are_prefixed_and_sorted() -> None:
    """Prefixed so an operator can tell timing from the counters beside it,
    and so a future phase cannot collide with an existing metric name."""
    timings = PhaseTimings()
    timings.add("publish", 2.0)
    timings.add("read", 1.0)

    metrics = timings.as_metrics()

    assert list(metrics) == ["seconds_publish", "seconds_read"]


def test_totals_are_rounded_to_milliseconds() -> None:
    """These are spans over network and warehouse calls; sub-millisecond
    digits are noise in a number used to decide which term dominates."""
    timings = PhaseTimings()
    timings.add("read", 1.23456789)

    assert timings.as_metrics()["seconds_read"] == 1.235


def test_merge_folds_another_accumulator_in() -> None:
    """The adapter breaks a read into transfer, decode and client copy because
    only it can; the executor owns the run's metrics. This is that seam."""
    run = PhaseTimings()
    run.add("read", 5.0)
    adapter = PhaseTimings()
    adapter.add("read_transfer", 3.0)
    adapter.add("read_decode", 1.5)

    run.merge(adapter)

    metrics = run.as_metrics()
    assert metrics["seconds_read"] == 5.0
    assert metrics["seconds_read_transfer"] == 3.0
    assert metrics["seconds_read_decode"] == 1.5


def test_merge_accumulates_rather_than_replaces() -> None:
    run = PhaseTimings()
    run.add("read_transfer", 1.0)
    other = PhaseTimings()
    other.add("read_transfer", 2.0)

    run.merge(other)

    assert run.as_metrics()["seconds_read_transfer"] == 3.0
