"""Wall-clock attribution across the phases of one model run (issue #432).

Both open performance issues on the streaming stages begin by asking for the
same thing and neither can be answered from a run today. #432 wants "wall-clock
attribution for one production embed run — provider wait, warehouse write, stel
CPU, lock wait", and says plainly that everything after it "is a guess about
which term dominates, and the answer decides the order". #454 wants to know what
share of a snapshot read is transfer rather than client-side decode, because
compressing a client that is already CPU-bound makes it slower.

So this measures rather than optimizes. It exists to turn one production run
into evidence, and to keep a fixed regression from going quiet again — the
`batch_size` collapse in #452 cost roughly fourteen hours per publish and was
invisible until someone counted pages by hand.

**Summed, not sliced.** Provider batches overlap (#434), so phase totals can
exceed the run's wall clock and a reader must not treat them as shares of it.
Summed thread time is the right measure for "where does the work go"; the run's
own `duration_seconds` is the wall clock, and the ratio between them is how much
concurrency actually happened. Both are reported for exactly that reason.
"""
from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter

# Metric keys are prefixed so an operator reading run_results.json can tell
# timing from the counters beside it, and so a future phase cannot collide
# with an existing metric name.
METRIC_PREFIX = "seconds_"


class PhaseTimings:
    """Thread-safe accumulator of seconds spent per named phase.

    `perf_counter`, not `time`: this measures elapsed intervals, and a wall
    clock that steps (NTP, DST) would silently corrupt exactly the numbers a
    decision is about to rest on.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._totals: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Time a block, crediting it to `name` even if it raises.

        A failed provider call still spent the time, and a timing that only
        counted successes would flatter exactly the slow path worth finding.
        """
        started = perf_counter()
        try:
            yield
        finally:
            self.add(name, perf_counter() - started)

    def add(self, name: str, seconds: float) -> None:
        with self._lock:
            self._totals[name] = self._totals.get(name, 0.0) + seconds

    def as_metrics(self) -> dict[str, float]:
        """Phase totals as run-result metrics, rounded to milliseconds.

        Sub-millisecond precision would be noise: these are wall-clock spans
        over network and warehouse calls, and the decisions they inform are
        about which term dominates, not about microseconds.
        """
        with self._lock:
            return {
                f"{METRIC_PREFIX}{name}": round(seconds, 3)
                for name, seconds in sorted(self._totals.items())
            }
