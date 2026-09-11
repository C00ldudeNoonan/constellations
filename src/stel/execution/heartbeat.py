"""Periodic progress-line throttling for a long warehouse scan (issue #469).

Every execution kind that streams a whole parent table before any batch
exists to log progress against needs the same thing: a heartbeat that still
fires from elapsed time alone while the calling thread is blocked inside a
warehouse read (a slow query, or a stalled batch fetch), not only between
processed rows. `run_sql_model`'s classification pass was the first to need
it; `run_chunk_model`'s parent scan needed the identical shape rather than a
second implementation of it (issue #573 -- the chunk kind logged nothing for
its whole parent scan, the same silence #469 fixed one model kind over).
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Iterator
from time import monotonic

DEFAULT_HEARTBEAT_ROWS = 5_000
DEFAULT_HEARTBEAT_SECONDS = 15.0


class Heartbeat:
    """Tracks whether a periodic progress line has earned its place, and can
    run a background watchdog so one still fires purely from elapsed time
    while the calling thread is blocked inside a warehouse read (issue #469
    Codex review): checking only between processed rows misses a slow query
    or a stalled batch fetch, either of which can hold the thread -- with
    nothing yet to check a heartbeat against -- for the run's whole duration.

    `update()`/`try_claim()` from the calling thread and the watchdog's own
    timer share one lock, so whichever notices first claims the heartbeat and
    the other is a no-op rather than a duplicate log line.
    """

    def __init__(
        self,
        *,
        rows: int = DEFAULT_HEARTBEAT_ROWS,
        seconds: float = DEFAULT_HEARTBEAT_SECONDS,
    ) -> None:
        self._rows = rows
        self._seconds = seconds
        self._start = monotonic()
        self._at_count = 0
        self._at_time = self._start
        self._count = 0
        self._lock = threading.Lock()

    def update(self, count: int) -> None:
        """Record progress the calling thread has made; no logging here."""
        with self._lock:
            self._count = count

    def try_claim(self) -> tuple[int, float] | None:
        """If a heartbeat is due, atomically claims it and returns
        (count, elapsed) to log. Returns None otherwise, including when
        another caller (the watchdog, or the row loop) claimed it first."""
        with self._lock:
            now = monotonic()
            if (
                self._count - self._at_count < self._rows
                and now - self._at_time < self._seconds
            ):
                return None
            self._at_count = self._count
            self._at_time = now
            return self._at_count, now - self._start

    @contextlib.contextmanager
    def watch(self, log_line: Callable[[int, float], None]) -> Iterator[None]:
        """Calls `log_line(count, elapsed)` from a background thread whenever
        a heartbeat is due, covering any stretch where the calling thread is
        blocked in I/O rather than between rows."""
        stop = threading.Event()

        def _run() -> None:
            while not stop.wait(self._seconds):
                claimed = self.try_claim()
                if claimed is not None:
                    log_line(*claimed)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=self._seconds)
