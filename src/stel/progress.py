"""Long-running-command progress feedback (issue #268).

Verbose mode (``-v``, ``STEL_VERBOSE=1``) attaches a live per-model progress
bar to stderr for the extraction loop and prints one line per source-discovery
completion. Non-TTY callers (Dagster captures, redirected stderr) get the plain
``log.info`` lines emitted by discovery/extraction/runner instead, since a
carriage-return bar redrawn every flush is noise in a captured stream.

On a TTY the log channel runs *alongside* the bar rather than in place of it
(issue #403): records reach :meth:`ProgressReporter.detail`, which buffers them
while a bar is live and flushes them once it finishes. Before that, a terminal
run saw only the four events this module renders itself, so provider batch
polls and source-scan lines were visible only when stderr was redirected —
watching a run live showed strictly less than piping it to a file.

The active reporter is module-global to mirror the existing ``logging`` pattern:
extraction and the runner query :func:`get_reporter` instead of threading a
reporter argument through every executor.
"""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from types import TracebackType
from typing import Any, Protocol, Self, TextIO

import click


class ProgressTask(Protocol):
    def advance(self, n: int = 1) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...


class ProgressReporter(Protocol):
    def source_discovered(self, source_name: str, count: int) -> None: ...
    def model_task(self, model_name: str, kind: str, total: int) -> ProgressTask: ...
    def model_finished(
        self, model_name: str, rows: int, duration: float, status: str | None
    ) -> None: ...
    def publication(self, message: str) -> None: ...
    def detail(self, message: str) -> None: ...


class _NullTask:
    def advance(self, n: int = 1) -> None:
        return

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return


class _NullReporter:
    def source_discovered(self, source_name: str, count: int) -> None:
        return

    def model_task(self, model_name: str, kind: str, total: int) -> ProgressTask:
        return _NullTask()

    def model_finished(
        self, model_name: str, rows: int, duration: float, status: str | None
    ) -> None:
        return

    def publication(self, message: str) -> None:
        return

    def detail(self, message: str) -> None:
        return


class _TerminalTask:
    """Wraps ``click.progressbar`` and skips the render entirely when total is 0
    so an empty extraction doesn't leave a stale header on the screen. While the
    bar is live it signals the reporter so per-flush detail lines (publication
    telemetry) are deferred instead of smearing the partially rendered bar."""

    def __init__(
        self,
        label: str,
        total: int,
        stream: TextIO,
        reporter: _TerminalReporter | None = None,
    ) -> None:
        self._total = total
        self._reporter = reporter
        self._bar: Any = None
        if total > 0:
            self._bar = click.progressbar(
                length=total,
                label=label,
                file=stream,
                show_pos=True,
                show_percent=True,
                show_eta=True,
            )

    def advance(self, n: int = 1) -> None:
        bar = self._bar
        if bar is not None:
            bar.update(n)

    def __enter__(self) -> Self:
        bar = self._bar
        if bar is not None:
            bar.__enter__()
            if self._reporter is not None:
                self._reporter._bar_started()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        bar = self._bar
        if bar is not None:
            bar.__exit__(exc_type, exc, tb)
            if self._reporter is not None:
                self._reporter._bar_finished()


# Deferred detail lines are dropped oldest-first past this many. A bar that
# stays live for hours (a provider batch poll every few seconds) would otherwise
# hold every line it produced in memory to print them all at once, which helps
# nobody: the operator wants the tail. Overflow is reported, never silent.
_MAX_DEFERRED_LINES = 500


class _TerminalReporter:
    """Terminal-friendly reporter. Per-model progress bars go on stderr; source
    discovery, model-finished status, publication telemetry and forwarded log
    records get one-line summaries around the bar."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        # A live per-model bar defers detail lines; >0 while one is active. Only
        # one bar is ever active on this channel — verbose over multiple
        # concurrent models (run --threads N) uses the log channel, not the
        # reporter — so a single counter and buffer suffice.
        self._bar_depth = 0
        self._pending: deque[str] = deque(maxlen=_MAX_DEFERRED_LINES)
        self._dropped = 0
        # Forwarded log records arrive on provider worker threads (extraction
        # fans out over a pool), so the buffer is not single-threaded the way
        # the publication-only version was.
        self._lock = threading.Lock()

    def _echo(self, message: str) -> None:
        click.echo(message, file=self._stream)

    def _defer_or_echo(self, line: str) -> None:
        """Print now, or hold until the live bar finishes. ``click.echo`` does
        not clear and redraw a ``click.progressbar``, so writing while one is
        active smears it."""
        with self._lock:
            if self._bar_depth > 0:
                if len(self._pending) == self._pending.maxlen:
                    self._dropped += 1
                self._pending.append(line)
                return
        self._echo(line)

    def _bar_started(self) -> None:
        with self._lock:
            self._bar_depth += 1

    def _bar_finished(self) -> None:
        with self._lock:
            self._bar_depth = max(0, self._bar_depth - 1)
            if self._bar_depth > 0:
                return
            pending = list(self._pending)
            dropped = self._dropped
            self._pending.clear()
            self._dropped = 0
        # Echo outside the lock: click.echo touches the stream, and a reentrant
        # log record from another thread would otherwise deadlock behind it.
        if dropped:
            self._echo(f"... {dropped:,} earlier detail line(s) dropped ...")
        for message in pending:
            self._echo(message)

    def source_discovered(self, source_name: str, count: int) -> None:
        self._echo(f"[source] {source_name}: discovered {count:,} object(s)")

    def model_task(self, model_name: str, kind: str, total: int) -> ProgressTask:
        label = f"[{kind}] {model_name}"
        if total == 0:
            self._echo(f"{label}: 0 documents (nothing to process)")
        return _TerminalTask(label, total, self._stream, reporter=self)

    def model_finished(
        self, model_name: str, rows: int, duration: float, status: str | None
    ) -> None:
        status_suffix = f" [{status}]" if status else ""
        self._echo(
            f"[done]   {model_name}: {rows:,} row(s) in "
            f"{_format_duration(duration)}{status_suffix}"
        )

    def publication(self, message: str) -> None:
        # Publication telemetry (issue #292) fires per flush, inside the model's
        # live progress bar, so it takes the deferral path.
        self._defer_or_echo(f"[publish] {message}")

    def detail(self, message: str) -> None:
        """Render one already-formatted line from the forwarded log channel.

        No prefix: the record arrives pre-formatted by the logging handler,
        which supplies its own timestamp/level/logger shape."""
        self._defer_or_echo(message)


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {secs}s"


_reporter: ProgressReporter = _NullReporter()


def get_reporter() -> ProgressReporter:
    return _reporter


def set_reporter(reporter: ProgressReporter | None) -> None:
    global _reporter
    _reporter = reporter if reporter is not None else _NullReporter()


def configure_progress(verbosity: int, *, stream: TextIO | None = None) -> bool:
    """Install a terminal reporter when verbose is requested and stderr is a
    TTY. Non-TTY callers keep the null reporter so the plain ``log.info`` lines
    stay the sole channel — a captured log stream doesn't want a carriage-return
    progress bar re-written every flush.

    Returns True when a terminal reporter was installed. Callers use this to
    avoid enabling the INFO log handler at the same time; a redrawing
    ``click.progressbar`` and a plain ``log.info`` line share stderr, so
    running both would corrupt the bar and double-print discovery / model
    boundary events (once from the log call, once from the reporter callback).
    """
    target = stream if stream is not None else sys.stderr
    if verbosity <= 0 or not _is_tty(target):
        set_reporter(None)
        return False
    set_reporter(_TerminalReporter(target))
    return True


def _is_tty(stream: TextIO) -> bool:
    isatty = getattr(stream, "isatty", None)
    if isatty is None:
        return False
    try:
        return bool(isatty())
    except (ValueError, OSError):
        return False


# Kept for readability at import sites — a caller that wants a real timer can
# still just use time.monotonic() directly.
now_monotonic = time.monotonic
