"""Long-running-command progress feedback (issues #268, #403, #404).

Every non-JSON invocation gets a reporter, at one of three levels:

``QUIET``
    ``--json``. Nothing on stderr; the payload on stdout is the whole output.

``NORMAL``
    The default. A running ledger — a header naming the run, one line per model
    as it completes, and a footer — so a build that takes forty minutes says
    what it is doing without the operator having had to predict that in advance
    (issue #404). No progress bars, TTY or not.

``VERBOSE``
    ``-v`` / ``STEL_VERBOSE=1``. The ledger, plus per-source discovery lines,
    per-model progress bars on a TTY, and the forwarded ``stel`` INFO log.

Bars are TTY-only: a carriage-return bar redrawn every flush is noise in a
captured stream. The ledger is not — a CI log wants it as much as a terminal
does, and it goes to stderr so ``--json`` on stdout stays parseable.

On a TTY the log channel runs *alongside* the bar rather than in place of it
(issue #403): records reach :meth:`ProgressReporter.detail`, which buffers them
while a bar is live and flushes them once it finishes. Before that, a terminal
run saw only the events this module renders itself, so provider batch polls and
source-scan lines were visible only when stderr was redirected — watching a run
live showed strictly less than piping it to a file.

The active reporter is module-global to mirror the existing ``logging`` pattern:
extraction and the runner query :func:`get_reporter` instead of threading a
reporter argument through every executor.
"""

from __future__ import annotations

import enum
import sys
import threading
import time
from collections import deque
from types import TracebackType
from typing import Any, Protocol, Self, TextIO

import click


class OutputLevel(enum.IntEnum):
    """How much the CLI says while a command runs. Ordered, so callers can ask
    ``level >= OutputLevel.VERBOSE`` rather than enumerating members."""

    QUIET = 0
    NORMAL = 1
    VERBOSE = 2


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
    def run_started(
        self, total: int, *, target: str, warehouse: str, project_total: int
    ) -> None: ...
    def source_discovered(self, source_name: str, count: int) -> None: ...
    def model_task(self, model_name: str, kind: str, total: int) -> ProgressTask: ...
    def model_finished(
        self,
        model_name: str,
        kind: str,
        rows: int,
        duration: float,
        status: str | None,
        *,
        failed: bool,
    ) -> None: ...
    def model_skipped(self, model_name: str, reason: str) -> None: ...
    def run_finished(self, *, ok: int, errored: int, skipped: int) -> None: ...
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
    def run_started(
        self, total: int, *, target: str, warehouse: str, project_total: int
    ) -> None:
        return

    def source_discovered(self, source_name: str, count: int) -> None:
        return

    def model_task(self, model_name: str, kind: str, total: int) -> ProgressTask:
        return _NullTask()

    def model_finished(
        self,
        model_name: str,
        kind: str,
        rows: int,
        duration: float,
        status: str | None,
        *,
        failed: bool,
    ) -> None:
        return

    def model_skipped(self, model_name: str, reason: str) -> None:
        return

    def run_finished(self, *, ok: int, errored: int, skipped: int) -> None:
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
    """Renders the run ledger on stderr, plus — at ``VERBOSE`` — source
    discovery lines, per-model progress bars, publication telemetry and
    forwarded log records around it."""

    def __init__(
        self, stream: TextIO, *, level: OutputLevel, bars: bool
    ) -> None:
        self._stream = stream
        self._level = level
        # Bars need a TTY; the ledger does not. Kept as a flag rather than
        # re-checked per task so one run cannot change its mind halfway.
        self._bars = bars
        self._total = 0
        self._completed = 0
        self._started_at: float | None = None
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

    def run_started(
        self, total: int, *, target: str, warehouse: str, project_total: int
    ) -> None:
        self._total = total
        self._completed = 0
        self._started_at = now_monotonic()
        noun = "model" if total == 1 else "models"
        # "3 of 12" only when a selector actually narrowed the run; otherwise
        # the second number says nothing and reads as though something was
        # dropped.
        scope = f"{total} of {project_total}" if total != project_total else f"{total}"
        # ASCII only: this lands on a Windows console whose code page is not
        # guaranteed to carry U+00B7, and a mojibake header is worse than a
        # plain one.
        self._echo(f"Running {scope} {noun} (target: {target}, {warehouse})")

    def source_discovered(self, source_name: str, count: int) -> None:
        # Discovery detail belongs to -v: at NORMAL the per-model ledger line
        # already reports what was processed, and a project with many sources
        # would otherwise push the ledger off the screen before it starts.
        if self._level < OutputLevel.VERBOSE:
            return
        # "selected", not "discovered": the runner passes the post-filter count
        # (issue #348), and the two differ under --source-filter. Matching the
        # log line's wording also keeps the captured and terminal channels
        # saying the same thing about the same number.
        self._echo(f"[source] {source_name}: {count:,} document(s) selected")

    def model_task(self, model_name: str, kind: str, total: int) -> ProgressTask:
        if not self._bars:
            return _NullTask()
        label = f"[{kind}] {model_name}"
        if total == 0:
            self._echo(f"{label}: 0 documents (nothing to process)")
        return _TerminalTask(label, total, self._stream, reporter=self)

    def model_finished(
        self,
        model_name: str,
        kind: str,
        rows: int,
        duration: float,
        status: str | None,
        *,
        failed: bool,
    ) -> None:
        self._completed += 1
        outcome = status.upper() if status else ("ERROR" if failed else "OK")
        self._echo(
            f"{self._counter()} {model_name:<24}{kind:<12}"
            f"{rows:>9,} rows{_format_duration(duration):>9}  {outcome}"
        )

    def model_skipped(self, model_name: str, reason: str) -> None:
        self._completed += 1
        self._echo(
            f"{self._counter()} {model_name:<24}{'-':<12}"
            f"{'-':>9}     {'-':>8}  SKIPPED ({reason})"
        )

    def run_finished(self, *, ok: int, errored: int, skipped: int) -> None:
        parts = [f"{ok} ok"]
        if errored:
            parts.append(f"{errored} error" + ("s" if errored != 1 else ""))
        if skipped:
            parts.append(f"{skipped} skipped")
        elapsed = (
            ""
            if self._started_at is None
            else f" in {_format_duration(now_monotonic() - self._started_at)}"
        )
        self._echo(f"Completed{elapsed}: {', '.join(parts)}")

    def _counter(self) -> str:
        """``[3/7]`` — completions, not launch order, so `--threads N` finishing
        out of order still counts up rather than jumping around."""
        width = len(str(self._total))
        return f"[{self._completed:>{width}}/{self._total}]"

    def publication(self, message: str) -> None:
        # Per-flush telemetry (issue #292) is `-v` detail: at NORMAL it would
        # put one line per flush between consecutive ledger lines, which is
        # exactly the wall of text the ledger exists to replace.
        if self._level < OutputLevel.VERBOSE:
            return
        # Fires inside the model's live progress bar, so it takes the deferral
        # path rather than writing over the redraw.
        self._defer_or_echo(f"[publish] {message}")

    def detail(self, message: str) -> None:
        """Render one already-formatted line from the forwarded log channel.

        No prefix: the record arrives pre-formatted by the logging handler,
        which supplies its own timestamp/level/logger shape. Reached only at
        VERBOSE, since that is the only level with a log handler installed."""
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


def reporter_is_active() -> bool:
    """True when something is rendering run events to the operator.

    The log channel asks this to decide whether an event that the reporter also
    renders should print from a log record too — see ``REPORTER_ECHO`` in
    ``logging_setup``.
    """
    return not isinstance(_reporter, _NullReporter)


def configure_progress(
    level: OutputLevel,
    *,
    bars_safe: bool = True,
    stream: TextIO | None = None,
) -> bool:
    """Install the reporter for ``level`` and report whether bars are live.

    Bars need ``VERBOSE``, a TTY, and ``bars_safe`` — concurrent models
    (``run --threads N``) each open their own ``click.progressbar`` on the one
    stderr and their redraws interleave, so that case takes the ledger and the
    plain log channel instead. The ledger itself is installed at ``NORMAL`` and
    above regardless of TTY: a CI log wants it too.

    Returns True when bars are live, which tells the caller to route the INFO
    log handler through the reporter so records defer past a bar instead of
    writing over it (issue #403).
    """
    target = stream if stream is not None else sys.stderr
    if level <= OutputLevel.QUIET:
        set_reporter(None)
        return False
    bars = bars_safe and level >= OutputLevel.VERBOSE and _is_tty(target)
    set_reporter(_TerminalReporter(target, level=level, bars=bars))
    return bars


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
