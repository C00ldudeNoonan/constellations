"""Long-running-command progress feedback (issue #268).

Verbose mode (``-v``, ``DBT_ML_VERBOSE=1``) attaches a live per-model progress
bar to stderr for the extraction loop and prints one line per source-discovery
completion. Non-TTY callers (Dagster captures, redirected stderr) fall back to
the plain ``log.info`` lines already emitted by discovery/extraction/runner, so
the same wiring works in both places without duplicating output on a terminal.

The active reporter is module-global to mirror the existing ``logging`` pattern:
extraction and the runner query :func:`get_reporter` instead of threading a
reporter argument through every executor.
"""

from __future__ import annotations

import sys
import time
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


class _TerminalTask:
    """Wraps ``click.progressbar`` and skips the render entirely when total is 0
    so an empty extraction doesn't leave a stale header on the screen."""

    def __init__(self, label: str, total: int, stream: TextIO) -> None:
        self._total = total
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


class _TerminalReporter:
    """Terminal-friendly reporter. Per-model progress bars go on stderr; source
    discovery and model-finished status get one-line summaries above the bar."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def _echo(self, message: str) -> None:
        click.echo(message, file=self._stream)

    def source_discovered(self, source_name: str, count: int) -> None:
        self._echo(f"[source] {source_name}: discovered {count:,} object(s)")

    def model_task(self, model_name: str, kind: str, total: int) -> ProgressTask:
        label = f"[{kind}] {model_name}"
        if total == 0:
            self._echo(f"{label}: 0 documents (nothing to process)")
        return _TerminalTask(label, total, self._stream)

    def model_finished(
        self, model_name: str, rows: int, duration: float, status: str | None
    ) -> None:
        status_suffix = f" [{status}]" if status else ""
        self._echo(
            f"[done]   {model_name}: {rows:,} row(s) in "
            f"{_format_duration(duration)}{status_suffix}"
        )


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
