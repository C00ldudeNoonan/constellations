"""Turn CLI termination into an unwind while preventing new warehouse work."""

from __future__ import annotations

import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from types import FrameType


@dataclass
class _InterruptState:
    requested: bool = False
    deferred: int = 0


_active: _InterruptState | None = None


@contextmanager
def interrupt_on_termination() -> Iterator[None]:
    global _active
    if threading.current_thread() is not threading.main_thread() or _active is not None:
        yield
        return

    state = _InterruptState()

    def interrupt(_signum: int, _frame: FrameType | None) -> None:
        if state.requested:
            return  # Let the first interrupt finish unwinding.
        state.requested = True
        if not state.deferred:
            raise KeyboardInterrupt

    previous = {}
    for kind in (signal.SIGINT, signal.SIGTERM):
        handler = signal.getsignal(kind)
        if handler is not None:
            previous[kind] = handler
    try:
        _active = state
        for kind in previous:
            signal.signal(kind, interrupt)
        yield
    finally:
        state.deferred += 1
        try:
            for kind, handler in previous.items():
                signal.signal(kind, handler)
        finally:
            _active = None


def interruption_requested() -> bool:
    state = _active
    return state is not None and state.requested


def check_interruption() -> None:
    """Raise at a safe boundary, including in a worker thread."""
    if interruption_requested():
        raise KeyboardInterrupt


@contextmanager
def defer_interrupts() -> Iterator[None]:
    """Let an API call record its outcome before the caller checks for a signal."""
    state = _active
    if state is None or threading.current_thread() is not threading.main_thread():
        yield
        return
    state.deferred += 1
    try:
        yield
    finally:
        state.deferred -= 1
