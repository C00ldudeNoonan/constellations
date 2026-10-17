"""Cooperatively drained session pins; a clock never revokes a reader (#541).

Only the pin's owner deletes it, after its last local holder has finished.
The idle timeout bounds retention by a healthy process, not the lifetime of
the durable lease. A killed or partitioned owner still needs explicit recovery.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from threading import Event, RLock, Thread
from time import monotonic

from ..adapters.base import AdapterError, StateScope, WarehouseAdapter
from ..timing import PhaseTimings
from .coordination import (
    QueryLease,
    ServingBusyError,
    ServingCoordinationError,
    ServingCoordinator,
)

log = logging.getLogger(__name__)
IDLE_PIN_SECONDS = 30.0
PIN_SWEEP_SECONDS = 5.0


@dataclass
class _Pin:
    lease: QueryLease
    holders: int
    idle_since: float
    retiring: bool = False


class SessionQueryPins:
    """One durable pin per scope, shared only across operator queries.

    The warehouse callback counts maintenance as an operation on the session's
    held connection, so reconnect and close cannot invalidate its adapter.
    The lock covers admission/release, never embedding or store I/O.
    """

    def __init__(
        self, warehouse: Callable[[], AbstractContextManager[WarehouseAdapter]]
    ) -> None:
        self._warehouse = warehouse
        self._pins: dict[StateScope, _Pin] = {}
        self._lock = RLock()
        self._stopped = Event()
        self._worker = Thread(target=self._run, name="stel-query-pins", daemon=True)
        self._worker.start()

    @contextmanager
    def query(
        self, coordinator: ServingCoordinator, scope: StateScope, *, legacy_scope: StateScope,
        timings: PhaseTimings,
    ) -> Iterator[QueryLease]:
        with timings.phase("lease"), self._lock:
            pin, lease = self._admit(coordinator, scope, legacy_scope=legacy_scope)
            pin.holders += 1
        succeeded = False
        try:
            yield lease
            succeeded = True
        finally:
            self._return(coordinator, scope, pin, succeeded=succeeded, timings=timings)

    def _return(
        self, coordinator: ServingCoordinator, scope: StateScope, pin: _Pin, *,
        succeeded: bool, timings: PhaseTimings,
    ) -> None:
        with timings.phase("lease"), self._lock:
            pin.holders -= 1
            pin.idle_since = monotonic()
            pin.retiring |= not succeeded or self._stopped.is_set()
            if pin.retiring and pin.holders == 0:
                self._release(coordinator, scope, pin)

    def _admit(
        self, coordinator: ServingCoordinator, scope: StateScope, *, legacy_scope: StateScope
    ) -> tuple[_Pin, QueryLease]:
        if self._stopped.is_set():
            raise ServingBusyError("The search session is closing; retry with a new session")
        pin = self._pins.get(scope)
        if pin is not None:
            return pin, self._refresh(coordinator, scope, pin)
        lease = coordinator.acquire_query(scope, legacy_scope=legacy_scope)
        pin = _Pin(lease=lease, holders=0, idle_since=monotonic())
        self._pins[scope] = pin
        return pin, lease

    def _refresh(
        self, coordinator: ServingCoordinator, scope: StateScope, pin: _Pin
    ) -> QueryLease:
        if pin.retiring:
            raise ServingBusyError("Query readers are draining; retry the query shortly")
        try:
            return coordinator.refresh_query(pin.lease)
        except ServingCoordinationError:
            pin.retiring = True
            if pin.holders == 0:
                self._release(coordinator, scope, pin)
            raise

    def _release(self, coordinator: ServingCoordinator, scope: StateScope, pin: _Pin) -> None:
        # Keep a failed DELETE registered and non-reusable, so maintenance
        # can retry it on the session's replacement connection.
        pin.retiring = True
        coordinator.release_query(pin.lease)
        del self._pins[scope]

    def sweep(self, *, closing: bool = False) -> None:
        """Release idle pins only; even shutdown cannot unpin a live worker."""
        with self._lock:
            idle = [
                (scope, pin) for scope, pin in self._pins.items()
                if pin.holders == 0
                and (closing or pin.retiring or monotonic() - pin.idle_since >= IDLE_PIN_SECONDS)
            ]
            if not idle:
                return
            self._release_idle(idle)

    def _release_idle(self, idle: list[tuple[StateScope, _Pin]]) -> None:
        with self._warehouse() as adapter:
            coordinator = ServingCoordinator(adapter, ensure_schema=False)
            for scope, pin in idle:
                self._release(coordinator, scope, pin)

    def _run(self) -> None:
        while not self._stopped.wait(PIN_SWEEP_SECONDS):
            try:
                self.sweep()
            except (AdapterError, ServingCoordinationError):
                # Background warehouse I/O is an error boundary. Never log
                # native exception text; failed releases remain registered.
                log.warning("Session query-pin cleanup failed; it will be retried")

    def close(self) -> None:
        self._stopped.set()
        self._worker.join(timeout=PIN_SWEEP_SECONDS)
        try:
            self.sweep(closing=True)
        except (AdapterError, ServingCoordinationError):
            log.warning("Session query pins could not be released; serving recovery may be needed")
