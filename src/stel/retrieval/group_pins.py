"""Pin every selected search-group member before reading any collection."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass

from ..adapters.base import StateScope
from .coordination import QueryLease, ServingCoordinator, ServingNotReadyError


@dataclass(frozen=True, slots=True)
class GroupPinRequest:
    scope: StateScope
    legacy_scope: StateScope
    config_fingerprint: str


@contextmanager
def pin_group(
    coordinator: ServingCoordinator, requests: Sequence[GroupPinRequest]
) -> Iterator[tuple[QueryLease, ...]]:
    """Admit the whole group or release every acquired member pin.

    Validation after the last admission rejects a cutover during admission;
    validation after the read rejects a pin revoked by recovery. Individual
    member pins remain held throughout, so their publishers observe readers.
    """
    with ExitStack() as stack:
        leases: list[QueryLease] = []
        for request in requests:
            lease = coordinator.acquire_query(request.scope, legacy_scope=request.legacy_scope)
            stack.callback(coordinator.release_query, lease)
            if lease.config_fingerprint != request.config_fingerprint:
                raise ServingNotReadyError(
                    f"Search group member '{request.scope.model_name}' was published under "
                    "a different configuration; republish it"
                )
            leases.append(lease)
        for lease in leases:
            coordinator.validate_query(lease, require_active=True)
        yield tuple(leases)
        for lease in leases:
            coordinator.validate_query(lease)
