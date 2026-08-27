"""Checkpoint ordering for stages that publish as they go (issue #401).

Extraction, transforms, embeds, and llm maps all publish in windows and
advance state per window. Each of them has to get the same three things right,
and each of them getting it right *separately* is how a subtle ordering bug
ends up in one stage and not the others — which is exactly what happened when
embed and llm grew flushing independently and both reproduced the same
first-flush state hazard.

So the rules live here, once:

1. **A full rebuild clears state before it writes anything.** The dangerous
   direction is state that outlives its rows. If the opening flush replaces
   the target and the state reset then fails, the old fingerprints survive
   against a table that now holds one flush — and the next run reads that
   state, decides those rows are done, and leaves the target permanently
   partial while reporting success. Clearing first means a failure can only
   leave state *behind* the target, which costs a re-run and never silently
   drops rows.

2. **State advances only after the write lands.** An interrupted run must
   never record a row it did not write.

3. **A publication failure is reported without the warehouse's own words.**
   Adapter exception text can carry source values and SQL, and a `RunError`
   is persisted into `run_results.json`. A publication failure must not be
   dressed up as a provider failure either: the provider error path exists to
   sanitize provider text, and routing a DuckDB conversion error through it
   passes the raw message straight to the fallback.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

from ..adapters import AdapterError, StateRecord, StateScope, WarehouseAdapter
from .contracts import RunError


class FlushPublisher:
    """Publishes one window at a time in the only safe order.

    `use_full` means this run rebuilds the target: the first window replaces
    it and later windows merge into it. That makes a full refresh no longer a
    single atomic swap, which is the deliberate trade behind flushing at all —
    an interrupted rebuild leaves a partially rebuilt table that the next run
    completes from state, rather than a corpus that cannot be rebuilt.
    """

    def __init__(
        self,
        adapter: WarehouseAdapter,
        *,
        model_name: str,
        state_scope: StateScope,
        use_full: bool,
    ) -> None:
        self._adapter = adapter
        self._model_name = model_name
        self._state_scope = state_scope
        self._use_full = use_full
        self._first = True
        self._published = False
        self.rows_written = 0

    @property
    def first_publication(self) -> bool:
        return self._first

    @property
    def published_any(self) -> bool:
        return self._published

    def publish(
        self,
        *,
        write_full: Callable[[], int],
        write_incremental: Callable[[], int],
        state_records: Sequence[StateRecord],
        advances_state_itself: bool = False,
    ) -> None:
        """Write one window and record its rows as processed.

        `advances_state_itself` is for adapter calls that take the state
        records and apply them in the same transaction as the rows —
        `replace_children` does. Advancing state again afterwards would be
        harmless but redundant, and pretending the caller had done it would
        hide which path owns the ordering.
        """
        replacing = self._use_full and self._first
        if replacing:
            # Rule 1: clear before writing, so a failure can only leave state
            # behind the target, never ahead of it.
            self._run(lambda: self._adapter.replace_state(self._state_scope, []))
        written = self._run(write_full if replacing else write_incremental)
        self.rows_written += written
        if not advances_state_itself and state_records:
            # Rule 2: only after the write landed.
            self._run(
                lambda: self._adapter.upsert_state(
                    self._state_scope, list(state_records)
                )
            )
        self._first = False
        self._published = True

    def _run[T](self, action: Callable[[], T]) -> T:
        try:
            return action()
        except AdapterError as error:
            # The adapter's own error surface is already the sanitized one.
            raise RunError(str(error)) from None
        except Exception as error:
            # Rule 3: never the exception's text. A warehouse driver's message
            # can quote the offending row and the statement that touched it.
            raise RunError(
                f"Publishing model '{self._model_name}' failed "
                f"({type(error).__name__}); the warehouse rejected the write. "
                "Rows published by earlier flushes are retained and their "
                "state recorded."
            ) from None
