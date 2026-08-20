"""Bounded publication-state reconciliation (issue #153).

Reconciles one immutable upstream key stream against warehouse-owned
publication state without loading either complete domain into memory:

- Upstream rows are classified per batch (new / changed / unchanged) through
  bounded `fetch_state_subset` point lookups, so classification residency is
  proportional to one publication batch.
- Stale IDs — state rows whose keys no longer exist upstream — stream in
  strict record-key order through an absence-filtered `state_page_reader`,
  so delete discovery is complete (including for empty upstream input) and
  deterministic while residency stays proportional to one page.

This module owns the core-side validation of adapter paging: cross-page key
monotonicity, duplicate/empty-key rejection, and cursor progression. It
complements bounded upstream reads (issue #140), which cap the other memory
ceiling; the two are deliberately separate operations.
"""
from __future__ import annotations

from collections.abc import Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass

from .adapters.base import (
    AdapterError,
    StateAbsenceProbe,
    StatePageReader,
    StatePageRecord,
    StateScope,
    StateValue,
    WarehouseAdapter,
)


@dataclass(frozen=True)
class UpstreamRecord:
    """Key and input fingerprint of one upstream row awaiting publication."""

    record_key: str
    input_fingerprint: str


@dataclass(frozen=True)
class BatchClassification:
    """Deterministic per-batch merge outcome, in upstream batch order."""

    new: tuple[UpstreamRecord, ...]
    changed: tuple[UpstreamRecord, ...]
    unchanged: tuple[UpstreamRecord, ...]


def classify_batch(
    records: Sequence[UpstreamRecord],
    *,
    prior: Mapping[str, StateValue],
    code_version: str,
    force_publish: bool = False,
) -> BatchClassification:
    """Split one upstream batch against its scoped prior state.

    `prior` is the bounded subset lookup for exactly this batch's keys. A row
    is unchanged only when both its input fingerprint and the publishing code
    version match; `force_publish` republishes known rows (they stay
    classified by prior existence so callers can count inserts vs updates
    and revoke superseded governed rows)."""
    new: list[UpstreamRecord] = []
    changed: list[UpstreamRecord] = []
    unchanged: list[UpstreamRecord] = []
    for record in records:
        previous = prior.get(record.record_key)
        if previous is None:
            new.append(record)
        elif (
            not force_publish
            and previous.input_fingerprint == record.input_fingerprint
            and previous.code_version == code_version
        ):
            unchanged.append(record)
        else:
            changed.append(record)
    return BatchClassification(tuple(new), tuple(changed), tuple(unchanged))


def iter_validated_state_pages(
    reader: StatePageReader,
) -> Iterator[tuple[StatePageRecord, ...]]:
    """Drive a state page reader to exhaustion, enforcing stream invariants.

    Yields non-empty pages whose keys are strictly ascending across the whole
    stream. A page that repeats keys, moves backwards, or fails to progress
    its cursor aborts the reconciliation instead of silently skipping or
    repeating IDs."""
    cursor: str | None = None
    last_key: str | None = None
    while True:
        page = reader.fetch_page(cursor)
        if page.records:
            first_key = page.records[0].record_key
            if last_key is not None and first_key <= last_key:
                raise AdapterError(
                    "State pages are not strictly ordered across page boundaries"
                )
            last_key = page.records[-1].record_key
            yield page.records
        if page.next_cursor is None:
            return
        if not page.records:
            raise AdapterError(
                "State page reader returned an empty page that is not final"
            )
        if page.next_cursor == cursor:
            raise AdapterError("State page cursor did not advance between pages")
        cursor = page.next_cursor


class BoundedReconciler:
    """Bounded merge/diff between an upstream stream and one state scope."""

    def __init__(
        self,
        adapter: WarehouseAdapter,
        scope: StateScope,
        *,
        code_version: str,
        page_size: int,
    ) -> None:
        self._adapter = adapter
        self._scope = scope
        self._code_version = code_version
        self._page_size = page_size

    def prior_state_for(
        self, records: Sequence[UpstreamRecord]
    ) -> dict[str, StateValue]:
        return self._adapter.fetch_state_subset(
            self._scope, [record.record_key for record in records]
        )

    def classify(
        self,
        records: Sequence[UpstreamRecord],
        *,
        prior: Mapping[str, StateValue],
        force_publish: bool = False,
    ) -> BatchClassification:
        return classify_batch(
            records,
            prior=prior,
            code_version=self._code_version,
            force_publish=force_publish,
        )

    def iter_stale_pages(
        self, *, upstream_table: str, key_column: str
    ) -> Generator[tuple[StatePageRecord, ...], None, None]:
        """Stream state records whose keys are absent from the upstream
        relation, in deterministic ascending key order. Complete even when
        the upstream relation is empty: every scoped state row is stale."""
        probe = StateAbsenceProbe(table=upstream_table, key_column=key_column)
        with self._adapter.state_page_reader(
            self._scope, page_size=self._page_size, absent_from=probe
        ) as reader:
            yield from iter_validated_state_pages(reader)
