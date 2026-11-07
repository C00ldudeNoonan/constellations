"""A keyed warehouse read is cut into segments that all read as of one instant,
and a publish can continue a read from the point an earlier attempt reached
(issue #614).

A single BigQuery Storage session dies at six hours, and a publish over a large
corpus runs longer than that, so every retry re-read from the start. These tests
pin the two halves of the fix: the segments are one relation however long the
read takes, and a recorded point lets a later attempt skip what is already
published without reading it again.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pytest

from stel.adapters import bigquery as bigquery_module
from stel.adapters import create_adapter, parse_warehouse_config
from stel.adapters.base import (
    AdapterError,
    ReadPredicate,
    ReadPredicateOperator,
    SnapshotResumePoint,
)
from stel.adapters.bigquery import BigQueryAdapter, _predicate_restrictions

# Relative to now: a resume is refused once its point is older than the time
# travel window, so a fixed date would make these tests depend on the day.
_CLOCK = datetime.now(UTC).replace(microsecond=123456)
_KEYS = ["a", "b", "c", "d", "e"]
# Segments of two rows over five keys: [a, b], [c, d], [e]. The boundaries are
# the keys that open the second and third segments.
_SEGMENT_ROWS = 2
# A WHERE on the table read itself, as opposed to the boundary query's own.
_AS_OF_FILTER = re.compile(r"TIMESTAMP\('[^']*'\) WHERE ")
_RESTRICTION_CLAUSE = re.compile(r"(\w+) (>=|<=|!=|<|>|=) ('[^']*'|-?\d+)")


class _Row(tuple[Any, ...]):
    def values(self) -> tuple[Any, ...]:
        return tuple(self)


class _Job:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = [_Row(row) for row in rows]

    def result(self, **_kwargs: Any) -> list[_Row]:
        return list(self._rows)


class _Page:
    def __init__(self, batch: pa.RecordBatch) -> None:
        self._batch = batch

    def to_arrow(self) -> pa.RecordBatch:
        return self._batch


class _Storage:
    """The Storage Read surface a segmented read uses.

    Each session serves the rows its row restriction selects, and records the
    instant it was asked to read as of. The protobuf types are the real ones, so
    a wrong field name fails here rather than only against BigQuery.
    """

    def __init__(self, table: pa.Table) -> None:
        self.table = table
        self.transport = self
        self.sessions: list[SimpleNamespace] = []
        self._streams: dict[str, pa.Table] = {}

    def close(self) -> None:
        pass

    def create_read_session(
        self,
        *,
        parent: str,
        read_session: Any,
        max_stream_count: int,
        timeout: Any = None,
    ) -> Any:
        assert max_stream_count == 1
        restriction = read_session.read_options.row_restriction or None
        # proto-plus hands a Timestamp back as a datetime already.
        as_of = read_session.table_modifiers.snapshot_time
        self.sessions.append(SimpleNamespace(restriction=restriction, as_of=as_of))
        rows = self._select(restriction)
        name = f"stream-{len(self.sessions)}"
        self._streams[name] = rows
        return SimpleNamespace(
            arrow_schema=SimpleNamespace(serialized_schema=self.table.schema.serialize()),
            streams=[SimpleNamespace(name=name)] if rows.num_rows else [],
        )

    def read_rows(self, name: str, timeout: Any = None) -> Any:
        pages = [_Page(batch) for batch in self._streams[name].to_batches()]
        return SimpleNamespace(
            rows=lambda: SimpleNamespace(pages=iter(pages)),
            cancel=lambda: None,
        )

    def _select(self, restriction: str | None) -> pa.Table:
        if restriction is None:
            return self.table
        keep = [True] * self.table.num_rows
        for column, operator, literal in _RESTRICTION_CLAUSE.findall(restriction):
            value: Any = literal[1:-1] if literal.startswith("'") else int(literal)
            for index, cell in enumerate(self.table.column(column).to_pylist()):
                keep[index] = keep[index] and _COMPARE[operator](cell, value)
        return self.table.filter(pa.array(keep))


_COMPARE = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    "=": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


class _Client:
    """Answers the statements a keyed read runs before its first session."""

    def __init__(self, keys: list[str], filtered: list[str] | None = None) -> None:
        self.keys = sorted(keys)
        # The keys a predicated statement sees, when it carries a WHERE.
        self.filtered = sorted(filtered) if filtered is not None else self.keys
        self.clock = _CLOCK
        # The table's etag. It changes only when the table does, so it is the
        # generation identity a retry compares against (issue #508).
        self.etag = 'etag-1'
        self.statements: list[str] = []
        # How many bound parameters each statement carried, in step with them.
        self.parameter_counts: list[int] = []

    def get_table(self, _table_id: str) -> Any:
        return SimpleNamespace(
            schema=[
                SimpleNamespace(name="chunk_id", field_type="STRING"),
                SimpleNamespace(name="value", field_type="INTEGER"),
            ],
            etag=self.etag,
            modified=None,
            num_rows=len(self.keys),
        )

    def query(self, sql: str, job_config: Any = None, **_kwargs: Any) -> _Job:
        self.statements.append(sql)
        self.parameter_counts.append(len(job_config.query_parameters) if job_config else 0)
        if "CURRENT_TIMESTAMP()" in sql:
            return _Job([(self.clock,)])
        if "COUNT(DISTINCT" in sql:
            return _Job([(0, 0)])
        if "ROW_NUMBER()" in sql:
            keys = self.filtered if _AS_OF_FILTER.search(sql) else self.keys
            boundaries = [
                (key,)
                for position, key in enumerate(keys, start=1)
                if position > 1 and (position - 1) % _SEGMENT_ROWS == 0
            ]
            return _Job(boundaries)
        raise AssertionError(f"unexpected statement: {sql}")

    def close(self) -> None:
        pass


@pytest.fixture
def segments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bigquery_module, "_SEGMENT_ROWS", _SEGMENT_ROWS)


def _adapter(client: _Client) -> tuple[BigQueryAdapter, _Storage]:
    table = pa.table({"chunk_id": _KEYS, "value": list(range(len(_KEYS)))})
    storage = _Storage(table)
    config = parse_warehouse_config({"type": "bigquery", "project": "proj", "dataset": "ds"})
    adapter = create_adapter(config)
    assert isinstance(adapter, BigQueryAdapter)
    adapter._client = client
    adapter._bqstorage_client = storage
    return adapter, storage


def _keys(snapshot: Any) -> list[str]:
    return [key for batch in snapshot for key in batch.column("chunk_id").to_pylist()]


def test_the_read_is_cut_into_segments_that_all_read_as_of_one_instant(
    segments: None,
) -> None:
    adapter, storage = _adapter(_Client(_KEYS))

    with adapter.table_snapshot("chunks", key_column="chunk_id", batch_size=100) as snapshot:
        keys = _keys(snapshot)

    assert sorted(keys) == _KEYS
    assert [session.restriction for session in storage.sessions] == [
        "chunk_id < 'c'",
        "chunk_id >= 'c' AND chunk_id < 'e'",
        "chunk_id >= 'e'",
    ]
    assert {session.as_of for session in storage.sessions} == {_CLOCK}


def test_progress_moves_past_a_segment_only_once_its_rows_are_yielded(
    segments: None,
) -> None:
    adapter, _storage = _adapter(_Client(_KEYS))

    with adapter.table_snapshot("chunks", key_column="chunk_id", batch_size=100) as snapshot:
        batches = iter(snapshot)
        next(batches)
        # The first segment's rows are out, but the caller has not asked for
        # more, so the segment is not yet complete.
        assert snapshot.resume_point is not None
        assert snapshot.resume_point.completed == 0
        next(batches)
        assert snapshot.resume_point.completed == 1
        list(batches)
        assert snapshot.resume_point.completed == 3


def test_a_read_continues_from_a_recorded_point_without_rereading_its_segments(
    segments: None,
) -> None:
    first_client = _Client(_KEYS)
    first_adapter, _first_storage = _adapter(first_client)
    with first_adapter.table_snapshot(
        "chunks", key_column="chunk_id", batch_size=100
    ) as first:
        batches = iter(first)
        next(batches)
        next(batches)
        recorded = first.resume_point
    assert recorded is not None and recorded.completed == 1
    pinned = datetime.fromisoformat(recorded.snapshot_time)

    # A later attempt, after the table has moved on, opens a fresh snapshot and
    # is re-planned onto the recorded point.
    later_client = _Client(_KEYS)
    later_client.clock = _CLOCK + timedelta(hours=7)
    later_adapter, later_storage = _adapter(later_client)
    with later_adapter.table_snapshot(
        "chunks", key_column="chunk_id", batch_size=100
    ) as later:
        assert later.resume_from(recorded)
        keys = _keys(later)
        final_progress = later.resume_point

    assert keys == ["c", "d", "e"]
    assert final_progress is not None and final_progress.completed == 3
    # The recorded point counted the two rows it published; the resumed read
    # carries that count forward, so the total is the whole relation.
    assert recorded.rows == 2
    assert final_progress.rows == len(_KEYS)
    pinned_restrictions = [
        session.restriction
        for session in later_storage.sessions
        if session.as_of == pinned
    ]
    assert pinned_restrictions == ["chunk_id >= 'c' AND chunk_id < 'e'", "chunk_id >= 'e'"]


def test_the_generation_is_the_table_version_so_a_retry_over_an_unchanged_table_matches(
    segments: None,
) -> None:
    # The generation identity must not move with the read instant: a retry over an
    # unchanged table has to match the stamp a finished publish wrote (issue #508).
    first_adapter, _ = _adapter(_Client(_KEYS))
    later_client = _Client(_KEYS)
    later_client.clock = _CLOCK + timedelta(hours=1)
    later_adapter, _ = _adapter(later_client)
    with first_adapter.table_snapshot("chunks", key_column="chunk_id") as first:
        first_generation = first.generation_fingerprint
    with later_adapter.table_snapshot("chunks", key_column="chunk_id") as later:
        assert later.generation_fingerprint == first_generation


def test_a_resumed_read_keeps_the_generation_it_was_planned_against(
    segments: None,
) -> None:
    # Its rows are the table as of the recorded instant, so it must not claim
    # the table's current version. A retry would otherwise skip a read it needs.
    planned_client = _Client(_KEYS)
    planned_adapter, _ = _adapter(planned_client)
    with planned_adapter.table_snapshot(
        "chunks", key_column="chunk_id", batch_size=100
    ) as planned:
        batches = iter(planned)
        next(batches)
        next(batches)
        recorded = planned.resume_point
        planned_generation = planned.generation_fingerprint
    assert recorded is not None

    moved_client = _Client(_KEYS)
    moved_client.etag = "etag-2"
    moved_adapter, _ = _adapter(moved_client)
    with moved_adapter.table_snapshot(
        "chunks", key_column="chunk_id", batch_size=100
    ) as moved:
        assert moved.resume_from(recorded)
        assert moved.generation_fingerprint == planned_generation


def test_a_point_too_old_to_time_travel_to_is_not_resumed(segments: None) -> None:
    adapter, _storage = _adapter(_Client(_KEYS))
    stale = SnapshotResumePoint(
        snapshot_time=(_CLOCK - timedelta(days=3)).isoformat(),
        key_type="STRING",
        boundaries=("c", "e"),
        completed=1,
        rows=250_000,
        generation="generation-a",
        predicates="",
    )

    with adapter.table_snapshot("chunks", key_column="chunk_id", batch_size=100) as snapshot:
        assert not snapshot.resume_from(stale)
        keys = _keys(snapshot)

    # Refused, so the read starts from the beginning, as it would with no point.
    assert sorted(keys) == _KEYS


def test_a_read_already_under_way_is_not_re_planned(segments: None) -> None:
    adapter, _storage = _adapter(_Client(_KEYS))
    with adapter.table_snapshot("chunks", key_column="chunk_id", batch_size=100) as snapshot:
        batches = iter(snapshot)
        next(batches)
        point = SnapshotResumePoint(
            snapshot_time=_CLOCK.isoformat(),
            key_type="STRING",
            boundaries=("c", "e"),
            completed=2,
            rows=500_000,
            generation="generation-a",
            predicates="",
        )
        assert not snapshot.resume_from(point)


def test_a_resume_point_survives_its_stamp_and_an_unreadable_one_reads_fresh() -> None:
    point = SnapshotResumePoint(
        snapshot_time=_CLOCK.isoformat(),
        key_type="INT64",
        boundaries=("10", "20"),
        completed=2,
        rows=500_000,
        generation="generation-a",
        predicates="",
    )

    assert SnapshotResumePoint.from_stamp(point.to_stamp()) == point
    assert SnapshotResumePoint.from_stamp(None) is None
    assert SnapshotResumePoint.from_stamp("not json") is None


# ─── predicated reads are segmented too (issue #754) ─────────────────────────
#
# A partition member reads only its own upstream rows, which is a predicated
# read. It used to go through a query job with no resumable segments, so a
# failure late in a 3.6M-row read restarted it.

_FROM_THE_SECOND_ROW = ReadPredicate("value", ReadPredicateOperator.GREATER_THAN_OR_EQUAL, 1)
# What that predicate keeps: b, c, d, e. Two-row segments over four keys.
_KEPT = ["b", "c", "d", "e"]


def _predicated(adapter: BigQueryAdapter, predicate: ReadPredicate) -> Any:
    return adapter.table_snapshot(
        "chunks", key_column="chunk_id", batch_size=100, predicate=predicate
    )


def test_a_predicated_read_is_segmented_over_the_rows_the_predicate_keeps(
    segments: None,
) -> None:
    client = _Client(_KEYS, filtered=_KEPT)
    adapter, storage = _adapter(client)

    with _predicated(adapter, _FROM_THE_SECOND_ROW) as snapshot:
        keys = _keys(snapshot)

    # The same rows as the unpredicated read filtered afterwards.
    assert sorted(keys) == _KEPT
    # Boundaries are cut over the filtered relation, so the segments hold two
    # rows each; cut over the whole table they would be [b], [c, d], [e].
    assert [session.restriction for session in storage.sessions] == [
        "value >= 1 AND chunk_id < 'd'",
        "value >= 1 AND chunk_id >= 'd'",
    ]
    assert {session.as_of for session in storage.sessions} == {_CLOCK}
    # The key-domain check and the boundaries both run at the pinned instant,
    # over the predicated rows, and no statement reads the payload.
    probes = [sql for sql in client.statements if "CURRENT_TIMESTAMP()" not in sql]
    assert len(probes) == 2
    for statement in probes:
        assert _AS_OF_FILTER.search(statement)
    # The `?` in each is bound: unbound it is a syntax error at BigQuery.
    assert [
        count
        for sql, count in zip(client.statements, client.parameter_counts, strict=True)
        if "CURRENT_TIMESTAMP()" not in sql
    ] == [1, 1]


def test_a_predicated_read_resumes_after_an_interrupted_segment(segments: None) -> None:
    first_adapter, _ = _adapter(_Client(_KEYS, filtered=_KEPT))
    with _predicated(first_adapter, _FROM_THE_SECOND_ROW) as first:
        batches = iter(first)
        next(batches)
        next(batches)
        recorded = first.resume_point
    assert recorded is not None and recorded.completed == 1
    assert recorded.predicates != ""

    # A later attempt, after the table moved on and the clock with it.
    later_client = _Client(_KEYS, filtered=_KEPT)
    later_client.clock = _CLOCK + timedelta(hours=7)
    later_adapter, later_storage = _adapter(later_client)
    with _predicated(later_adapter, _FROM_THE_SECOND_ROW) as later:
        assert later.resume_from(recorded)
        keys = _keys(later)
        final = later.resume_point

    # Only the segment the first attempt did not finish, at its pinned instant.
    assert keys == ["d", "e"]
    assert final is not None and final.completed == 2
    assert final.rows == len(_KEPT)
    pinned = datetime.fromisoformat(recorded.snapshot_time)
    resumed = [s.restriction for s in later_storage.sessions if s.as_of == pinned]
    assert resumed == ["value >= 1 AND chunk_id >= 'd'"]


def test_a_point_cut_under_another_filter_is_not_resumed(segments: None) -> None:
    """The boundaries and the completed count describe one filtered relation.
    Continuing them under a different filter would skip rows this read owes."""
    first_adapter, _ = _adapter(_Client(_KEYS, filtered=_KEPT))
    with _predicated(first_adapter, _FROM_THE_SECOND_ROW) as first:
        batches = iter(first)
        next(batches)
        next(batches)
        recorded = first.resume_point
    assert recorded is not None

    other = ReadPredicate("value", ReadPredicateOperator.GREATER_THAN_OR_EQUAL, 2)
    adapter, _ = _adapter(_Client(_KEYS, filtered=["c", "d", "e"]))
    with _predicated(adapter, other) as snapshot:
        assert not snapshot.resume_from(recorded)
    # And an unfiltered read does not take a filtered point either.
    adapter, _ = _adapter(_Client(_KEYS))
    with adapter.table_snapshot("chunks", key_column="chunk_id", batch_size=100) as snapshot:
        assert not snapshot.resume_from(recorded)


def test_a_stamp_written_before_predicates_were_segmented_reads_as_unfiltered() -> None:
    stamp = SnapshotResumePoint(
        snapshot_time=_CLOCK.isoformat(),
        key_type="STRING",
        boundaries=("c",),
        completed=1,
        rows=2,
        generation="generation-a",
        predicates="",
    ).to_stamp()
    legacy = stamp.replace(',"predicates":""', "")
    assert "predicates" not in legacy

    point = SnapshotResumePoint.from_stamp(legacy)

    assert point is not None and point.predicates == ""


def test_a_predicate_that_cannot_be_a_literal_keeps_the_query_path(
    segments: None,
) -> None:
    """A control character cannot be written into a restriction safely, so the
    read is not segmented: it opens no pinned session and plans no boundaries."""
    client = _Client(_KEYS)
    adapter, storage = _adapter(client)
    predicate = ReadPredicate("chunk_id", ReadPredicateOperator.EQUAL, "a\nb")

    with pytest.raises(AdapterError):
        # The fake serves no payload query, which is the point: it got that far.
        with _predicated(adapter, predicate):
            pass

    assert storage.sessions == []
    assert not any("ROW_NUMBER()" in sql for sql in client.statements)
    assert not any("CURRENT_TIMESTAMP()" in sql for sql in client.statements)


_TYPES = {
    "name": "STRING",
    "n": "INTEGER",
    "x": "FLOAT",
    "flag": "BOOLEAN",
    "day": "DATE",
    "at": "TIMESTAMP",
    "local": "DATETIME",
}
_EQ = ReadPredicateOperator.EQUAL
_PLUS_TWO = timezone(timedelta(hours=2))


@pytest.mark.parametrize(
    ("predicate", "expected"),
    [
        (ReadPredicate("name", _EQ, "it's"), ["name = 'it\\'s'"]),
        (ReadPredicate("name", _EQ, "a\\b"), ["name = 'a\\\\b'"]),
        (ReadPredicate("n", ReadPredicateOperator.LESS_THAN, 7), ["n < 7"]),
        (ReadPredicate("x", ReadPredicateOperator.GREATER_THAN, 1.5), ["x > 1.5"]),
        (ReadPredicate("x", ReadPredicateOperator.GREATER_THAN, 2), ["x > 2"]),
        (ReadPredicate("flag", _EQ, True), ["flag = TRUE"]),
        (ReadPredicate("day", _EQ, date(2026, 1, 2)), ["day = DATE '2026-01-02'"]),
        (
            ReadPredicate("at", _EQ, datetime(2026, 1, 2, 3, 4, 5)),
            ["at = TIMESTAMP '2026-01-02 03:04:05+00:00'"],
        ),
        (
            ReadPredicate("at", _EQ, datetime(2026, 1, 2, 3, 4, 5, tzinfo=_PLUS_TWO)),
            ["at = TIMESTAMP '2026-01-02 01:04:05+00:00'"],
        ),
        (
            ReadPredicate("local", _EQ, datetime(2026, 1, 2, 3, 4, 5)),
            ["local = DATETIME '2026-01-02 03:04:05'"],
        ),
        (ReadPredicate("name", ReadPredicateOperator.IN, ("a", "b")), ["name IN ('a', 'b')"]),
        (ReadPredicate("n", ReadPredicateOperator.NOT_IN, (1, 2)), ["n NOT IN (1, 2)"]),
        (ReadPredicate("name", ReadPredicateOperator.IS_NULL), ["name IS NULL"]),
        (ReadPredicate("name", ReadPredicateOperator.IS_NOT_NULL), ["name IS NOT NULL"]),
    ],
)
def test_predicates_render_as_the_literals_the_bound_query_would_select(
    predicate: ReadPredicate, expected: list[str]
) -> None:
    assert _predicate_restrictions([predicate], _TYPES) == expected


@pytest.mark.parametrize(
    "predicate",
    [
        # A pairing the column's type would not take as a parameter either: the
        # query path keeps whatever it did with it.
        ReadPredicate("n", _EQ, "7"),
        ReadPredicate("n", _EQ, True),
        ReadPredicate("name", _EQ, 7),
        ReadPredicate("n", _EQ, 1.5),
        ReadPredicate("day", _EQ, datetime(2026, 1, 2)),
        ReadPredicate("local", _EQ, datetime(2026, 1, 2, tzinfo=UTC)),
        ReadPredicate("name", ReadPredicateOperator.IN, ("a", "b\tc")),
        ReadPredicate("missing", _EQ, 1),
        ReadPredicate("bad name", ReadPredicateOperator.IS_NULL),
    ],
)
def test_a_predicate_with_no_safe_literal_is_not_written_as_a_restriction(
    predicate: ReadPredicate,
) -> None:
    assert _predicate_restrictions([predicate], _TYPES) is None


def test_a_predicate_value_is_bound_in_every_statement_and_only_restricts_the_session(
    segments: None,
) -> None:
    """The probes bind the value, as the query path does, so it never appears in
    statement text. It does appear in the read session's row restriction, which
    takes literals and has no parameters; that is the one place it is written."""
    sentinel = "zzz-sentinel"
    client = _Client(_KEYS)
    adapter, storage = _adapter(client)
    predicate = ReadPredicate("chunk_id", ReadPredicateOperator.NOT_EQUAL, sentinel)

    with _predicated(adapter, predicate) as snapshot:
        list(snapshot)

    assert client.statements
    assert not any(sentinel in sql for sql in client.statements)
    assert all(sentinel in str(session.restriction) for session in storage.sessions)
