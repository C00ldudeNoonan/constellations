from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import duckdb
import polars as pl
import pyarrow as pa
import pytest

from stel.adapters import (
    AdapterCapabilityError,
    AdapterError,
    ReadOrdering,
    ReadPredicate,
    ReadPredicateOperator,
    TableReadRequest,
    TableReadSnapshot,
    TableSnapshotGenerationChangedError,
    WarehouseCapability,
    create_adapter,
    parse_warehouse_config,
)
from stel.adapters.bigquery import BigQueryAdapter
from stel.adapters.duckdb import DuckDBAdapter, _duckdb_arrow_batches


def _duckdb_config(path: Path) -> Any:
    return parse_warehouse_config({"type": "duckdb", "path": str(path), "schema": "testns"})


def _assert_sentinel_absent_from_error(error: BaseException, sentinel: str) -> None:
    assert sentinel not in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None
    traceback = error.__traceback__
    while traceback is not None:
        if "/src/stel/" in traceback.tb_frame.f_code.co_filename:
            assert sentinel not in repr(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next


def _sanitized_native_cause(error: BaseException, sentinel: str) -> AdapterError:
    assert sentinel not in str(error)
    assert sentinel not in repr(error)
    assert error.__context__ is None
    cause = error.__cause__
    assert isinstance(cause, AdapterError)
    assert sentinel not in str(cause)
    assert sentinel not in repr(cause)
    assert cause.__traceback__ is None
    assert cause.__context__ is None
    return cause


def test_duckdb_streams_projected_filtered_batches_from_one_snapshot(
    tmp_path: Path,
) -> None:
    path = tmp_path / "stream.duckdb"
    with create_adapter(_duckdb_config(path)) as adapter:
        adapter.materialize_full(
            "records",
            pl.DataFrame(
                {
                    "record_id": ["a", "b", "c", "d", "e"],
                    "tenant": ["red", "blue", "red", "blue", "red"],
                    "value": [1, 2, 3, 4, 5],
                    "unused": [10, 20, 30, 40, 50],
                }
            ),
        )
        predicate = (
            ReadPredicate("tenant", ReadPredicateOperator.EQUAL, "red"),
            ReadPredicate("value", ReadPredicateOperator.GREATER_THAN, 1),
        )

        with adapter.table_snapshot(
            "records",
            columns=("record_id", "value"),
            batch_size=1,
            predicate=predicate,
            key_column="record_id",
        ) as snapshot:
            batches = list(snapshot)
            assert snapshot.schema == pa.schema(
                [pa.field("record_id", pa.string()), pa.field("value", pa.int64())]
            )
            assert snapshot.ordering is ReadOrdering.UNSPECIFIED
            assert re.fullmatch(r"[0-9a-f]{32}", snapshot.fingerprint)
            assert [batch.to_pydict() for batch in batches] == [
                {"record_id": ["c"], "value": [3]},
                {"record_id": ["e"], "value": [5]},
            ]
            assert snapshot.generation_fingerprint is not None
            assert re.fullmatch(
                r"[0-9a-f]{32}", snapshot.generation_fingerprint
            )

        assert snapshot.closed


def test_duckdb_unconsumed_snapshot_releases_the_database_file(tmp_path: Path) -> None:
    path = tmp_path / "unconsumed.duckdb"
    with create_adapter(_duckdb_config(path)) as adapter:
        adapter.materialize_full(
            "records",
            pl.DataFrame({"record_id": ["a", "b", "c"], "value": [1, 2, 3]}),
        )
        with adapter.table_snapshot("records", batch_size=1):
            pass  # opened for its schema only; never iterated

    # An unexhausted Arrow reader used to keep the database file pinned, so this
    # read-write re-open failed on Windows with "used by another process".
    connection = duckdb.connect(str(path))
    try:
        assert connection.execute("SELECT 1").fetchone() == (1,)
    finally:
        connection.close()


def test_duckdb_snapshot_is_immutable_during_concurrent_change(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.duckdb"
    with create_adapter(_duckdb_config(path)) as adapter:
        adapter.materialize_full(
            "records",
            pl.DataFrame({"record_id": ["a", "b"], "value": [1, 2]}),
        )
        with pytest.raises(AdapterError, match="changed during"):
            with adapter.table_snapshot(
                "records", batch_size=1, key_column="record_id"
            ) as snapshot:
                batches = iter(snapshot)
                assert next(batches).to_pydict() == {
                    "record_id": ["a"],
                    "value": [1],
                }
                writer = duckdb.connect(str(path))
                try:
                    writer.execute(
                        "UPDATE testns.records SET value = 9 WHERE record_id = 'b'"
                    )
                finally:
                    writer.close()
                assert [batch.to_pydict() for batch in batches] == [
                    {"record_id": ["b"], "value": [2]}
                ]

        assert adapter.read_table("records")["value"].to_list() == [1, 9]


def test_duckdb_empty_snapshot_retains_projected_schema(tmp_path: Path) -> None:
    with create_adapter(_duckdb_config(tmp_path / "empty.duckdb")) as adapter:
        adapter.materialize_full(
            "records",
            pl.DataFrame(schema={"record_id": pl.String, "value": pl.Int64}),
        )

        with adapter.table_snapshot(
            "records", columns=("record_id",), key_column="record_id"
        ) as snapshot:
            assert list(snapshot) == []
            assert snapshot.schema == pa.schema([pa.field("record_id", pa.string())])


def test_duckdb_snapshot_rejects_missing_projected_or_predicate_columns(
    tmp_path: Path,
) -> None:
    with create_adapter(_duckdb_config(tmp_path / "missing.duckdb")) as adapter:
        adapter.materialize_full("records", pl.DataFrame({"record_id": ["a"]}))

        with pytest.raises(AdapterError, match=r"missing column.*missing"):
            with adapter.table_snapshot("records", columns=("missing",)):
                pass
        with pytest.raises(AdapterError, match=r"missing column.*tenant"):
            with adapter.table_snapshot(
                "records",
                predicate=ReadPredicate("tenant", ReadPredicateOperator.EQUAL, "redacted-value"),
            ):
                pass


def test_duckdb_native_predicate_failure_does_not_retain_value(
    tmp_path: Path,
) -> None:
    sentinel = "sensitive-predicate-value"
    with create_adapter(_duckdb_config(tmp_path / "safe-error.duckdb")) as adapter:
        adapter.materialize_full("records", pl.DataFrame({"value": [1]}))

        with pytest.raises(AdapterError, match="could not be opened") as exc_info:
            with adapter.table_snapshot(
                "records",
                predicate=ReadPredicate("value", ReadPredicateOperator.EQUAL, sentinel),
            ):
                pass

    cause = _sanitized_native_cause(exc_info.value, sentinel)
    assert cause.args[0].startswith("Native adapter error type: ")


def test_duckdb_snapshot_open_and_generation_failures_sanitize_causes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = "diagnostic-only-native-error"

    def fail_predicates(*_args: Any) -> tuple[str, list[Any]]:
        raise RuntimeError(sentinel)

    with create_adapter(_duckdb_config(tmp_path / "causes.duckdb")) as adapter:
        assert isinstance(adapter, DuckDBAdapter)
        adapter.materialize_full("records", pl.DataFrame({"record_id": ["a"]}))
        monkeypatch.setattr("stel.adapters.duckdb._duckdb_read_predicates", fail_predicates)

        with pytest.raises(AdapterError, match="could not be opened") as open_error:
            with adapter.table_snapshot("records"):
                pass
        assert (
            str(_sanitized_native_cause(open_error.value, sentinel))
            == "Native adapter error type: RuntimeError"
        )

        with pytest.raises(AdapterError, match="generation could not be validated") as digest_error:
            adapter._current_table_digest(TableReadRequest("records", None, 1, (), None))
        assert (
            str(_sanitized_native_cause(digest_error.value, sentinel))
            == "Native adapter error type: RuntimeError"
        )


def test_duckdb_snapshot_batch_failure_sanitizes_cause() -> None:
    sentinel = "diagnostic-only-arrow-error"

    class FailingReader:
        schema = pa.schema([pa.field("value", pa.int64())])

        def __init__(self) -> None:
            self.closed = False

        def __iter__(self) -> FailingReader:
            return self

        def __next__(self) -> pa.RecordBatch:
            raise RuntimeError(sentinel)

        def close(self) -> None:
            self.closed = True

    reader = FailingReader()
    with pytest.raises(AdapterError, match="batch read failed") as exc_info:
        list(_duckdb_arrow_batches(cast(pa.RecordBatchReader, reader), lambda _digest: None))

    assert (
        str(_sanitized_native_cause(exc_info.value, sentinel))
        == "Native adapter error type: RuntimeError"
    )
    assert reader.closed


@pytest.mark.parametrize(
    "frame",
    [
        pl.DataFrame({"record_id": [None], "value": [1]}),
        pl.DataFrame(
            {"record_id": ["sensitive-row-value", "sensitive-row-value"], "value": [1, 2]}
        ),
    ],
)
def test_duckdb_key_domain_preflight_fails_before_yielding_rows(
    tmp_path: Path, frame: pl.DataFrame
) -> None:
    with create_adapter(_duckdb_config(tmp_path / "keys.duckdb")) as adapter:
        adapter.materialize_full("records", frame)

        with pytest.raises(AdapterError, match="key domain is invalid") as exc_info:
            with adapter.table_snapshot("records", key_column="record_id"):
                pytest.fail("invalid key domain opened a consumer-visible snapshot")

    assert "sensitive-row-value" not in str(exc_info.value)


def test_streaming_read_request_validation_and_capability_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = create_adapter(_duckdb_config(tmp_path / "guard.duckdb"))
    monkeypatch.setattr(type(adapter), "capabilities", classmethod(lambda cls: frozenset()))

    with pytest.raises(AdapterCapabilityError, match="streaming_tabular_reads"):
        with adapter.table_snapshot("records"):
            pass

    with pytest.raises(AdapterError, match="batch_size"):
        TableReadRequest("records", None, 0, (), None)
    with pytest.raises(AdapterError, match="duplicate names"):
        TableReadRequest("records", ("id", "id"), 1, (), None)
    with pytest.raises(AdapterError, match="key_column must be included"):
        TableReadRequest("records", ("value",), 1, (), "id")

    monkeypatch.setattr(
        type(adapter),
        "capabilities",
        classmethod(lambda cls: frozenset({WarehouseCapability.STREAMING_TABULAR_READS})),
    )
    with pytest.raises(AdapterCapabilityError, match="tabular_predicate_pushdown"):
        with adapter.table_snapshot(
            "records",
            predicate=ReadPredicate("id", ReadPredicateOperator.EQUAL, "safe"),
        ):
            pass


def test_read_predicate_repr_redacts_value() -> None:
    sentinel = "sensitive-policy-literal"
    predicate = ReadPredicate("tenant", ReadPredicateOperator.EQUAL, sentinel)

    assert sentinel not in repr(predicate)
    assert "<redacted>" in repr(predicate)
    with pytest.raises(AdapterError, match="non-empty tuple"):
        ReadPredicate("tenant", ReadPredicateOperator.IN, ())
    with pytest.raises(AdapterError, match="share one type"):
        ReadPredicate("tenant", ReadPredicateOperator.IN, ("one", 2))
    with pytest.raises(AdapterError, match="unsupported value"):
        ReadPredicate("value", ReadPredicateOperator.EQUAL, float("nan"))


def test_snapshot_contract_keeps_only_one_fake_batch_resident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = create_adapter(_duckdb_config(tmp_path / "fake.duckdb"))
    resident = 0
    peak_resident = 0
    closed = False

    def open_snapshot(_request: TableReadRequest) -> TableReadSnapshot:
        def batches() -> Iterator[pa.RecordBatch]:
            nonlocal resident, peak_resident
            for value in range(4):
                resident += 1
                peak_resident = max(peak_resident, resident)
                yield pa.record_batch([[value]], names=["value"])
                resident -= 1

        def close() -> None:
            nonlocal closed, resident
            resident = 0
            closed = True

        return TableReadSnapshot(
            schema=pa.schema([pa.field("value", pa.int64())]),
            fingerprint="0" * 32,
            batches=batches(),
            validate_unchanged=lambda: None,
            close=close,
        )

    monkeypatch.setattr(adapter, "_open_table_snapshot", open_snapshot)
    with adapter.table_snapshot("records", batch_size=1) as snapshot:
        assert [batch.column(0)[0].as_py() for batch in snapshot] == [0, 1, 2, 3]

    assert peak_resident == 1
    assert resident == 0
    assert closed


def test_snapshot_contract_closes_after_midstream_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = create_adapter(_duckdb_config(tmp_path / "failure.duckdb"))
    closed = False

    def open_snapshot(_request: TableReadRequest) -> TableReadSnapshot:
        def batches() -> Iterator[pa.RecordBatch]:
            yield pa.record_batch([[1]], names=["value"])
            raise AdapterError("safe simulated adapter read failure")

        def close() -> None:
            nonlocal closed
            closed = True

        return TableReadSnapshot(
            schema=pa.schema([pa.field("value", pa.int64())]),
            fingerprint="0" * 32,
            batches=batches(),
            validate_unchanged=lambda: None,
            close=close,
        )

    monkeypatch.setattr(adapter, "_open_table_snapshot", open_snapshot)
    with pytest.raises(AdapterError, match="safe simulated"):
        with adapter.table_snapshot("records") as snapshot:
            list(snapshot)

    assert snapshot.closed
    assert closed


def test_snapshot_schema_failure_does_not_retain_row_payload() -> None:
    sentinel = "sensitive-row-payload"

    def batches() -> Iterator[pa.RecordBatch]:
        yield pa.record_batch([[sentinel]], names=["unexpected"])

    snapshot = TableReadSnapshot(
        schema=pa.schema([pa.field("expected", pa.string())]),
        fingerprint="0" * 32,
        batches=batches(),
        validate_unchanged=lambda: None,
        close=lambda: None,
    )

    with pytest.raises(AdapterError, match="unstable schema") as exc_info:
        list(snapshot)

    _assert_sentinel_absent_from_error(exc_info.value, sentinel)


class _FakeReadRowsStream:
    def __init__(self, table: pa.Table, *, fail_with: str | None = None) -> None:
        self._table = table
        self._fail_with = fail_with
        self.cancelled = False

    def rows(self) -> Any:
        if self._fail_with is not None:
            raise RuntimeError(self._fail_with)
        pages = [
            SimpleNamespace(to_arrow=_returning(batch))
            for batch in self._table.to_batches()
        ]
        return SimpleNamespace(pages=iter(pages))

    def cancel(self) -> None:
        self.cancelled = True


def _returning(value: Any) -> Any:
    def _get() -> Any:
        return value

    return _get


class _FakeBigQueryStorageClient:
    """The Storage Read surface a snapshot uses, in place of the REST result
    endpoint that issue #441 could not get a wide row through."""

    def __init__(self) -> None:
        self.closed = False
        self.transport = self
        self.payload: pa.Table | None = None
        self.read_fails_with: str | None = None
        self.sessions: list[tuple[str, int]] = []
        self.streams: list[_FakeReadRowsStream] = []
        self.session_parents: list[str] = []
        self.timeouts: list[Any] = []

    def close(self) -> None:
        self.closed = True

    def create_read_session(
        self,
        *,
        parent: str,
        read_session: Any,
        max_stream_count: int,
        timeout: Any = None,
    ) -> Any:
        assert parent.startswith("projects/")
        assert max_stream_count == 1
        self.session_parents.append(parent)
        self.timeouts.append(timeout)
        self.sessions.append((read_session.table, max_stream_count))
        table = self.payload
        assert table is not None
        return SimpleNamespace(
            arrow_schema=SimpleNamespace(
                serialized_schema=table.schema.serialize()
            ),
            streams=(
                [SimpleNamespace(name="stream-0")] if table.num_rows else []
            ),
        )

    def read_rows(self, name: str, timeout: Any = None) -> _FakeReadRowsStream:
        del name
        self.timeouts.append(timeout)
        assert self.payload is not None
        stream = _FakeReadRowsStream(self.payload, fail_with=self.read_fails_with)
        self.streams.append(stream)
        return stream


class _FakeBigQueryJob:
    """Awaited through jobs.get. The two getQueryResults entry points raise:
    both reject on the underlying result size at any requested row count, so
    a regression that reaches for either must fail here (issue #441)."""

    def __init__(self, table: pa.Table) -> None:
        self.table = table
        self.job_id = "safe-job-id"
        self.cancelled = False
        self.error_result: dict[str, str] | None = None
        self.done_calls = 0
        self.destination = SimpleNamespace(
            project="project", dataset_id="_anon", table_id="anon_result"
        )

    def done(self, **_kwargs: Any) -> bool:
        self.done_calls += 1
        return True

    def result(self, **_kwargs: Any) -> Any:
        raise AssertionError("snapshot read used job.result() (issue #441)")

    def to_arrow(self, **_kwargs: Any) -> Any:
        raise AssertionError("snapshot read used job.to_arrow() (issue #441)")

    def cancel(self) -> None:
        self.cancelled = True


class _FakeAggregateRow:
    def __init__(self, values: tuple[Any, ...]) -> None:
        self._values = values

    def values(self) -> tuple[Any, ...]:
        return self._values


class _FakeAggregateJob:
    """The key-domain aggregate: one row of scalars, no Arrow payload."""

    def __init__(self, null_count: int, duplicate_count: int) -> None:
        self._row = _FakeAggregateRow((null_count, duplicate_count))
        self.job_id = "safe-aggregate-job-id"

    def result(self, **_kwargs: Any) -> list[_FakeAggregateRow]:
        return [self._row]


class _FakeBigQueryClient:
    def __init__(
        self,
        data: dict[str, pa.Array[Any]],
        *,
        generations: list[str] | None = None,
        null_count: int = 0,
        duplicate_count: int = 0,
    ) -> None:
        self.data = data
        self.generations = list(generations or ["generation-a"])
        self.null_count = null_count
        self.duplicate_count = duplicate_count
        self.queries: list[tuple[str, Any]] = []
        self.validation_queries: list[str] = []
        # None until the *payload* query runs. An invalid key domain now fails
        # before that happens, so this staying None is the assertion that the
        # expensive read was never started (issue #418).
        self.job: _FakeBigQueryJob | None = None
        self.get_table_calls = 0
        self.closed = False
        # Set by _bigquery_adapter: the payload query hands its table to the
        # read session, which is where the snapshot now reads it from.
        self.storage: Any = None

    def get_table(self, _table_id: str) -> Any:
        index = min(self.get_table_calls, len(self.generations) - 1)
        self.get_table_calls += 1
        schema = [SimpleNamespace(name=name) for name in self.data]
        return SimpleNamespace(
            schema=schema,
            etag=self.generations[index],
            modified=None,
            num_rows=len(next(iter(self.data.values()))) if self.data else 0,
        )

    def query(self, sql: str, job_config: Any = None, **_kwargs: Any) -> Any:
        self.queries.append((sql, job_config))
        if "COUNT(DISTINCT" in sql:
            # The key-domain aggregate is its own statement since #418; it
            # returns one row of scalars, not the payload.
            self.validation_queries.append(sql)
            return _FakeAggregateJob(self.null_count, self.duplicate_count)
        table = pa.table(dict(self.data))
        self.job = _FakeBigQueryJob(table)
        if self.storage is not None:
            self.storage.payload = table
        return self.job

    def close(self) -> None:
        self.closed = True


def _bigquery_adapter(client: _FakeBigQueryClient) -> BigQueryAdapter:
    config = parse_warehouse_config(
        {"type": "bigquery", "project": "project", "dataset": "dataset"}
    )
    adapter = create_adapter(config)
    assert isinstance(adapter, BigQueryAdapter)
    adapter._client = client
    adapter._bqstorage_client = _FakeBigQueryStorageClient()
    client.storage = adapter._bqstorage_client
    return adapter


def test_bigquery_streams_pages_with_projection_predicate_and_key_check() -> None:
    sentinel = "sensitive-policy-literal"
    client = _FakeBigQueryClient(
        {
            "record_id": pa.array(["a", "b", "c"]),
            "value": pa.array([1, 2, 3]),
        }
    )
    adapter = _bigquery_adapter(client)

    with adapter.table_snapshot(
        "records",
        columns=("record_id", "value"),
        batch_size=2,
        predicate=ReadPredicate("record_id", ReadPredicateOperator.NOT_EQUAL, sentinel),
        key_column="record_id",
    ) as snapshot:
        assert [len(batch) for batch in snapshot] == [2, 1]
        assert snapshot.schema.names == ["record_id", "value"]
        assert snapshot.generation_fingerprint is not None

    assert client.job is not None
    # Awaited through jobs.get and read from the job's own destination table
    # through one Storage Read stream — never through getQueryResults, which
    # rejects a wide row at any requested row count (issue #441).
    assert client.job.done_calls >= 1
    storage = adapter._bqstorage_client
    assert isinstance(storage, _FakeBigQueryStorageClient)
    assert storage.sessions == [
        ("projects/project/datasets/_anon/tables/anon_result", 1)
    ]
    assert len(storage.streams) == 1
    validation_sql, payload_sql = (sql for sql, _cfg in client.queries)
    assert sentinel not in validation_sql
    assert sentinel not in payload_sql
    # The key domain is checked over the key column alone, in its own
    # statement; the payload carries no analytic frame at all (issue #418).
    assert "COUNTIF(`record_id` IS NULL)" in validation_sql
    assert "`value`" not in validation_sql
    assert "COUNTIF" not in payload_sql
    assert "OVER" not in payload_sql
    assert "SELECT `record_id`, `value`" in payload_sql
    # Both statements bind the predicate rather than inlining it, and both
    # bypass the query cache: a cached aggregate would be answering about a
    # different read than the one it guards, and the generation fence compares
    # table etags rather than query results, so it could not see that.
    for _sql, job_config in client.queries:
        assert job_config.query_parameters[0].value == sentinel
        assert job_config.use_query_cache is False


def test_bigquery_key_failure_is_sanitized_and_cancels_result() -> None:
    client = _FakeBigQueryClient(
        {
            "record_id": pa.array(["sensitive-row", "sensitive-row"]),
            "value": pa.array([1, 2]),
        },
        duplicate_count=1,
    )
    adapter = _bigquery_adapter(client)

    with pytest.raises(AdapterError, match="key domain is invalid") as exc_info:
        with adapter.table_snapshot("records", key_column="record_id") as snapshot:
            list(snapshot)

    _assert_sentinel_absent_from_error(exc_info.value, "sensitive-row")
    # Nothing to cancel: validation now runs before the payload query is
    # started, so an invalid key domain never costs a scan of the payload
    # (issue #418). Previously the read had been launched and then cancelled.
    assert client.job is None
    assert client.validation_queries


def test_bigquery_generation_change_fails_final_validation() -> None:
    client = _FakeBigQueryClient(
        {"record_id": pa.array(["a"])},
        generations=["generation-a", "generation-a", "generation-b"],
    )
    adapter = _bigquery_adapter(client)

    with pytest.raises(TableSnapshotGenerationChangedError, match="changed during"):
        with adapter.table_snapshot("records") as snapshot:
            assert len(list(snapshot)) == 1


def test_bigquery_generation_change_fails_snapshot_open_with_typed_error() -> None:
    client = _FakeBigQueryClient(
        {"record_id": pa.array(["a"])},
        generations=["generation-a", "generation-b"],
    )
    adapter = _bigquery_adapter(client)

    with pytest.raises(
        TableSnapshotGenerationChangedError, match="changed while opening"
    ):
        with adapter.table_snapshot("records"):
            pass


def test_bigquery_empty_relation_keeps_typed_schema() -> None:
    client = _FakeBigQueryClient({"record_id": pa.array([], type=pa.string())})
    adapter = _bigquery_adapter(client)

    with adapter.table_snapshot("records") as snapshot:
        assert list(snapshot) == []
        assert snapshot.schema == pa.schema([pa.field("record_id", pa.string())])


def test_bigquery_snapshot_open_failure_sanitizes_cause() -> None:
    sentinel = "diagnostic-only-open-error"

    class FailingClient(_FakeBigQueryClient):
        def get_table(self, _table_id: str) -> Any:
            raise RuntimeError(sentinel)

    adapter = _bigquery_adapter(FailingClient({"record_id": pa.array(["a"])}))
    with pytest.raises(
        AdapterError, match=r"could not be opened.*Native adapter error type: RuntimeError"
    ) as exc_info:
        with adapter.table_snapshot("records"):
            pass

    assert (
        str(_sanitized_native_cause(exc_info.value, sentinel))
        == "Native adapter error type: RuntimeError"
    )


def test_bigquery_snapshot_batch_failure_sanitizes_cause() -> None:
    sentinel = "diagnostic-only-batch-error"

    adapter = _bigquery_adapter(_FakeBigQueryClient({"record_id": pa.array(["a"])}))
    # The stream itself fails mid-read, which is where a native message would
    # otherwise reach the caller.
    adapter._bqstorage_client.read_fails_with = sentinel
    with adapter.table_snapshot("records") as snapshot:
        with pytest.raises(AdapterError, match="batch read failed") as exc_info:
            list(snapshot)

    assert (
        str(_sanitized_native_cause(exc_info.value, sentinel))
        == "Native adapter error type: RuntimeError"
    )


def test_bigquery_snapshot_generation_failure_sanitizes_cause() -> None:
    sentinel = "diagnostic-only-generation-error"

    class FailingValidationClient(_FakeBigQueryClient):
        def get_table(self, _table_id: str) -> Any:
            if self.get_table_calls >= 2:
                raise RuntimeError(sentinel)
            return super().get_table(_table_id)

    adapter = _bigquery_adapter(
        FailingValidationClient({"record_id": pa.array(["a"])})
    )
    with pytest.raises(AdapterError, match="generation could not be validated") as exc_info:
        with adapter.table_snapshot("records") as snapshot:
            assert len(list(snapshot)) == 1

    assert (
        str(_sanitized_native_cause(exc_info.value, sentinel))
        == "Native adapter error type: RuntimeError"
    )


def test_bigquery_early_close_cancels_unconsumed_result() -> None:
    client = _FakeBigQueryClient({"record_id": pa.array(["a", "b"])})
    adapter = _bigquery_adapter(client)

    with adapter.table_snapshot("records", batch_size=1) as snapshot:
        assert next(iter(snapshot)).column(0)[0].as_py() == "a"

    assert client.job is not None and client.job.cancelled


def test_bigquery_adapter_closes_owned_storage_client() -> None:
    client = _FakeBigQueryClient({"record_id": pa.array(["a"])})
    adapter = _bigquery_adapter(client)
    storage_client = adapter._bqstorage_client

    adapter._close()

    assert isinstance(storage_client, _FakeBigQueryStorageClient)
    assert storage_client.closed
    assert client.closed
    assert adapter._bqstorage_client is None


def test_bigquery_adapter_lazily_reuses_storage_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google.cloud import bigquery_storage

    client = _FakeBigQueryClient({"record_id": pa.array(["a"])})
    config = parse_warehouse_config(
        {"type": "bigquery", "project": "project", "dataset": "dataset"}
    )
    adapter = create_adapter(config)
    assert isinstance(adapter, BigQueryAdapter)
    adapter._client = client
    credential = object()
    created: list[Any] = []

    def make_storage_client(*, credentials: Any) -> _FakeBigQueryStorageClient:
        assert credentials is credential
        storage_client = _FakeBigQueryStorageClient()
        created.append(storage_client)
        return storage_client

    monkeypatch.setattr(adapter, "_credentials", lambda: credential)
    monkeypatch.setattr(bigquery_storage, "BigQueryReadClient", make_storage_client)

    first = adapter._ensure_bqstorage_client()
    second = adapter._ensure_bqstorage_client()
    adapter._close()

    assert first is second
    assert created == [first]
    assert first.closed
    assert client.closed


def test_implemented_adapters_declare_streaming_read_capability() -> None:
    assert WarehouseCapability.STREAMING_TABULAR_READS in BigQueryAdapter.capabilities()
    assert WarehouseCapability.TABULAR_PREDICATE_PUSHDOWN in BigQueryAdapter.capabilities()
