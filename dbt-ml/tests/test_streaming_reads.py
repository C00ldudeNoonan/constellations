from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import duckdb
import polars as pl
import pyarrow as pa
import pytest

from dbt_ml.adapters import (
    AdapterCapabilityError,
    AdapterError,
    ReadOrdering,
    ReadPredicate,
    ReadPredicateOperator,
    TableReadRequest,
    TableReadSnapshot,
    WarehouseCapability,
    create_adapter,
    parse_warehouse_config,
)
from dbt_ml.adapters.bigquery import BigQueryAdapter


def _duckdb_config(path: Path) -> Any:
    return parse_warehouse_config({"type": "duckdb", "path": str(path), "schema": "testns"})


def _assert_sentinel_absent_from_error(error: BaseException, sentinel: str) -> None:
    assert sentinel not in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None
    traceback = error.__traceback__
    while traceback is not None:
        if "/src/dbt_ml/" in traceback.tb_frame.f_code.co_filename:
            assert sentinel not in repr(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next


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

    _assert_sentinel_absent_from_error(exc_info.value, sentinel)


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


class _FakeBigQueryRows:
    def __init__(self, table: pa.Table, page_size: int) -> None:
        self.table = table
        self.page_size = page_size
        self.max_queue_size: int | None = None

    def to_arrow_iterable(self, **kwargs: Any) -> Iterator[pa.RecordBatch]:
        assert kwargs["bqstorage_client"] is None
        self.max_queue_size = int(kwargs["max_queue_size"])
        yield from self.table.to_batches(max_chunksize=self.page_size)


class _FakeBigQueryJob:
    def __init__(self, table: pa.Table) -> None:
        self.table = table
        self.job_id = "safe-job-id"
        self.page_size: int | None = None
        self.rows: _FakeBigQueryRows | None = None
        self.cancelled = False

    def result(self, *, page_size: int, timeout: float | None) -> _FakeBigQueryRows:
        self.page_size = page_size
        self.rows = _FakeBigQueryRows(self.table, page_size)
        return self.rows

    def to_arrow(self, **kwargs: Any) -> pa.Table:
        assert kwargs["max_results"] == 1
        return self.table.slice(0, 1)

    def cancel(self) -> None:
        self.cancelled = True


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
        self.job: _FakeBigQueryJob | None = None
        self.get_table_calls = 0

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

    def query(self, sql: str, job_config: Any = None, **_kwargs: Any) -> _FakeBigQueryJob:
        self.queries.append((sql, job_config))
        columns = dict(self.data)
        null_match = re.search(r"dbt_ml_read_nulls_[0-9a-f]+", sql)
        duplicate_match = re.search(r"dbt_ml_read_duplicates_[0-9a-f]+", sql)
        row_count = len(next(iter(columns.values()))) if columns else 0
        if null_match is not None:
            columns[null_match.group()] = pa.array([self.null_count] * row_count)
        if duplicate_match is not None:
            columns[duplicate_match.group()] = pa.array([self.duplicate_count] * row_count)
        self.job = _FakeBigQueryJob(pa.table(columns))
        return self.job


def _bigquery_adapter(client: _FakeBigQueryClient) -> BigQueryAdapter:
    config = parse_warehouse_config(
        {"type": "bigquery", "project": "project", "dataset": "dataset"}
    )
    adapter = create_adapter(config)
    assert isinstance(adapter, BigQueryAdapter)
    adapter._client = client
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
    assert client.job.page_size == 2
    assert client.job.rows is not None
    assert client.job.rows.max_queue_size == 1
    sql, job_config = client.queries[0]
    assert sentinel not in sql
    assert "COUNTIF" in sql
    assert "SELECT `record_id`, `value`" in sql
    assert job_config.query_parameters[0].value == sentinel


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
    assert client.job is not None and client.job.cancelled


def test_bigquery_generation_change_fails_final_validation() -> None:
    client = _FakeBigQueryClient(
        {"record_id": pa.array(["a"])},
        generations=["generation-a", "generation-a", "generation-b"],
    )
    adapter = _bigquery_adapter(client)

    with pytest.raises(AdapterError, match="changed during"):
        with adapter.table_snapshot("records") as snapshot:
            assert len(list(snapshot)) == 1


def test_bigquery_empty_relation_keeps_typed_schema() -> None:
    client = _FakeBigQueryClient({"record_id": pa.array([], type=pa.string())})
    adapter = _bigquery_adapter(client)

    with adapter.table_snapshot("records") as snapshot:
        assert list(snapshot) == []
        assert snapshot.schema == pa.schema([pa.field("record_id", pa.string())])


def test_bigquery_early_close_cancels_unconsumed_result() -> None:
    client = _FakeBigQueryClient({"record_id": pa.array(["a", "b"])})
    adapter = _bigquery_adapter(client)

    with adapter.table_snapshot("records", batch_size=1) as snapshot:
        assert next(iter(snapshot)).column(0)[0].as_py() == "a"

    assert client.job is not None and client.job.cancelled


def test_implemented_adapters_declare_streaming_read_capability() -> None:
    assert WarehouseCapability.STREAMING_TABULAR_READS in BigQueryAdapter.capabilities()
    assert WarehouseCapability.TABULAR_PREDICATE_PUSHDOWN in BigQueryAdapter.capabilities()
