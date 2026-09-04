"""LanceDB failures stay artifact-safe while remaining diagnosable (issue #490).

A native LanceDB error used to be discarded with ``raise ... from None``,
leaving a multi-hour publish that died at the index step with no reason, no
exception type, and no record of which index it was building. These tests pin
the replacement contract: the message names the operation, the step, and the
native exception's type; the cause chain carries the type alone; and the full
native exception reaches only the DEBUG log.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from stel.retrieval import (
    CollectionSpec,
    IndexedRow,
    LanceDBStore,
    RetrievalError,
    StoreRole,
    parse_store_config,
)

# Shaped like what LanceDB actually quotes back: an object-store URI carrying
# a credential-looking query string. None of it may reach the message.
SENTINEL = "gs://distinctive-bucket/prefix?token=distinctive-native-secret"
PHYSICAL = "demo__dev__context"


def _store(tmp_path: Path, **overrides: object) -> LanceDBStore:
    config = parse_store_config(
        {"type": "lancedb", "path": str(tmp_path / "lance"), **overrides}
    )
    return LanceDBStore(
        config,
        project_name="demo",
        target_name="dev",
        alias="primary",
        role=StoreRole.PUBLISH,
    )


def _spec(
    *,
    scalar_index_fields: tuple[str, ...] = ("category",),
    full_text_fields: tuple[str, ...] = ("text",),
) -> CollectionSpec:
    return CollectionSpec(
        logical_name="context",
        physical_name=PHYSICAL,
        id_field="chunk_id",
        text_fields=("text",),
        full_text_fields=full_text_fields,
        attribute_fields=("category",),
        scalar_index_fields=scalar_index_fields,
        display_fields=(),
        vector_field="embedding",
        vector_dimensions=2,
        distance_metric="cosine",
        vector_search="approximate",
        vector_index="ivf_hnsw_flat",
        config_fingerprint="fingerprint",
        descriptor="{}",
        legacy_config_fingerprint="legacy",
        row_fingerprint="row-fp",
        arrow_schema=pa.schema(
            [
                pa.field("chunk_id", pa.string(), nullable=False),
                pa.field("text", pa.string()),
                pa.field("category", pa.string()),
                pa.field("embedding", pa.list_(pa.float32(), 2)),
            ]
        ),
    )


def _rows() -> list[IndexedRow]:
    return [
        IndexedRow(
            "c1",
            {"chunk_id": "c1", "text": "alpha", "category": "a", "embedding": [0.1, 0.9]},
            "f1",
        ),
        IndexedRow(
            "c2",
            {"chunk_id": "c2", "text": "beta", "category": "b", "embedding": [0.8, 0.2]},
            "f2",
        ),
    ]


def _publish(store: LanceDBStore, spec: CollectionSpec) -> None:
    store.create_collection(spec)
    store.upsert(PHYSICAL, _rows(), id_field="chunk_id", mutation_digest="digest")


class _FailingTable:
    """Delegates to a real LanceDB table except for one method that raises."""

    def __init__(self, table: Any, failing: str) -> None:
        self._table = table
        self._failing = failing

    def __getattr__(self, name: str) -> Any:
        if name == self._failing:

            def fail(*args: Any, **kwargs: Any) -> None:
                raise RuntimeError(SENTINEL)

            return fail
        return getattr(self._table, name)


def _fail_on(monkeypatch: pytest.MonkeyPatch, failing: str) -> None:
    original = LanceDBStore._open_owned_table

    def open_failing(self: LanceDBStore, name: str) -> Any:
        return _FailingTable(original(self, name), failing)

    monkeypatch.setattr(LanceDBStore, "_open_owned_table", open_failing)


def _assert_sanitized(error: RetrievalError, caplog: pytest.LogCaptureFixture) -> None:
    rendered = "".join((str(error), repr(error)))
    assert "distinctive" not in rendered
    cause = error.__cause__
    assert isinstance(cause, RetrievalError)
    assert str(cause) == "Native retrieval error type: RuntimeError"
    assert cause.__traceback__ is None
    assert cause.__cause__ is None
    assert cause.__context__ is None
    # Raised outside the except block, so the native exception is not kept
    # reachable through implicit chaining either.
    assert error.__context__ is None

    # The native exception is available to an operator who attaches a DEBUG
    # handler, and nowhere else: no INFO-or-louder record carries it.
    debug = [
        record
        for record in caplog.records
        if record.name == "stel.retrieval.lancedb" and record.levelno == logging.DEBUG
    ]
    assert len(debug) == 1
    exc_info = debug[0].exc_info
    assert exc_info is not None
    assert isinstance(exc_info[1], RuntimeError)
    assert SENTINEL in str(exc_info[1])
    for record in caplog.records:
        if record.levelno >= logging.INFO:
            assert "distinctive" not in record.getMessage()


@pytest.mark.parametrize(
    ("scalar", "full_text", "step"),
    [
        (("category",), ("text",), "BTree index for 'category'"),
        ((), ("text",), "FTS index for 'text'"),
        ((), (), "IvfHnswFlat vector index for 'embedding'"),
    ],
)
def test_index_failure_names_the_index_and_native_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    scalar: tuple[str, ...],
    full_text: tuple[str, ...],
    step: str,
) -> None:
    caplog.set_level(logging.DEBUG, logger="stel.retrieval.lancedb")
    spec = _spec(scalar_index_fields=scalar, full_text_fields=full_text)
    # One attempt: the retry policy (#491) is pinned in test_lancedb_index_retry.py;
    # this test pins the message shape of a build that fails outright.
    with _store(tmp_path, index_build_attempts=1) as store:
        _publish(store, spec)
        _fail_on(monkeypatch, "create_index")
        with pytest.raises(RetrievalError) as exc_info:
            store.ensure_indexes(spec)

    assert str(exc_info.value) == (
        f"LanceDB operation 'index creation' failed on {step} "
        "[RuntimeError] (code=lancedb_index_failed)"
    )
    _assert_sanitized(exc_info.value, caplog)


def test_index_step_reports_the_listing_when_no_index_was_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="stel.retrieval.lancedb")
    spec = _spec()
    with _store(tmp_path) as store:
        _publish(store, spec)
        _fail_on(monkeypatch, "list_indices")
        with pytest.raises(RetrievalError) as exc_info:
            store.ensure_indexes(spec)

    assert str(exc_info.value) == (
        "LanceDB operation 'index creation' failed on index listing "
        "[RuntimeError] (code=lancedb_index_failed)"
    )
    _assert_sanitized(exc_info.value, caplog)


def test_store_authored_errors_pass_through_index_creation_unwrapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A RetrievalError raised inside ensure_indexes already carries a safe,
    specific message and must not be relabelled as a generic index failure."""
    spec = _spec()

    def refuse(self: LanceDBStore, name: str) -> Any:
        raise RetrievalError("LanceDB collection is not owned by stel (code=test_specific)")

    with _store(tmp_path) as store:
        _publish(store, spec)
        monkeypatch.setattr(LanceDBStore, "_open_owned_table", refuse)
        with pytest.raises(RetrievalError, match="code=test_specific") as exc_info:
            store.ensure_indexes(spec)
    assert exc_info.value.__cause__ is None


def test_upsert_failure_reports_native_type_without_native_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="stel.retrieval.lancedb")
    spec = _spec()
    with _store(tmp_path) as store:
        store.create_collection(spec)
        _fail_on(monkeypatch, "merge_insert")
        with pytest.raises(RetrievalError) as exc_info:
            store.upsert(PHYSICAL, _rows(), id_field="chunk_id", mutation_digest="digest")

    assert str(exc_info.value) == (
        "LanceDB operation 'upsert' failed [RuntimeError] (code=lancedb_upsert_failed)"
    )
    _assert_sanitized(exc_info.value, caplog)


def test_append_failure_reports_native_type_without_native_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="stel.retrieval.lancedb")
    spec = _spec()
    with _store(tmp_path) as store:
        store.create_collection(spec)
        _fail_on(monkeypatch, "add")
        with pytest.raises(RetrievalError) as exc_info:
            store.append(PHYSICAL, _rows(), id_field="chunk_id", mutation_digest="digest")

    assert str(exc_info.value) == (
        "LanceDB operation 'append' failed [RuntimeError] (code=lancedb_append_failed)"
    )
    _assert_sanitized(exc_info.value, caplog)


def test_delete_failure_reports_native_type_without_native_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="stel.retrieval.lancedb")
    spec = _spec()
    with _store(tmp_path) as store:
        _publish(store, spec)
        _fail_on(monkeypatch, "delete")
        with pytest.raises(RetrievalError) as exc_info:
            store.delete(PHYSICAL, ["c1"], id_field="chunk_id", mutation_digest="digest")

    assert str(exc_info.value) == (
        "LanceDB operation 'delete' failed [RuntimeError] (code=lancedb_delete_failed)"
    )
    _assert_sanitized(exc_info.value, caplog)


def test_connect_failure_reports_native_type_without_native_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="stel.retrieval.lancedb")

    def fail_connect(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError(SENTINEL)

    monkeypatch.setattr("lancedb.connect", fail_connect)
    with pytest.raises(RetrievalError) as exc_info:
        with _store(tmp_path):
            pass

    assert str(exc_info.value) == (
        "LanceDB operation 'connect' failed [RuntimeError] (code=lancedb_connect_failed)"
    )
    _assert_sanitized(exc_info.value, caplog)
