"""LanceDB index builds retry transient failures with backoff (issue #491).

The index step is the last of a publish that may have written rows for hours,
and it used to be single-shot: one transient object-store error discarded the
run, and the retry cost a full corpus re-read. These tests pin the policy —
bounded attempts, doubling delay, `replace=True` on every retry — and the two
things it must not do: retry a deliberate refusal, or leak the native error
text through the retry warning.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from pydantic import ValidationError

from stel.retrieval import (
    CollectionSpec,
    IndexedRow,
    LanceDBConfig,
    LanceDBStore,
    RetrievalError,
    StoreRole,
)
from stel.retrieval import lancedb as lancedb_store

SENTINEL = "gs://distinctive-bucket/prefix?token=distinctive-native-secret"
PHYSICAL = "demo__dev__context"


def _config(tmp_path: Path, **overrides: Any) -> LanceDBConfig:
    payload: dict[str, Any] = {"type": "lancedb", "path": str(tmp_path / "lance")}
    payload.update(overrides)
    return LanceDBConfig.model_validate(payload)


def _store(tmp_path: Path, **overrides: Any) -> LanceDBStore:
    return LanceDBStore(
        _config(tmp_path, **overrides),
        project_name="demo",
        target_name="dev",
        alias="primary",
        role=StoreRole.PUBLISH,
    )


def _spec(
    *,
    scalar_index_fields: tuple[str, ...] = ("category",),
    full_text_fields: tuple[str, ...] = (),
    vector_search: str = "exact",
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
        vector_search=vector_search,
        vector_index="ivf_hnsw_flat" if vector_search == "approximate" else None,
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


def _publish(store: LanceDBStore, spec: CollectionSpec) -> None:
    store.create_collection(spec)
    rows = [
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
    store.upsert(PHYSICAL, rows, id_field="chunk_id", mutation_digest="digest")


class _FlakyTable:
    """A real table whose `create_index` fails the first `failures` times."""

    def __init__(
        self, table: Any, failures: int, *, raise_: type[Exception] = RuntimeError
    ) -> None:
        self._table = table
        self._failures = failures
        self._raise = raise_
        self.calls: list[dict[str, Any]] = []

    def create_index(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if len(self.calls) <= self._failures:
            raise self._raise(SENTINEL)
        return self._table.create_index(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._table, name)


def _flaky(
    monkeypatch: pytest.MonkeyPatch, failures: int, **kwargs: Any
) -> dict[str, _FlakyTable]:
    original = LanceDBStore._open_owned_table
    seen: dict[str, _FlakyTable] = {}

    def open_flaky(self: LanceDBStore, name: str) -> Any:
        if "table" not in seen:
            seen["table"] = _FlakyTable(original(self, name), failures, **kwargs)
        return seen["table"]

    monkeypatch.setattr(LanceDBStore, "_open_owned_table", open_flaky)
    return seen


def _record_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    sleeps: list[float] = []
    monkeypatch.setattr(lancedb_store, "_sleep", sleeps.append)
    return sleeps


# ─── the policy ─────────────────────────────────────────────────────────────


def test_transient_build_failures_are_retried_with_doubling_backoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="stel.retrieval.lancedb")
    sleeps = _record_sleeps(monkeypatch)
    spec = _spec()
    with _store(tmp_path) as store:
        _publish(store, spec)
        seen = _flaky(monkeypatch, failures=2)
        metadata = store.ensure_indexes(spec)
        assert metadata.row_count == 2

    calls = seen["table"].calls
    assert len(calls) == 3
    # A fresh index is built without replace; every retry replaces, because a
    # failed first attempt may have left a committed index behind.
    assert [call["replace"] for call in calls] == [False, True, True]
    assert sleeps == [5.0, 10.0]

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert [r.getMessage() for r in warnings] == [
        "LanceDB index build on BTree index for 'category' failed [RuntimeError]; "
        "retrying in 5.0s (attempt 1 of 3)",
        "LanceDB index build on BTree index for 'category' failed [RuntimeError]; "
        "retrying in 10.0s (attempt 2 of 3)",
    ]
    for record in caplog.records:
        if record.levelno >= logging.INFO:
            assert "distinctive" not in record.getMessage()
    # The native text is reachable only through the DEBUG record's exc_info.
    debug = [r for r in caplog.records if r.levelno == logging.DEBUG and r.exc_info]
    assert len(debug) == 2
    assert all(SENTINEL in str(r.exc_info[1]) for r in debug if r.exc_info)


def test_exhausted_retries_report_the_attempt_count_and_native_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps = _record_sleeps(monkeypatch)
    spec = _spec()
    with _store(tmp_path, index_build_attempts=2, index_build_retry_seconds=1.5) as store:
        _publish(store, spec)
        seen = _flaky(monkeypatch, failures=5)
        with pytest.raises(RetrievalError) as exc_info:
            store.ensure_indexes(spec)

    assert str(exc_info.value) == (
        "LanceDB operation 'index creation' failed on BTree index for 'category' "
        "after 2 attempts [RuntimeError] (code=lancedb_index_failed)"
    )
    assert len(seen["table"].calls) == 2
    assert sleeps == [1.5]
    assert "distinctive" not in repr(exc_info.value)
    cause = exc_info.value.__cause__
    assert isinstance(cause, RetrievalError)
    assert str(cause) == "Native retrieval error type: RuntimeError"
    assert exc_info.value.__context__ is None


def test_a_single_attempt_disables_retry_and_keeps_the_plain_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps = _record_sleeps(monkeypatch)
    spec = _spec()
    with _store(tmp_path, index_build_attempts=1) as store:
        _publish(store, spec)
        seen = _flaky(monkeypatch, failures=5)
        with pytest.raises(RetrievalError) as exc_info:
            store.ensure_indexes(spec)

    assert str(exc_info.value) == (
        "LanceDB operation 'index creation' failed on BTree index for 'category' "
        "[RuntimeError] (code=lancedb_index_failed)"
    )
    assert len(seen["table"].calls) == 1
    assert sleeps == []


def test_each_index_kind_gets_its_own_retry_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The budget is per build, not per publish: a BTree that needed two
    attempts must not leave the vector build with one."""
    sleeps = _record_sleeps(monkeypatch)
    spec = _spec(full_text_fields=("text",), vector_search="approximate")
    with _store(tmp_path, index_build_attempts=2) as store:
        _publish(store, spec)
        original = LanceDBStore._open_owned_table
        counts: dict[str, int] = {}

        class _EveryFirstAttemptFails:
            def __init__(self, table: Any) -> None:
                self._table = table

            def create_index(self, column: str, *args: Any, **kwargs: Any) -> Any:
                counts[column] = counts.get(column, 0) + 1
                if counts[column] == 1:
                    raise RuntimeError(SENTINEL)
                return self._table.create_index(column, *args, **kwargs)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._table, name)

        monkeypatch.setattr(
            LanceDBStore,
            "_open_owned_table",
            lambda self, name: _EveryFirstAttemptFails(original(self, name)),
        )
        store.ensure_indexes(spec)

    assert counts == {"category": 2, "text": 2, "embedding": 2}
    assert sleeps == [5.0, 5.0, 5.0]


# ─── what is never retried ──────────────────────────────────────────────────


def test_store_refusals_inside_a_build_are_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps = _record_sleeps(monkeypatch)
    spec = _spec()
    with _store(tmp_path) as store:
        _publish(store, spec)
        seen = _flaky(monkeypatch, failures=5, raise_=RetrievalError)
        with pytest.raises(RetrievalError, match="distinctive") as exc_info:
            store.ensure_indexes(spec)

    # A RetrievalError is stel's own safe text, raised through untouched.
    assert str(exc_info.value) == SENTINEL
    assert len(seen["table"].calls) == 1
    assert sleeps == []


def test_failures_outside_the_build_are_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps = _record_sleeps(monkeypatch)
    spec = _spec()
    with _store(tmp_path) as store:
        _publish(store, spec)
        original = LanceDBStore._open_owned_table

        class _ListingFails:
            def __init__(self, table: Any) -> None:
                self._table = table

            def list_indices(self) -> Any:
                raise RuntimeError(SENTINEL)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._table, name)

        monkeypatch.setattr(
            LanceDBStore,
            "_open_owned_table",
            lambda self, name: _ListingFails(original(self, name)),
        )
        with pytest.raises(RetrievalError, match="on index listing \\[RuntimeError\\]"):
            store.ensure_indexes(spec)
    assert sleeps == []


# ─── configuration ──────────────────────────────────────────────────────────


def test_defaults_and_bounds(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert config.index_build_attempts == 3
    assert config.index_build_retry_seconds == 5.0
    with pytest.raises(ValidationError):
        _config(tmp_path, index_build_attempts=0)
    with pytest.raises(ValidationError):
        _config(tmp_path, index_build_attempts=11)
    with pytest.raises(ValidationError):
        _config(tmp_path, index_build_retry_seconds=-1)
    with pytest.raises(ValidationError):
        _config(tmp_path, index_build_retry_seconds=601)


def test_retry_settings_do_not_enter_the_store_identity(tmp_path: Path) -> None:
    """Tuning the retry must not reclassify a published collection or strand
    its incremental state, exactly as the cache budgets (#479) must not."""
    plain = _store(tmp_path)
    tuned = _store(tmp_path, index_build_attempts=8, index_build_retry_seconds=30)
    assert tuned.safe_descriptor() == plain.safe_descriptor()
    assert tuned.state_descriptor("context") == plain.state_descriptor("context")


# ─── a pool refusal is not transient (issue #636) ───────────────────────────

# What `lance` actually raises when the DataFusion pool will not grow. Carries
# byte counts and a struct name -- no URI, no credential -- but is still only
# ever read to classify, never copied into the failure.
POOL_ERROR = (
    "lance error: LanceError(IO): Resources exhausted: Failed to allocate "
    "additional 352.0 KB for ExternalSorterMerge[0] with 32.5 MB already "
    "allocated for this reservation - 281.1 KB remain available for the "
    "total pool, lance-datafusion-8.0.0/src/chunker.rs:49:46"
)


class _PoolExhaustedTable:
    """A real table whose `create_index` always refuses for want of pool."""

    def __init__(self, table: Any) -> None:
        self._table = table
        self.calls: list[dict[str, Any]] = []

    def create_index(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        raise RuntimeError(POOL_ERROR)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._table, name)


def _pool_exhausted(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    original = LanceDBStore._open_owned_table
    seen: dict[str, Any] = {}

    def open_refusing(self: LanceDBStore, name: str) -> Any:
        if "table" not in seen:
            seen["table"] = _PoolExhaustedTable(original(self, name))
        return seen["table"]

    monkeypatch.setattr(LanceDBStore, "_open_owned_table", open_refusing)
    return seen


def test_a_pool_refusal_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three attempts cannot survive a deterministic refusal.

    `Resources exhausted` from the DataFusion pool depends on the pool size,
    not on timing: a BTree over 3.64M keys failed in 2.7s and kept failing,
    while the operator saw only `[RuntimeError]` for the two backoffs
    (issue #636). Retrying it spends 15s to reach the same place.
    """
    sleeps = _record_sleeps(monkeypatch)
    spec = _spec()
    with _store(tmp_path) as store:
        _publish(store, spec)
        seen = _pool_exhausted(monkeypatch)
        with pytest.raises(RetrievalError) as exc_info:
            store.ensure_indexes(spec)

    assert len(seen["table"].calls) == 1, "a pool refusal must not be retried"
    assert sleeps == [], "no backoff should be spent on a deterministic failure"
    assert "deterministic rather than transient" in str(exc_info.value)


def test_the_pool_refusal_names_the_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The operator's next step is a pool size, and before this the message
    # gave them `[RuntimeError]` and nothing else.
    _record_sleeps(monkeypatch)
    spec = _spec()
    with _store(tmp_path) as store:
        _publish(store, spec)
        _pool_exhausted(monkeypatch)
        with pytest.raises(RetrievalError) as exc_info:
            store.ensure_indexes(spec)

    message = str(exc_info.value)
    assert lancedb_store.MEM_POOL_ENV in message
    assert "memory_pool_size_mb" in message
    assert "lancedb_index_pool_exhausted" in message
    # Classified from the native text, never quoting it.
    assert "ExternalSorterMerge" not in message
    assert "chunker.rs" not in message


def test_a_transient_failure_is_still_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The #491 policy must survive the #636 classification: anything that is
    # not a pool refusal keeps its retry budget.
    sleeps = _record_sleeps(monkeypatch)
    spec = _spec()
    with _store(tmp_path) as store:
        _publish(store, spec)
        seen = _flaky(monkeypatch, failures=2)
        store.ensure_indexes(spec)

    assert len(seen["table"].calls) == 3
    assert sleeps == [5.0, 10.0]


# ─── sizing the pool ────────────────────────────────────────────────────────


def test_the_configured_pool_size_is_exported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Lance offers no per-connection seam for this, so the store sets the
    # process environment -- and says so.
    monkeypatch.delenv(lancedb_store.MEM_POOL_ENV, raising=False)
    with _store(tmp_path, memory_pool_size_mb=512):
        assert os.environ[lancedb_store.MEM_POOL_ENV] == str(512 * 1024 * 1024)


def test_an_operator_set_pool_size_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Their value may be sized for a larger job in the same process.

    A store-level default exists to fill a gap, not to overrule someone who
    has already made the decision.
    """
    monkeypatch.setenv(lancedb_store.MEM_POOL_ENV, "123456789")
    with _store(tmp_path, memory_pool_size_mb=512):
        assert os.environ[lancedb_store.MEM_POOL_ENV] == "123456789"


def test_an_unset_pool_size_touches_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The default must change nothing for an existing deployment.
    monkeypatch.delenv(lancedb_store.MEM_POOL_ENV, raising=False)
    with _store(tmp_path):
        assert lancedb_store.MEM_POOL_ENV not in os.environ
