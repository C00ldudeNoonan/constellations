"""A merge page is bounded by bytes, not by rows (issue #592).

`merge_insert` reserves its whole payload for the join build side, out of a
pool fixed at 100 MB and not reachable from configuration. `batch_size`
counts rows. Two consecutive weekly publishes of a 3.6M-row collection died
on that mismatch: 20,000 rows fit at 91.8 MB, 25,000 asked for 111.7 MB of a
100 MB pool.

These pin the split that removes the mismatch, and — in the last test — that
it is the *measurement* doing the work rather than an average, because the
whole failure was an average-shaped assumption about row size.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from stel.retrieval import (
    CollectionSpec,
    IndexedRow,
    LanceDBStore,
    RetrievalFeature,
    StoreRole,
    parse_store_config,
)
from stel.retrieval.lancedb import MERGE_PAYLOAD_LIMIT_BYTES, _byte_bounded_slices

PHYSICAL = "demo__dev__context"


def _table(rows: int, width: int) -> pa.Table:
    return pa.table(
        {
            "chunk_id": [f"c{i}" for i in range(rows)],
            "text": ["x" * width for _ in range(rows)],
        }
    )


# ─── the splitter ───────────────────────────────────────────────────────────


def test_a_payload_under_the_limit_is_one_slice() -> None:
    # The common case must not change shape: one page, one merge_insert.
    payload = _table(10, 100)
    slices = list(_byte_bounded_slices(payload, MERGE_PAYLOAD_LIMIT_BYTES))
    assert len(slices) == 1
    assert slices[0].num_rows == 10


def test_an_empty_payload_yields_nothing() -> None:
    assert list(_byte_bounded_slices(_table(0, 10), 1024)) == []


def test_every_slice_is_within_the_limit() -> None:
    payload = _table(400, 1000)
    limit = 20_000
    slices = list(_byte_bounded_slices(payload, limit))
    assert len(slices) > 1
    # Every slice but a forced single-row one has to fit.
    assert all(s.nbytes <= limit for s in slices if s.num_rows > 1)


def test_the_slices_reconstruct_the_payload_exactly() -> None:
    """Completeness, in order, no duplication.

    This is the property the publish depends on: a split page must still
    write every row it was handed, or the state advanced after it would
    record rows that were never stored.
    """
    payload = _table(250, 900)
    slices = list(_byte_bounded_slices(payload, 15_000))
    assert pa.concat_tables(slices).equals(payload)
    assert sum(s.num_rows for s in slices) == payload.num_rows


def test_a_single_row_over_the_limit_is_yielded_alone() -> None:
    # Not refused, and not an infinite loop. The pool is larger than our
    # ceiling, so such a row may still succeed; failing it here to enforce
    # our own headroom would be the same mistake in the other direction.
    payload = _table(3, 50_000)
    slices = list(_byte_bounded_slices(payload, 1_000))
    assert [s.num_rows for s in slices] == [1, 1, 1]


def test_the_split_is_measured_rather_than_averaged() -> None:
    """The #592 shape itself: row size varies, so an average overshoots.

    One huge row followed by many tiny ones. The mean row size is small
    enough that a count derived from it would put the huge row in a slice
    far above the limit — which is exactly how a correct-looking
    `batch_size` sat 15 MB over a hard pool limit in production.
    """
    payload = pa.concat_tables([_table(1, 40_000), _table(200, 10)])
    limit = 20_000
    mean_row_bytes = payload.nbytes / payload.num_rows
    # An average-based splitter would take this many rows for slice one...
    averaged = int(limit / mean_row_bytes)
    assert averaged > 1, "precondition: the average alone would group the big row"

    slices = list(_byte_bounded_slices(payload, limit))
    # ...but measuring forces the oversized row to stand alone.
    assert slices[0].num_rows == 1
    assert slices[0].nbytes > limit
    assert all(s.nbytes <= limit for s in slices[1:] if s.num_rows > 1)
    assert pa.concat_tables(slices).equals(payload)


# ─── the upsert that uses it ────────────────────────────────────────────────


def _store(tmp_path: Path) -> LanceDBStore:
    config = parse_store_config({"type": "lancedb", "path": str(tmp_path / "lance")})
    return LanceDBStore(
        config,
        project_name="demo",
        target_name="dev",
        alias="primary",
        role=StoreRole.PUBLISH,
    )


def _spec() -> CollectionSpec:
    return CollectionSpec(
        logical_name="context",
        physical_name=PHYSICAL,
        id_field="chunk_id",
        text_fields=("text",),
        full_text_fields=(),
        attribute_fields=(),
        scalar_index_fields=(),
        display_fields=(),
        vector_field="embedding",
        vector_dimensions=2,
        distance_metric="cosine",
        vector_search="exact",
        vector_index=None,
        config_fingerprint="fingerprint",
        descriptor="{}",
        legacy_config_fingerprint="legacy",
        row_fingerprint="row-fp",
        arrow_schema=pa.schema(
            [
                pa.field("chunk_id", pa.string(), nullable=False),
                pa.field("text", pa.string()),
                pa.field("embedding", pa.list_(pa.float32(), 2)),
            ]
        ),
    )


def _rows(count: int, width: int) -> list[IndexedRow]:
    return [
        IndexedRow(
            f"c{i}",
            {"chunk_id": f"c{i}", "text": "x" * width, "embedding": [0.1, 0.9]},
            f"f{i}",
        )
        for i in range(count)
    ]


class _CountingTable:
    """Delegates to a real table, counting `merge_insert` calls."""

    def __init__(self, table: Any, calls: list[int]) -> None:
        self._table = table
        self._calls = calls

    def merge_insert(self, *args: Any, **kwargs: Any) -> Any:
        self._calls.append(1)
        return self._table.merge_insert(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._table, name)


def _count_merges(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    original = LanceDBStore._open_owned_table

    def counting(self: LanceDBStore, name: str) -> Any:
        return _CountingTable(original(self, name), calls)

    monkeypatch.setattr(LanceDBStore, "_open_owned_table", counting)
    return calls


def test_an_oversized_page_becomes_several_merges_and_still_lands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fix, end to end against a real LanceDB table.

    The limit is lowered rather than the payload raised to 64 MB: the
    behaviour under test is the split, and a test that allocated the real
    ceiling would be slow for no extra coverage.
    """
    monkeypatch.setattr("stel.retrieval.lancedb.MERGE_PAYLOAD_LIMIT_BYTES", 8_000)
    store = _store(tmp_path)
    with store:
        store.create_collection(_spec())
        calls = _count_merges(monkeypatch)
        rows = _rows(60, 500)
        receipt = store.upsert(
            PHYSICAL, rows, id_field="chunk_id", mutation_digest="digest"
        )
        assert receipt.acknowledged
        assert len(receipt.outcomes) == len(rows)
        assert len(calls) > 1, "an oversized page must not be one merge_insert"
        written = store.inspect_collection(PHYSICAL)
        assert written is not None
        assert written.row_count == 60


def test_a_page_within_the_limit_is_still_one_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No extra round trips for the case that already worked.
    store = _store(tmp_path)
    with store:
        store.create_collection(_spec())
        calls = _count_merges(monkeypatch)
        store.upsert(
            PHYSICAL, _rows(20, 100), id_field="chunk_id", mutation_digest="digest"
        )
        assert len(calls) == 1


# ─── what splitting costs the capability model ──────────────────────────────


def test_lancedb_no_longer_claims_batch_atomicity(tmp_path: Path) -> None:
    """Splitting makes the atomicity claim false, so it is withdrawn.

    A page over the ceiling becomes several Lance transactions, and a failure
    part-way leaves the earlier ones committed. Claiming `ATOMIC_BATCH_MUTATION`
    would be false for exactly the pages that need the split.
    """
    with _store(tmp_path) as store:
        features = store.capabilities().features
    assert RetrievalFeature.ATOMIC_BATCH_MUTATION not in features


def test_lancedb_proves_exact_receipts_instead(tmp_path: Path) -> None:
    """The other proof the receipt contract accepts.

    `upsert` confirms every id it was handed is durably present before
    returning, and raises otherwise — so a receipt is never ahead of the
    store, which is what the publish loop gates state on.
    """
    with _store(tmp_path) as store:
        features = store.capabilities().features
    assert RetrievalFeature.EXACT_MUTATION_RECEIPTS in features


def test_duckdb_still_claims_atomicity(tmp_path: Path) -> None:
    # The withdrawal is specific to the store that had to split, not a
    # general weakening: DuckDB's batch really is one transaction.
    from stel.retrieval import DuckDBStore

    config = parse_store_config(
        {"type": "duckdb", "path": str(tmp_path / "retrieval.duckdb")}
    )
    store = DuckDBStore(
        config,
        project_name="demo",
        target_name="dev",
        alias="primary",
        role=StoreRole.PUBLISH,
    )
    with store:
        assert RetrievalFeature.ATOMIC_BATCH_MUTATION in store.capabilities().features


def test_a_store_proving_neither_receipt_guarantee_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The requirement is "either", not "neither".

    Relaxing the compiler from `ATOMIC_BATCH_MUTATION` to either proof must
    not relax it to nothing: a store that can promise no trustworthy receipt
    still cannot back incremental publication, because core gates state on
    that receipt. Written because a mutation removing the check entirely was
    caught by no existing test.
    """
    from stel.compiler import validate_project_contract, validate_retrieval_capabilities
    from stel.config import ConfigError, load_project
    from stel.profile import resolve_profile
    from tests.support_retrieval import write_project

    write_project(tmp_path)
    project, sources, models = load_project(tmp_path)
    validate_project_contract(project, sources, models, tmp_path)
    resolved = resolve_profile(project, tmp_path)

    original = LanceDBStore.capabilities

    def without_receipts() -> Any:
        full = original()
        return replace(
            full,
            features=full.features
            - {
                RetrievalFeature.ATOMIC_BATCH_MUTATION,
                RetrievalFeature.EXACT_MUTATION_RECEIPTS,
            },
        )

    monkeypatch.setattr(LanceDBStore, "capabilities", staticmethod(without_receipts))
    with pytest.raises(ConfigError, match="exact whole-batch receipts"):
        validate_retrieval_capabilities(models, project, resolved)


def test_either_receipt_guarantee_alone_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The point of the change: atomicity is sufficient but no longer
    # necessary, so a store proving only exact receipts passes preflight.
    from stel.compiler import validate_project_contract, validate_retrieval_capabilities
    from stel.config import load_project
    from stel.profile import resolve_profile
    from tests.support_retrieval import write_project

    write_project(tmp_path)
    project, sources, models = load_project(tmp_path)
    validate_project_contract(project, sources, models, tmp_path)
    resolved = resolve_profile(project, tmp_path)

    original = LanceDBStore.capabilities

    def atomic_only() -> Any:
        full = original()
        return replace(
            full,
            features=(full.features - {RetrievalFeature.EXACT_MUTATION_RECEIPTS})
            | {RetrievalFeature.ATOMIC_BATCH_MUTATION},
        )

    monkeypatch.setattr(LanceDBStore, "capabilities", staticmethod(atomic_only))
    validate_retrieval_capabilities(models, project, resolved)


def test_a_split_page_upserts_idempotently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-running a split page must not duplicate rows.

    This is what makes splitting safe: the publish advances state only after
    the write, so a page that fails part-way is republished whole, and
    `merge_insert` on the id has to absorb the rows that already landed.
    """
    monkeypatch.setattr("stel.retrieval.lancedb.MERGE_PAYLOAD_LIMIT_BYTES", 8_000)
    store = _store(tmp_path)
    with store:
        store.create_collection(_spec())
        rows = _rows(60, 500)
        for _ in range(2):
            store.upsert(
                PHYSICAL, rows, id_field="chunk_id", mutation_digest="digest"
            )
        written = store.inspect_collection(PHYSICAL)
        assert written is not None
        assert written.row_count == 60
