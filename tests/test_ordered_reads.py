"""Ordered snapshot reads (issue #749).

An embed window is cut from the upstream stream, and the layout-pruned MERGE
and reuse read-back prune only as far as the window is local in the target's
partition and cluster columns. The read therefore has to be orderable, on every
adapter, with one definition of the order.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from stel.adapters import (
    AdapterCapabilityError,
    AdapterError,
    LayoutPruningColumns,
    WarehouseCapability,
    create_adapter,
    parse_warehouse_config,
)
from stel.adapters.base import TableReadRequest
from stel.execution.embed import _layout_read_order


def _adapter(tmp_path: Path) -> Any:
    config = parse_warehouse_config(
        {"type": "duckdb", "path": str(tmp_path / "w.duckdb"), "schema": "docs"}
    )
    return create_adapter(config)


def _ids(adapter: Any, **kwargs: Any) -> list[str]:
    out: list[str] = []
    with adapter.table_snapshot("t", batch_size=2, **kwargs) as snapshot:
        for batch in snapshot:
            frame = pl.from_arrow(batch)
            assert isinstance(frame, pl.DataFrame)
            out.extend(frame["id"].to_list())
    return out


def test_an_ordered_read_is_ascending_in_the_named_columns_with_nulls_first(
    tmp_path: Path,
) -> None:
    """The contract is one order on every adapter. DuckDB sorts NULLs last by
    default and BigQuery first, so the adapters state it rather than inherit
    it; a NULL layout value sorting differently on the two would put the same
    rows in different windows."""
    with _adapter(tmp_path) as adapter:
        adapter.materialize_full(
            "t",
            pl.DataFrame(
                {
                    "id": ["a", "b", "c", "d", "e"],
                    "d": [
                        date(2026, 2, 1),
                        date(2026, 1, 1),
                        None,
                        date(2026, 1, 1),
                        date(2026, 2, 1),
                    ],
                    "s": ["x", "y", "z", "x", "a"],
                }
            ),
        )

        assert _ids(adapter, order_by=["d", "s", "id"]) == ["c", "d", "b", "e", "a"]


def test_the_last_order_column_makes_a_rerun_yield_the_same_sequence(
    tmp_path: Path,
) -> None:
    """Resume cuts windows from the rows still to do; that only reproduces the
    original windows if ties in the layout columns break the same way."""
    with _adapter(tmp_path) as adapter:
        adapter.materialize_full(
            "t",
            pl.DataFrame(
                {
                    "id": ["d", "b", "a", "c"],
                    "d": [date(2026, 1, 1)] * 4,
                }
            ),
        )

        first = _ids(adapter, order_by=["d", "id"])
        assert first == ["a", "b", "c", "d"]
        assert _ids(adapter, order_by=["d", "id"]) == first


def test_an_order_column_the_table_lacks_is_refused_at_open(tmp_path: Path) -> None:
    with _adapter(tmp_path) as adapter:
        adapter.materialize_full("t", pl.DataFrame({"id": ["a"]}))

        with pytest.raises(AdapterError, match="missing column"):
            _ids(adapter, order_by=["nope", "id"])


@pytest.mark.parametrize("order_by", [("",), ("id", "id")])
def test_a_malformed_order_is_refused(order_by: tuple[str, ...]) -> None:
    with pytest.raises(AdapterError, match="order_by"):
        TableReadRequest("t", None, 1, (), None, order_by)


def test_an_adapter_without_the_capability_refuses_an_ordered_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A silently unordered read would hand the consumer windows it believes
    are local and are not, and the cost would show up on the bill."""
    with _adapter(tmp_path) as adapter:
        adapter.materialize_full("t", pl.DataFrame({"id": ["a"]}))
        without = frozenset(
            type(adapter).capabilities() - {WarehouseCapability.ORDERED_TABULAR_READS}
        )
        monkeypatch.setattr(
            type(adapter), "capabilities", classmethod(lambda cls: without)
        )

        with pytest.raises(AdapterCapabilityError, match="ordered_tabular_reads"):
            _ids(adapter, order_by=["id"])


def test_an_unordered_request_hashes_as_it_did_before_order_by_existed() -> None:
    """A BigQuery resume point is keyed by the request fingerprint (issue #614).
    Adding a key to every request's payload would orphan every point in flight,
    so the key appears only on a request that orders."""
    plain = TableReadRequest("t", ("id",), 5, (), "id", ())._fingerprint_payload()
    assert set(plain) == {"table", "columns", "predicates", "key_column"}

    ordered = TableReadRequest("t", ("id",), 5, (), "id", ("id",))
    assert ordered._fingerprint_payload()["order_by"] == ["id"]


# ─── what embed asks for ─────────────────────────────────────────────────────


class _Orderable:
    def __init__(self, *, ordered: bool) -> None:
        self._ordered = ordered

    def supports(self, capability: WarehouseCapability) -> bool:
        return self._ordered and capability is WarehouseCapability.ORDERED_TABULAR_READS


def test_embed_reads_in_partition_then_cluster_order_ending_in_the_id() -> None:
    order = _layout_read_order(
        _Orderable(ordered=True),  # ty: ignore[invalid-argument-type]
        LayoutPruningColumns("filing_date_dt", ("symbol", "form_type")),
        upstream_columns=["chunk_id", "text", "filing_date_dt", "symbol", "form_type"],
        id_field="chunk_id",
    )

    assert order == ("filing_date_dt", "symbol", "form_type", "chunk_id")


def test_embed_orders_only_by_layout_columns_the_upstream_carries() -> None:
    order = _layout_read_order(
        _Orderable(ordered=True),  # ty: ignore[invalid-argument-type]
        LayoutPruningColumns("filing_date_dt", ("symbol", "form_type")),
        upstream_columns=["chunk_id", "text", "symbol"],
        id_field="chunk_id",
    )

    assert order == ("symbol", "chunk_id")


def test_embed_does_not_repeat_the_id_when_it_is_a_layout_column() -> None:
    order = _layout_read_order(
        _Orderable(ordered=True),  # ty: ignore[invalid-argument-type]
        LayoutPruningColumns(None, ("symbol", "chunk_id")),
        upstream_columns=["chunk_id", "symbol"],
        id_field="chunk_id",
    )

    assert order == ("symbol", "chunk_id")


@pytest.mark.parametrize(
    ("ordered", "pruning"),
    [
        (False, LayoutPruningColumns("d", ("s",))),
        (True, LayoutPruningColumns(None, ())),
        (True, LayoutPruningColumns("gone", ("also_gone",))),
    ],
    ids=["adapter-cannot-order", "no-declared-layout", "layout-not-in-upstream"],
)
def test_embed_reads_unordered_when_ordering_could_not_help(
    ordered: bool, pruning: LayoutPruningColumns
) -> None:
    """A sort nothing prunes on is cost without benefit, so these read as they
    always have."""
    assert (
        _layout_read_order(
            _Orderable(ordered=ordered),  # ty: ignore[invalid-argument-type]
            pruning,
            upstream_columns=["chunk_id", "text"],
            id_field="chunk_id",
        )
        == ()
    )
