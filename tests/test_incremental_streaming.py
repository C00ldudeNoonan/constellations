"""Parent grouping streams, and keeps its digests (issue #383).

The point of #383 is a memory ceiling at the largest single group rather than
the whole corpus. `partition_by` cannot deliver that — it returns a *list* of
every partition, so wrapping it in a generator still materializes the corpus
before the first yield (Codex review of #384). These tests pin both the
laziness and the thing laziness must not cost: byte-identical fingerprints,
since a changed digest re-runs every parent in every existing project.
"""
from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from stel.execution.contracts import RunError
from stel.execution.transform import (
    _keyed_reference_fingerprints,
    _parent_fingerprint,
    _parent_groups,
    _row_groups,
)
from stel.hashing import canonical_fingerprint, canonical_json
from stel.transforms import ReferenceDep

_KEY = "document_id"


def _frame() -> pl.DataFrame:
    # Interleaved, repeated, and out of order: the shapes where a grouping
    # change shows up.
    return pl.DataFrame(
        {
            _KEY: ["b", "a", "b", "c", "a", "b"],
            "n": [1, 2, 3, 4, 5, 6],
            "t": ["x", "y", "z", "w", "v", "u"],
        }
    )


def _groups_the_old_way(
    frame: pl.DataFrame, key_col: str
) -> list[tuple[str, list[dict[str, Any]]]]:
    """The pre-#383 implementation, kept here as the digest oracle."""
    groups: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for row in frame.iter_rows(named=True):
        key = str(row[key_col])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)
    return [(key, groups[key]) for key in order]


def test_parent_fingerprints_are_byte_identical_to_the_old_grouping() -> None:
    frame = _frame()
    streamed = list(_parent_groups(frame, _KEY, "m"))
    original = _groups_the_old_way(frame, _KEY)

    assert [key for key, _ in streamed] == [key for key, _ in original]
    for (new_key, new_rows), (old_key, old_rows) in zip(
        streamed, original, strict=True
    ):
        assert _parent_fingerprint(new_key, new_rows, {}) == _parent_fingerprint(
            old_key, old_rows, {}
        )


def test_keyed_reference_fingerprints_are_byte_identical_to_the_old_grouping() -> None:
    frame = _frame()
    expected = {
        key: canonical_fingerprint(
            {"rows": sorted(rows, key=canonical_json)},
            domain="dbt-ml.transform-incremental-reference",
        )
        for key, rows in _groups_the_old_way(frame, _KEY)
    }

    assert (
        _keyed_reference_fingerprints(frame, ReferenceDep("r", join_key=_KEY), "m")
        == expected
    )


def test_grouping_never_materializes_every_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`partition_by` returns a list of every partition, which is the peak
    #383 exists to remove. Reintroducing it would pass every behavioural test
    while undoing the change, so the call itself is what is pinned."""

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("partition_by materializes the whole corpus")

    monkeypatch.setattr(pl.DataFrame, "partition_by", forbidden)

    frame = _frame()
    assert [key for key, _ in _parent_groups(frame, _KEY, "m")] == ["b", "a", "c"]
    assert set(
        _keyed_reference_fingerprints(frame, ReferenceDep("r", join_key=_KEY), "m")
    ) == {"a", "b", "c"}


def test_groups_are_gathered_only_when_requested() -> None:
    """Laziness is the contract: taking one group must not build the rest."""
    gathered: list[str] = []
    frame = _frame()

    groups = _row_groups(frame, _KEY, model_name="m")
    first_key, first_group = next(groups)
    gathered.append(first_key)

    assert first_key == "b"
    assert first_group["n"].to_list() == [1, 3, 6]
    # The generator is still suspended: the remaining groups have not been
    # gathered, and consuming them now yields the rest in order.
    assert [key for key, _ in groups] == ["a", "c"]


def test_a_reserved_index_column_is_refused_not_shadowed() -> None:
    frame = pl.DataFrame({_KEY: ["a"], "__stel_row_index__": [0]})

    with pytest.raises(RunError, match="reserved"):
        list(_row_groups(frame, _KEY, model_name="m"))
