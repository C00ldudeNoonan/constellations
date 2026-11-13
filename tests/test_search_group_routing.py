"""Partition pruning must never discard a possible result (#778)."""

from __future__ import annotations

from datetime import date

import pytest

from stel.config.search_partition import SearchPartitionConfig
from stel.retrieval.base import RetrievalPredicate
from stel.retrieval.base import RetrievalPredicateOperator as Op
from stel.retrieval.group_routing import partition_may_match


@pytest.mark.parametrize(
    ("operator", "value", "expected"),
    [
        (Op.EQUAL, 10, True),
        (Op.EQUAL, 20, False),
        (Op.IN, (9, 20), False),
        (Op.IN, (9, 19), True),
        (Op.LESS_THAN, 10, False),
        (Op.LESS_THAN_OR_EQUAL, 10, True),
        (Op.GREATER_THAN, 20, False),
        (Op.GREATER_THAN_OR_EQUAL, 20, False),
        (Op.NOT_EQUAL, 15, True),
    ],
)
def test_half_open_partition_boundaries(
    operator: Op, value: int | tuple[int, ...], expected: bool
) -> None:
    partition = SearchPartitionConfig(field="year", type="integer", lower=10, upper=20)
    predicate = RetrievalPredicate("year", operator, value)
    assert partition_may_match(partition, (predicate,)) is expected


def test_unfiltered_null_and_date_partitions() -> None:
    nulls = SearchPartitionConfig(field="day", type="date", nulls=True)
    recent = SearchPartitionConfig(field="day", type="date", lower=date(2024, 1, 1))
    assert partition_may_match(nulls, ())
    predicate = RetrievalPredicate("day", Op.LESS_THAN, date(2024, 1, 1))
    assert not partition_may_match(nulls, (predicate,))
    assert not partition_may_match(recent, (predicate,))
    other = RetrievalPredicate("other", Op.EQUAL, "anything")
    assert partition_may_match(nulls, (other,))
    # An unexpectedly typed predicate is never used to prune.
    assert partition_may_match(recent, (RetrievalPredicate("day", Op.EQUAL, "2024"),))
