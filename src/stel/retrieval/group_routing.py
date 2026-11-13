"""Conservatively route validated predicates to compiled search partitions."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from typing import cast

from ..config.search_partition import PartitionBound, SearchPartitionConfig, bound_sort_key
from .base import RetrievalPredicate, RetrievalPredicateOperator


def partition_may_match(
    partition: SearchPartitionConfig, predicates: Sequence[RetrievalPredicate]
) -> bool:
    """Return false only when a predicate proves this partition cannot match.

    Predicates have already been validated and coerced against the common
    attribute contract. Unsupported operators never prune a member.
    """
    for predicate in predicates:
        if predicate.field == partition.field and not _may_match(partition, predicate):
            return False
    return True


def _may_match(partition: SearchPartitionConfig, predicate: RetrievalPredicate) -> bool:
    op = predicate.operator
    if op not in {
        RetrievalPredicateOperator.EQUAL,
        RetrievalPredicateOperator.IN,
        RetrievalPredicateOperator.LESS_THAN,
        RetrievalPredicateOperator.LESS_THAN_OR_EQUAL,
        RetrievalPredicateOperator.GREATER_THAN,
        RetrievalPredicateOperator.GREATER_THAN_OR_EQUAL,
    }:
        return True
    values = predicate.value if isinstance(predicate.value, tuple) else (predicate.value,)
    if any(not _matches_type(partition.type, value) for value in values):
        return True
    if partition.nulls:
        return False
    keys = [bound_sort_key(cast(PartitionBound, value)) for value in values]
    lower = bound_sort_key(partition.lower) if partition.lower is not None else None
    upper = bound_sort_key(partition.upper) if partition.upper is not None else None
    if op in {RetrievalPredicateOperator.EQUAL, RetrievalPredicateOperator.IN}:
        return any(
            (lower is None or lower <= key) and (upper is None or key < upper)
            for key in keys
        )
    value = keys[0]
    if op == RetrievalPredicateOperator.LESS_THAN:
        return lower is None or lower < value
    if op == RetrievalPredicateOperator.LESS_THAN_OR_EQUAL:
        return lower is None or lower <= value
    return upper is None or value < upper


def _matches_type(kind: str, value: object) -> bool:
    if kind == "date":
        return isinstance(value, date) and not isinstance(value, datetime)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    return kind == "string" and isinstance(value, str)
