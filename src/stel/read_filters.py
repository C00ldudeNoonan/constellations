"""Parse typed read predicates shared by execution and pending plans."""

from __future__ import annotations

import json
from collections.abc import Sequence

from .adapters import ReadPredicate, ReadPredicateOperator
from .execution import RunError

_READ_FILTER_OPERATORS = {
    "eq": ReadPredicateOperator.EQUAL,
    "ne": ReadPredicateOperator.NOT_EQUAL,
    "lt": ReadPredicateOperator.LESS_THAN,
    "le": ReadPredicateOperator.LESS_THAN_OR_EQUAL,
    "gt": ReadPredicateOperator.GREATER_THAN,
    "ge": ReadPredicateOperator.GREATER_THAN_OR_EQUAL,
    "in": ReadPredicateOperator.IN,
}


def parse_read_filters(
    read_filter: Sequence[tuple[str, str, str]],
) -> tuple[ReadPredicate, ...]:
    predicates: list[ReadPredicate] = []
    for column, operator_name, raw_value in read_filter:
        operator = _READ_FILTER_OPERATORS.get(operator_name)
        if operator is None:
            raise RunError(
                f"--read-filter operator '{operator_name}' is not one of "
                f"{sorted(_READ_FILTER_OPERATORS)}"
            )
        value: str | tuple[str, ...]
        if operator is ReadPredicateOperator.IN:
            try:
                decoded = json.loads(raw_value)
            except json.JSONDecodeError:
                raise RunError(
                    "--read-filter 'in' takes a JSON array of strings"
                ) from None
            if (
                not isinstance(decoded, list)
                or not decoded
                or not all(isinstance(item, str) for item in decoded)
            ):
                raise RunError(
                    "--read-filter 'in' takes a non-empty JSON array of strings"
                )
            value = tuple(decoded)
        else:
            value = raw_value
        predicates.append(ReadPredicate(column, operator, value))
    return tuple(predicates)
