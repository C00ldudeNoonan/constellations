"""The MCP catalog and request agree on array membership (issue #723)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from stel.mcp_server.catalog import _filter_capability
from stel.mcp_server.contracts import BusinessFilter


@pytest.mark.parametrize(
    ("data_type", "operators"),
    [
        ("array[string]", ("array_contains_any",)),
        ("boolean", ("eq", "ne", "in")),
        *[
            (kind, ("eq", "ne", "lt", "le", "gt", "ge", "in"))
            for kind in ("string", "integer", "float", "date", "timestamp")
        ],
    ],
)
def test_catalog_operators_match_the_attribute_shape(
    data_type: str,
    operators: tuple[str, ...],
) -> None:
    """Issue #723: overlap must not become a scalar capability either."""
    capability = _filter_capability({"name": "topic", "data_type": data_type})
    assert tuple(item.value for item in capability.operators) == operators


@pytest.mark.parametrize("values", [["macro"], ["macro", "labor"]])
def test_array_membership_accepts_a_json_array(values: list[str]) -> None:
    """Issue #723: actual MCP JSON must survive request validation."""
    request = BusinessFilter.model_validate_json(
        json.dumps(
            {"field": "topics", "operator": "array_contains_any", "value": values},
        )
    )
    assert request.value == tuple(values)


@pytest.mark.parametrize("operator", ["in", "array_contains_any"])
@pytest.mark.parametrize("value", [[], "macro"])
def test_set_operators_require_a_nonempty_array(operator: str, value: object) -> None:
    """Issue #723: invalid shapes fail before retrieval is called."""
    with pytest.raises(ValidationError, match=f"{operator} filters require one or more values"):
        BusinessFilter.model_validate({"field": "topics", "operator": operator, "value": value})


def test_scalar_operators_still_refuse_arrays() -> None:
    with pytest.raises(ValidationError, match="scalar filters accept exactly one value"):
        BusinessFilter.model_validate({"field": "title", "operator": "eq", "value": ["macro"]})
