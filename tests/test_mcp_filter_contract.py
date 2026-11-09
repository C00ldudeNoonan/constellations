"""The MCP catalog and request agree on array membership (issue #723)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import ValidationError

from stel.mcp_server.catalog import _filter_capability
from stel.mcp_server.contracts import BusinessFilter, FilterOperator
from stel.mcp_server.service import ContextService
from stel.search import SearchFilterOperator


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


@pytest.mark.parametrize(
    ("spelling", "canonical"),
    [("gte", FilterOperator.GREATER_THAN_OR_EQUAL), ("lte", FilterOperator.LESS_THAN_OR_EQUAL)],
)
def test_gte_and_lte_normalize_to_the_advertised_operators(
    spelling: str, canonical: FilterOperator
) -> None:
    """ALE-571: a caller may say `gte`/`lte`; the catalog, log and stores say `ge`/`le`."""
    parsed = BusinessFilter.model_validate_json(
        json.dumps({"field": "filing_date_dt", "operator": spelling, "value": "2025-01-01"})
    )
    assert parsed.operator is canonical
    assert parsed.model_dump(mode="json")["operator"] == canonical.value


def test_an_alias_reaches_the_search_layer_as_the_canonical_operator() -> None:
    """The alias must be gone before capability checks and `SearchFilter` see it."""
    resource = SimpleNamespace(
        business_filters=(_filter_capability({"name": "filing_date_dt", "data_type": "date"}),)
    )
    service = ContextService.__new__(ContextService)
    result = service._business_filter(
        cast("Any", resource),
        BusinessFilter.model_validate(
            {"field": "filing_date_dt", "operator": "gte", "value": "2025-01-01"}
        ),
    )
    assert result.operator is SearchFilterOperator.GREATER_THAN_OR_EQUAL


@pytest.mark.parametrize("spelling", ["ge", "le", "gt", "lt"])
def test_canonical_range_spellings_still_validate(spelling: str) -> None:
    parsed = BusinessFilter.model_validate(
        {"field": "filing_date_dt", "operator": spelling, "value": "2025-01-01"}
    )
    assert parsed.operator.value == spelling


@pytest.mark.parametrize("spelling", ["GTE", "ge ", "greater", "gtee"])
def test_unknown_operator_spellings_are_refused(spelling: str) -> None:
    with pytest.raises(ValidationError):
        BusinessFilter.model_validate(
            {"field": "filing_date_dt", "operator": spelling, "value": "2025-01-01"}
        )


def test_a_range_alias_refuses_an_array_value() -> None:
    with pytest.raises(ValidationError, match="scalar filters accept exactly one value"):
        BusinessFilter.model_validate(
            {"field": "filing_date_dt", "operator": "gte", "value": ["2025-01-01"]}
        )
