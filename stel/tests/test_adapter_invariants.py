"""Adapter-neutral invariant contracts (issue #190, Workstream C).

These pin the warehouse-independent invariants shared by every adapter, so a
new adapter (#186 MotherDuck, #187 Snowflake) inherits one enforceable
contract instead of re-implementing — and re-testing — the policy per backend.
The schema-change planner formerly lived in and was tested against the
BigQuery adapter; it now lives in `adapters.base` and is exercised here once.
"""

from __future__ import annotations

import pytest

from stel.adapters import AdapterError, SchemaChangePlan
from stel.adapters.base import plan_schema_change


def test_plan_no_drift() -> None:
    plan = plan_schema_change(["a", "b"], ["a", "b"], "fail", "t")
    assert plan == SchemaChangePlan(["a", "b"], False, [])


def test_plan_fail_on_new_and_removed() -> None:
    with pytest.raises(AdapterError, match="full-refresh"):
        plan_schema_change(["a"], ["a", "b"], "fail", "t")
    with pytest.raises(AdapterError, match="removed columns"):
        plan_schema_change(["a", "b"], ["a"], "fail", "t")


def test_plan_append_new_columns() -> None:
    plan = plan_schema_change(["a"], ["a", "b"], "append_new_columns", "t")
    assert plan.columns_to_load == ["a", "b"]
    assert plan.allow_field_addition
    # the new columns an adapter must materialize before the load
    assert plan.columns_to_add == ["b"]
    # removed-only drift needs no field addition
    plan = plan_schema_change(["a", "b"], ["a"], "append_new_columns", "t")
    assert not plan.allow_field_addition
    assert plan.columns_to_add == []


def test_plan_ignore_drops_new_columns() -> None:
    plan = plan_schema_change(["a"], ["a", "b"], "ignore", "t")
    assert plan.columns_to_load == ["a"]
    assert plan.columns_to_add == []


def test_plan_unknown_policy() -> None:
    with pytest.raises(AdapterError, match="Unknown on_schema_change"):
        plan_schema_change(["a"], ["a", "b"], "sync_all", "t")
