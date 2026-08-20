"""Enum fields declared once, derived everywhere (issue #304).

A classification task used to write its label list in three unrelated places:
the prompt text, an `accepted_values` test, and — implicitly — whatever the
provider's structured output happened to enforce. Nothing failed loudly when
they drifted; the taxonomy just rotted. These tests pin the single declaration
and each thing derived from it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from stel.adapters import WarehouseAdapter, create_adapter, parse_warehouse_config
from stel.backends.llm_backend import _apply_enum_portability, _input_schema
from stel.checks.runner import run_model_tests
from stel.config.model import FieldConfig, ModelConfig
from stel.llm_map import build_fields_spec

_LABELS = ["churn_risk", "expansion", "pricing", "support", "none"]


def _signal_field() -> FieldConfig:
    return FieldConfig(name="signal", type="enum", values=list(_LABELS))


# ─── the declaration ────────────────────────────────────────────────────────


def test_enum_field_declares_its_closed_set() -> None:
    field = _signal_field()
    assert field.data_type == "enum"
    assert field.values == _LABELS


def test_enum_without_values_is_rejected() -> None:
    # An enum with no closed set constrains nothing, so it is a typo, not a
    # degenerate case worth supporting.
    with pytest.raises(ValueError, match="declares no `values:`"):
        FieldConfig(name="signal", type="enum")


def test_values_without_enum_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="use `type: enum`"):
        FieldConfig(name="signal", type="string", values=["a"])


def test_duplicate_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="twice"):
        FieldConfig(name="signal", type="enum", values=["a", "a"])


# ─── derivation 1: the provider output schema ───────────────────────────────


def test_enum_reaches_the_provider_schema_as_a_constraint() -> None:
    spec = build_fields_spec([_signal_field(), FieldConfig(name="evidence")])

    schema = _input_schema(spec)

    assert schema["properties"]["signal"] == {"type": "string", "enum": _LABELS}
    # A field with no declared set is untouched — this is a no-op by default.
    assert schema["properties"]["evidence"] == {"type": "string"}


def test_enum_is_a_string_to_the_warehouse() -> None:
    # `enum` is stel's declaration, not a column type; the value that lands is
    # a string.
    assert build_fields_spec([_signal_field()])[0]["type"] == "string"


# ─── derivation 2: the accepted_values check ────────────────────────────────


@pytest.fixture
def signals(tmp_path: Path) -> Iterator[WarehouseAdapter]:
    cfg = parse_warehouse_config(
        {"type": "duckdb", "path": str(tmp_path / "e.duckdb"), "schema": "main"}
    )
    with create_adapter(cfg) as adapter:
        adapter.materialize_full(
            "classified",
            pl.DataFrame({"signal": ["churn_risk", "pricing", "invented_label"]}),
        )
        yield adapter


def _model(tests: list[Any] | None = None) -> ModelConfig:
    return ModelConfig(
        name="classified", fields=[_signal_field()], tests=tests or []
    )


def test_enum_field_is_checked_without_a_hand_written_test(
    signals: WarehouseAdapter,
) -> None:
    # No `tests:` declared at all — the check comes from the field.
    results = run_model_tests(_model(), signals)

    assert [r.test_name for r in results] == ["accepted_values"]
    assert results[0].status == "fail"
    assert results[0].column == "signal"
    # The check reports the count and the allowed set, not the stray value.
    assert "1 values outside" in results[0].message


def test_a_conforming_column_passes(tmp_path: Path) -> None:
    cfg = parse_warehouse_config(
        {"type": "duckdb", "path": str(tmp_path / "ok.duckdb"), "schema": "main"}
    )
    with create_adapter(cfg) as adapter:
        adapter.materialize_full(
            "classified", pl.DataFrame({"signal": ["pricing", "none"]})
        )

        results = run_model_tests(_model(), adapter)

    assert [r.status for r in results] == ["pass"]


def test_an_explicit_test_is_not_duplicated(signals: WarehouseAdapter) -> None:
    # Theirs runs, not two of them.
    model = _model(
        [{"accepted_values": {"column": "signal", "values": list(_LABELS)}}]
    )

    results = run_model_tests(model, signals)

    assert len(results) == 1


def test_fields_without_values_derive_nothing(signals: WarehouseAdapter) -> None:
    model = ModelConfig(name="classified", fields=[FieldConfig(name="signal")])

    assert run_model_tests(model, signals) == []


def test_disagreeing_explicit_test_warns_at_compile_time(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The drift this feature exists to prevent, caught where it is visible.

    Which list is right is the author's call, so this reports rather than
    deciding for them.
    """
    from stel.compiler import _validate_tests

    model = _model(
        [
            {"accepted_values": {"column": "signal", "values": ["churn_risk", "gone"]}}
        ]
    )

    with caplog.at_level(logging.WARNING):
        _validate_tests(model, set(), {"classified"}, tmp_path)

    assert "checking a different taxonomy" in caplog.text


def test_agreeing_explicit_test_does_not_warn(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from stel.compiler import _validate_tests

    model = _model(
        [
            {"accepted_values": {"column": "signal", "values": list(reversed(_LABELS))}}
        ]
    )

    with caplog.at_level(logging.WARNING):
        _validate_tests(model, set(), {"classified"}, tmp_path)

    assert caplog.text == ""


# ─── derivation 3: the prompt fallback ──────────────────────────────────────


class _SchemaEnumProvider:
    supports_schema_enum = True


class _NoSchemaEnumProvider:
    supports_schema_enum = False


def test_a_provider_that_carries_enums_gets_no_prompt_addendum() -> None:
    # Asking politely on top of a hard constraint only spends tokens.
    spec = build_fields_spec([_signal_field()])

    fields, system = _apply_enum_portability(_SchemaEnumProvider(), spec, "SYS")

    assert fields == spec
    assert system == "SYS"


def test_a_provider_without_schema_enums_gets_the_labels_in_the_prompt() -> None:
    spec = build_fields_spec([_signal_field()])

    fields, system = _apply_enum_portability(_NoSchemaEnumProvider(), spec, "SYS")

    # Stripped rather than sent to be ignored or rejected...
    assert "enum" not in fields[0]
    # ...and communicated instead.
    assert "Allowed values:" in system
    for label in _LABELS:
        assert label in system


def test_the_fallback_is_a_no_op_without_declared_values() -> None:
    spec = build_fields_spec([FieldConfig(name="evidence")])

    fields, system = _apply_enum_portability(_NoSchemaEnumProvider(), spec, "SYS")

    assert fields == spec
    assert system == "SYS"


def test_shipped_providers_can_carry_schema_enums() -> None:
    """Every provider stel ships forwards a JSON-Schema-shaped payload.

    If one of these ever flips, the fallback above is what covers it — but the
    flip should be deliberate, not silent.
    """
    from stel.providers.anthropic import AnthropicInferenceProvider
    from stel.providers.vertex import VertexInferenceProvider
    from stel.providers.vllm import VLLMInferenceProvider

    for provider in (
        AnthropicInferenceProvider,
        VertexInferenceProvider,
        VLLMInferenceProvider,
    ):
        assert provider.supports_schema_enum
