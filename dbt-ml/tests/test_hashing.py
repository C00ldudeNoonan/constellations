from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import pytest

from dbt_ml.hashing import canonical_fingerprint, canonical_json


def test_mapping_order_is_canonical_and_sequence_order_is_semantic() -> None:
    first = {
        "access": {"groups": ["analyst", "admin"], "tenant": "economic-data"},
        "nullable": None,
    }
    reordered = {
        "nullable": None,
        "access": {"tenant": "economic-data", "groups": ["analyst", "admin"]},
    }
    different_sequence = {
        "access": {"groups": ["admin", "analyst"], "tenant": "economic-data"},
        "nullable": None,
    }

    assert canonical_json(first) == canonical_json(reordered)
    assert canonical_json(first) != canonical_json(different_sequence)


def test_missing_null_and_scalar_types_remain_distinct() -> None:
    values = [{}, {"value": None}, {"value": False}, {"value": 0}, {"value": "0"}]

    assert len({canonical_json(value) for value in values}) == len(values)


def test_decimal_encoding_is_explicit_and_preserves_representation() -> None:
    assert canonical_json(Decimal("1.00")) == ('["special","decimal.Decimal",["string","1.00"]]')
    assert canonical_json(Decimal("1.00")) != canonical_json(Decimal("1.0"))
    assert canonical_json(Decimal("NaN")) == canonical_json(Decimal("NaN"))
    assert canonical_json(Decimal("Infinity")) == canonical_json(Decimal("Infinity"))


def test_aware_datetimes_normalize_to_utc_and_naive_values_stay_distinct() -> None:
    instant_utc = datetime(2026, 7, 15, 20, 30, 45, 123456, tzinfo=UTC)
    instant_pacific = datetime(
        2026,
        7,
        15,
        12,
        30,
        45,
        123456,
        tzinfo=timezone(timedelta(hours=-8)),
    )
    naive = instant_utc.replace(tzinfo=None)

    assert canonical_json(instant_utc) == canonical_json(instant_pacific)
    assert canonical_json(instant_utc) != canonical_json(naive)
    assert canonical_json(naive) != canonical_json(naive.replace(fold=1))


def test_binary_types_are_deterministic_and_type_preserving() -> None:
    payload = b"\x00\xffmetadata"

    assert canonical_json(payload) == canonical_json(bytes(payload))
    assert (
        len(
            {
                canonical_json(payload),
                canonical_json(bytearray(payload)),
                canonical_json(memoryview(payload)),
                canonical_json(payload.hex()),
            }
        )
        == 4
    )


def test_fingerprint_domain_version_and_value_are_separate_inputs() -> None:
    value = {"document_id": "doc", "metadata": {"tenant": "a"}}
    baseline = canonical_fingerprint(value, domain="chunk-input")

    assert baseline == canonical_fingerprint(value, domain="chunk-input")
    assert baseline != canonical_fingerprint(value, domain="chunk-row")
    assert baseline != canonical_fingerprint(value, domain="chunk-input", version=2)
    assert baseline != canonical_fingerprint(
        {"document_id": "doc", "metadata": {"tenant": "b"}},
        domain="chunk-input",
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"domain": ""}, "domain"),
        ({"domain": "chunk-input", "version": 0}, "version"),
        ({"domain": "chunk-input", "digest_size": 0}, "digest_size"),
        ({"domain": "chunk-input", "digest_size": 65}, "digest_size"),
    ],
)
def test_fingerprint_rejects_invalid_framing(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        canonical_fingerprint("value", **kwargs)


def test_unsupported_values_fail_closed() -> None:
    with pytest.raises(TypeError, match="not fingerprintable"):
        canonical_json(object())
