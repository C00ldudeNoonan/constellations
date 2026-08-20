from __future__ import annotations

import json
from pathlib import Path

import pytest

from stel.backends import BackendNotFoundError, get_backend, list_backends


def test_json_backend_registered() -> None:
    assert "json" in list_backends()


def test_unknown_backend() -> None:
    with pytest.raises(BackendNotFoundError):
        get_backend("does_not_exist")


def test_json_backend_extracts_requested_fields(tmp_path: Path) -> None:
    doc = tmp_path / "invoice_00001.json"
    doc.write_text(
        json.dumps(
            {
                "invoice_id": "INV-00001",
                "vendor": "Acme",
                "total": 99.99,
                "extra_field": "ignored",
            }
        )
    )
    backend = get_backend("json")
    result = backend.extract(doc, {"fields": ["invoice_id", "vendor", "total"]})
    assert result.fields == {
        "invoice_id": "INV-00001",
        "vendor": "Acme",
        "total": 99.99,
    }
    assert result.warnings == []


def test_json_backend_warns_on_missing_field(tmp_path: Path) -> None:
    doc = tmp_path / "doc.json"
    doc.write_text(json.dumps({"vendor": "Acme"}))
    backend = get_backend("json")
    result = backend.extract(doc, {"fields": ["vendor", "total"]})
    assert result.fields == {"vendor": "Acme", "total": None}
    assert any("total" in w for w in result.warnings)


def test_json_backend_no_fields_option_returns_all(tmp_path: Path) -> None:
    doc = tmp_path / "doc.json"
    payload = {"a": 1, "b": [1, 2], "c": {"nested": True}}
    doc.write_text(json.dumps(payload))
    backend = get_backend("json")
    result = backend.extract(doc, {})
    assert result.fields == payload


def test_json_backend_rejects_non_object(tmp_path: Path) -> None:
    doc = tmp_path / "doc.json"
    doc.write_text(json.dumps([1, 2, 3]))
    backend = get_backend("json")
    with pytest.raises(ValueError, match="JSON object"):
        backend.extract(doc, {})


def test_json_backend_supported_formats() -> None:
    backend = get_backend("json")
    assert ".json" in backend.supported_formats()
