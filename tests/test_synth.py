from __future__ import annotations

import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from stel.synth import (
    generate_arxiv_papers,
    generate_invoice_pdfs,
    generate_invoice_texts,
    generate_invoices,
    generate_posts,
    generate_product_pages,
    generate_support_emails,
    generate_support_tickets,
)


def test_utc_zone_database_is_available() -> None:
    """Faker no longer supplies the IANA database on Windows (issue #787)."""
    assert ZoneInfo("UTC").key == "UTC"


def test_generate_invoices_writes_count(tmp_path: Path) -> None:
    paths = generate_invoices(7, tmp_path, seed=1)
    assert len(paths) == 7
    assert sorted(p.name for p in paths) == [f"invoice_{i:05d}.json" for i in range(7)]


def test_generate_invoices_deterministic(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    generate_invoices(5, dir_a, seed=123)
    generate_invoices(5, dir_b, seed=123)
    for p_a in dir_a.glob("*.json"):
        p_b = dir_b / p_a.name
        assert p_a.read_text() == p_b.read_text()


def test_generate_invoices_schema(tmp_path: Path) -> None:
    paths = generate_invoices(3, tmp_path, seed=42)
    for p in paths:
        data = json.loads(p.read_text())
        assert set(data) == {
            "invoice_id",
            "vendor",
            "issue_date",
            "currency",
            "line_items",
            "total",
        }
        assert data["invoice_id"].startswith("INV-")
        assert isinstance(data["line_items"], list)
        assert len(data["line_items"]) >= 1
        for li in data["line_items"]:
            assert {"description", "qty", "unit_price"} <= set(li)
        assert data["total"] > 0


@pytest.mark.parametrize(
    "generate",
    [
        generate_arxiv_papers,
        generate_invoice_pdfs,
        generate_invoice_texts,
        generate_invoices,
        generate_posts,
        generate_product_pages,
        generate_support_emails,
        generate_support_tickets,
    ],
)
def test_seed_generators_are_reproducible(tmp_path: Path, generate) -> None:
    """Every seed type produces stable bytes for a seed (issue #787)."""
    first = generate(2, tmp_path / "first", seed=17)
    repeated = generate(2, tmp_path / "repeated", seed=17)
    different = generate(2, tmp_path / "different", seed=18)
    assert [path.read_bytes() for path in first] == [path.read_bytes() for path in repeated]
    assert [path.read_bytes() for path in first] != [path.read_bytes() for path in different]
