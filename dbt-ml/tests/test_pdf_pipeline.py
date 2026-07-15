"""End-to-end pipeline test for examples/pdf_invoice_pipeline with the LLM mocked.

Exercises the full chain — synthetic PDFs → pdf backend → DuckDB raw_pdf_text →
transform with TransformContext → LLM helper (cached + mocked) → extracted_invoices —
so the wiring is locked in even without an API key.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import duckdb
import pytest

from dbt_ml.backends import llm_backend
from dbt_ml.credentials import CredentialReference
from dbt_ml.runner import run_project
from dbt_ml.synth import generate_invoice_pdfs


@pytest.fixture(autouse=True)
def _default_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")


@pytest.fixture
def pdf_project(tmp_path: Path) -> Path:
    repo = Path(__file__).resolve().parents[1]
    src = repo / "examples" / "pdf_invoice_pipeline"
    dst = tmp_path / "pdf_proj"
    shutil.copytree(
        src, dst, ignore=shutil.ignore_patterns("data", "target", "__pycache__")
    )
    return dst


def test_pdf_pipeline_end_to_end(
    monkeypatch: pytest.MonkeyPatch, pdf_project: Path
) -> None:
    """PDFs → raw_pdf_text → extracted_invoices, with the API mocked."""
    generate_invoice_pdfs(3, pdf_project / "data" / "invoices_pdf", seed=1)

    call_count = {"n": 0}

    def fake_api(
        content: str, model: str, system: str, fields_spec: list, **kwargs: object
    ) -> dict:
        call_count["n"] += 1
        return {
            "invoice_id": f"INV-FROMTEXT-{call_count['n']}",
            "vendor": "Mocked Vendor",
            "issue_date": "2026-01-01",
            "currency": "USD",
            "total": 99.99 * call_count["n"],
        }

    monkeypatch.setattr(llm_backend, "_default_call_api", fake_api)

    results = run_project(pdf_project)
    by_name = {r.model_name: r for r in results}
    assert by_name["raw_pdf_text"].documents_processed == 3
    assert by_name["raw_pdf_text"].rows_written == 3
    assert by_name["extracted_invoices"].rows_written == 3
    assert call_count["n"] == 3, "expected one API call per row on first run"

    db = pdf_project / "target" / "dbt_ml.duckdb"
    con = duckdb.connect(str(db), read_only=True)
    try:
        rows = con.execute(
            'SELECT vendor, total FROM "dbt_ml"."pdf_invoices".extracted_invoices'
        ).fetchall()
    finally:
        con.close()
    assert len(rows) == 3
    assert all(r[0] == "Mocked Vendor" for r in rows)


def test_pdf_pipeline_caches_llm_calls(
    monkeypatch: pytest.MonkeyPatch, pdf_project: Path
) -> None:
    """Second run over the same PDFs should hit the cache for every LLM call."""
    generate_invoice_pdfs(3, pdf_project / "data" / "invoices_pdf", seed=1)

    api_calls = {"n": 0}

    def fake_api(
        content: str, model: str, system: str, fields_spec: list, **kwargs: object
    ) -> dict:
        api_calls["n"] += 1
        return {
            "invoice_id": "INV-X",
            "vendor": "Mocked",
            "issue_date": "2026-01-01",
            "currency": "USD",
            "total": 1.0,
        }

    monkeypatch.setattr(llm_backend, "_default_call_api", fake_api)

    run_project(pdf_project)
    assert api_calls["n"] == 3
    run_project(pdf_project)
    assert api_calls["n"] == 3, "second run should not invoke the API"


def test_pdf_transform_propagates_custom_api_key_env(
    monkeypatch: pytest.MonkeyPatch, pdf_project: Path
) -> None:
    generate_invoice_pdfs(1, pdf_project / "data" / "invoices_pdf", seed=1)
    profiles = pdf_project / "profiles.yml"
    profiles.write_text(
        profiles.read_text().replace(
            "api_key_env: ANTHROPIC_API_KEY",
            "api_key_env: DBT_ML_ANTHROPIC_KEY",
        )
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("DBT_ML_ANTHROPIC_KEY", "custom-test-key")
    seen_references: list[CredentialReference] = []

    def fake_api(
        content: str, model: str, system: str, fields_spec: list, **kwargs: object
    ) -> dict:
        reference = kwargs["api_key_env"]
        assert isinstance(reference, CredentialReference)
        seen_references.append(reference)
        return {
            "invoice_id": "INV-CUSTOM-KEY",
            "vendor": "Mocked Vendor",
            "issue_date": "2026-01-01",
            "currency": "USD",
            "total": 1.0,
        }

    monkeypatch.setattr(llm_backend, "_default_call_api", fake_api)

    results = run_project(pdf_project)

    assert len(seen_references) == 1
    assert seen_references[0].is_available()
    assert "DBT_ML_ANTHROPIC_KEY" not in repr(seen_references[0])
    assert not next(r for r in results if r.model_name == "extracted_invoices").errors
