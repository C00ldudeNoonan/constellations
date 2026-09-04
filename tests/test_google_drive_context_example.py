"""End-to-end proof for examples/google_drive_context (issue #514): a Drive
folder of Docs and Slides becomes governed, heading-attributed context in
DuckDB and answers a question through the stel MCP service — against the
in-memory Drive, so the default suite needs no credentials."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import duckdb
import pytest

from stel.checks import run_project_tests
from stel.mcp_server.contracts import SearchContextRequest
from stel.mcp_server.service import ContextService
from stel.runner import run_project
from stel.sources import GoogleDriveDocumentSource
from tests.test_gdrive_source import ROOT, FakeDrive

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "google_drive_context"
_T = "2026-08-20T09:00:00.000Z"


def _project(tmp_path: Path) -> Path:
    dst = tmp_path / "google_drive_context"
    shutil.copytree(_EXAMPLE, dst, ignore=shutil.ignore_patterns("target", "__pycache__"))
    return dst


def _slide(title: str, bullets: list[str]) -> dict[str, Any]:
    def shape(object_id: str, lines: list[str], placeholder: str, y: float) -> dict[str, Any]:
        elements: list[dict[str, Any]] = []
        for line in lines:
            marker: dict[str, Any] = {"bullet": {"nestingLevel": 0}} if placeholder == "BODY" else {}
            elements.append({"paragraphMarker": marker})
            elements.append({"textRun": {"content": line + "\n"}})
        return {
            "objectId": object_id,
            "shape": {"text": {"textElements": elements}, "placeholder": {"type": placeholder}},
            "transform": {"translateY": y, "translateX": 0},
        }

    return {"pageElements": [shape("t", [title], "TITLE", 0), shape("b", bullets, "BODY", 100)]}


def _drive() -> FakeDrive:
    drive = FakeDrive()
    drive.add_doc(
        ROOT,
        "doc-laptop",
        "Laptop setup",
        "# Laptop setup\n\n## Enroll the device\n\nEnter the enrollment code from IT.\n\n"
        "## Encrypt the disk\n\nDisk encryption is mandatory on every laptop. Store the "
        "passphrase in the password manager and never in a note.\n",
        modified=_T,
    )
    policies = drive.add_folder(ROOT, "folder-policies", "Policies")
    drive.add_doc(
        policies,
        "doc-expenses",
        "Expense policy",
        "# Expense policy\n\n## Limits\n\nMeals up to 60 per day while traveling.\n\n"
        "## Receipts\n\nItemized receipts are required over 25.\n",
        modified=_T,
    )
    drive.add_slides(
        ROOT,
        "deck-kickoff",
        "Q3 kickoff",
        {
            "title": "Q3 kickoff",
            "slides": [
                _slide("Goals", ["Ship the Drive source", "Query it over MCP"]),
                _slide("Risks", ["Drive API quota on large folders"]),
            ],
        },
        modified=_T,
    )
    return drive


@pytest.fixture
def built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    drive = _drive()
    monkeypatch.setattr(GoogleDriveDocumentSource, "_get_api", lambda self, project=None: drive)
    project = _project(tmp_path)
    run_project(project)
    return project


def _rows(project: Path, sql: str) -> list[tuple[Any, ...]]:
    con = duckdb.connect(str(project / "target" / "context.duckdb"), read_only=True)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def test_example_runs_and_all_checks_pass(built: Path) -> None:
    failed = [r for r in run_project_tests(built) if not r.passed]
    assert failed == [], [(r.test_name, r.message) for r in failed]


def test_docs_and_slides_become_section_attributed_context(built: Path) -> None:
    rows = _rows(
        built,
        "select title, section from drive.document_chunks order by title, section",
    )
    by_title: dict[str, set[str | None]] = {}
    for title, section in rows:
        by_title.setdefault(title, set()).add(section)
    assert by_title["Laptop setup"] >= {"Enroll the device", "Encrypt the disk"}
    assert by_title["Expense policy"] >= {"Limits", "Receipts"}
    # A deck's slides are its sections.
    assert by_title["Q3 kickoff"] >= {"1. Goals", "2. Risks"}


def test_the_folder_is_answerable_through_the_mcp_service(
    built: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STEL_MCP_PRINCIPAL_ID", "local-operator")
    monkeypatch.setenv("STEL_MCP_TENANT_ID", "drive")
    service = ContextService.from_project(built)
    try:
        models = service.list_context_models()
        assert [model.name for model in models.models] == ["context_search"]

        response = service.search_context(
            SearchContextRequest(model="context_search", query="disk encryption", mode="text")
        )
        assert response.error is None, response.error
        assert response.results, "the question about disk encryption found nothing"
        top = response.results[0]
        assert "encryption" in top.snippet.lower()
        assert top.citation.section_path == ("Laptop setup", "Encrypt the disk")
        assert top.citation.source_uri == "gdrive://doc-laptop#v1"

        deck = service.search_context(
            SearchContextRequest(model="context_search", query="API quota", mode="text")
        )
        assert deck.error is None and deck.results
        assert deck.results[0].citation.section_path == ("Q3 kickoff", "2. Risks")
    finally:
        service.close()
