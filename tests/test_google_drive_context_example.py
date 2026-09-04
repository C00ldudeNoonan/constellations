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
from stel.manifest import write_manifest
from stel.mcp_server.contracts import ListContextModelsRequest, SearchContextRequest
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
            bullet = placeholder == "BODY"
            marker: dict[str, Any] = {"bullet": {"nestingLevel": 0}} if bullet else {}
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
    # Sections are long enough that each becomes its own chunk at the
    # example's chunk size, as a real Doc's would.
    enroll = (
        "Power the laptop on, join the guest network, and enter the enrollment code "
        "from IT when the setup assistant asks for it. The code is single-use, so if "
        "enrollment fails at any step ask IT for a fresh one rather than retrying. "
        "Wait for the corporate profile to finish installing before signing in; it "
        "configures the VPN, the password manager, and the update policy."
    )
    encrypt = (
        "Disk encryption is mandatory on every laptop, without exception. On the "
        "corporate Linux image the installer prompts for a passphrase during the "
        "first boot; store that passphrase in the password manager and never in a "
        "note or a chat message. Verify the status afterwards and, if the volume is "
        "not reported as active, stop and contact IT before putting any data on it."
    )
    drive.add_doc(
        ROOT,
        "doc-laptop",
        "Laptop setup",
        f"# Laptop setup\n\n## Enroll the device\n\n{enroll}\n\n## Encrypt the disk\n\n{encrypt}\n",
        modified=_T,
    )
    limits = (
        "Meals are reimbursed up to 60 per day while traveling and lodging must be "
        "booked through the travel tool so the negotiated rates apply. Equipment over "
        "500 needs written approval from your manager before purchase, and software "
        "subscriptions of any size go through procurement rather than a personal card."
    )
    receipts = (
        "Itemized receipts are required for any amount over 25; a card statement line "
        "is not a receipt. Submit claims within 30 days of the expense. Claims older "
        "than that are paid only with a written exception from Finance, which is rare."
    )
    policies = drive.add_folder(ROOT, "folder-policies", "Policies")
    drive.add_doc(
        policies,
        "doc-expenses",
        "Expense policy",
        f"# Expense policy\n\n## Limits\n\n{limits}\n\n## Receipts\n\n{receipts}\n",
        modified=_T,
    )
    drive.add_slides(
        ROOT,
        "deck-kickoff",
        "Q3 kickoff",
        {
            "title": "Q3 kickoff",
            "slides": [
                _slide(
                    "Goals",
                    [
                        "Ship the Google Drive source so a shared folder of Docs and "
                        "Slides becomes governed context without a connector",
                        "Query that context over the stel MCP server from the assistants "
                        "the team already uses, with citations back to the file",
                        "Keep the default test suite offline by driving everything "
                        "through an in-memory Drive double",
                    ],
                ),
                _slide(
                    "Risks",
                    [
                        "Drive API quota on large folders: a recursive walk of a shared "
                        "drive root can burn the per-minute listing budget",
                        "Native files carry a change token rather than a content hash, "
                        "so a no-op save re-extracts a document",
                    ],
                ),
            ],
        },
        modified=_T,
    )
    return drive


@pytest.fixture
def built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    drive = _drive()
    monkeypatch.setattr(GoogleDriveDocumentSource, "_get_api", lambda self, project=None: drive)
    # The folder id reaches the source the way the README says: through the
    # profile's `source_paths` override, from the environment.
    monkeypatch.setenv("STEL_DRIVE_FOLDER", f"gdrive://{ROOT}")
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
    # Every chunk is attributed to a real heading of its document: the title
    # for the opening chunk, then the section whose text it holds.
    assert by_title["Laptop setup"] <= {"Laptop setup", "Enroll the device", "Encrypt the disk"}
    assert by_title["Expense policy"] <= {"Expense policy", "Limits", "Receipts"}
    assert by_title["Q3 kickoff"] <= {"Q3 kickoff", "1. Goals", "2. Risks"}
    # Each document splits into more than one section, and a later section
    # starts its own chunk rather than riding in the previous one's tail.
    assert "Encrypt the disk" in by_title["Laptop setup"]
    assert "Receipts" in by_title["Expense policy"]
    # A deck's slides are its sections.
    assert "2. Risks" in by_title["Q3 kickoff"]


def test_the_folder_is_answerable_through_the_mcp_service(
    built: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STEL_MCP_PRINCIPAL_ID", "local-operator")
    monkeypatch.setenv("STEL_MCP_TENANT_ID", "drive")
    # `stel run` writes the manifest the server reads; run_project alone does not.
    write_manifest(built)
    service = ContextService.from_project(built)
    try:
        models = service.list_context_models(ListContextModelsRequest())
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
