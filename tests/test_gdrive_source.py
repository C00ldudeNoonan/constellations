"""Google Drive document source (issue #514): discovery from listings,
identity without download, a change token named as such for native files,
pin-and-verify fetch, Slides rendering, and the incremental loop end to end
against an in-memory Drive. Real Drive runs only when STEL_GDRIVE_TEST_FOLDER
is set."""
from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import duckdb
import pytest

from stel.config import ConfigError
from stel.config.source import SourceConfig
from stel.runner import run_project
from stel.sources import (
    GoogleDriveDocumentSource,
    SourceError,
    get_document_source,
)
from stel.sources.gdrive import (
    DOC_MIME,
    FOLDER_MIME,
    SHORTCUT_MIME,
    SLIDES_MIME,
    DriveApiError,
    RestDriveApi,
    content_hash_for,
    parse_gdrive_path,
    render_presentation,
)
from stel.versioning import compute_document_id

# ─── an in-memory Drive ─────────────────────────────────────────────────────


class FakeDrive:
    """Implements the `DriveApi` protocol over dicts. `page_size` splits
    listings so pagination is exercised; entries are mutable so a test can
    move a file between discovery and fetch."""

    def __init__(self, *, page_size: int = 1000) -> None:
        self.children: dict[str, list[dict[str, Any]]] = {}
        self.blobs: dict[str, bytes] = {}
        self.exports: dict[str, bytes] = {}
        self.presentations: dict[str, dict[str, Any]] = {}
        self.page_size = page_size
        self.listed_folders: list[str] = []
        self.closed = False

    def add_folder(self, parent: str, file_id: str, name: str) -> str:
        self.children.setdefault(parent, []).append(
            {"id": file_id, "name": name, "mimeType": FOLDER_MIME}
        )
        self.children.setdefault(file_id, [])
        return file_id

    def add_doc(
        self,
        parent: str,
        file_id: str,
        name: str,
        markdown: str,
        *,
        modified: str,
        version: int = 1,
    ) -> dict[str, Any]:
        entry = {
            "id": file_id,
            "name": name,
            "mimeType": DOC_MIME,
            "modifiedTime": modified,
            "version": str(version),
        }
        self.children.setdefault(parent, []).append(entry)
        self.exports[file_id] = markdown.encode("utf-8")
        return entry

    def add_slides(
        self, parent: str, file_id: str, name: str, deck: dict[str, Any], *, modified: str
    ) -> dict[str, Any]:
        entry = {
            "id": file_id,
            "name": name,
            "mimeType": SLIDES_MIME,
            "modifiedTime": modified,
            "version": "3",
        }
        self.children.setdefault(parent, []).append(entry)
        self.presentations[file_id] = deck
        return entry

    def add_blob(
        self, parent: str, file_id: str, name: str, payload: bytes, *, mime: str, modified: str
    ) -> dict[str, Any]:
        entry = {
            "id": file_id,
            "name": name,
            "mimeType": mime,
            "modifiedTime": modified,
            "md5Checksum": hashlib.md5(payload).hexdigest(),
            "size": str(len(payload)),
            "version": "7",
        }
        self.children.setdefault(parent, []).append(entry)
        self.blobs[file_id] = payload
        return entry

    def add_raw(self, parent: str, entry: dict[str, Any]) -> None:
        self.children.setdefault(parent, []).append(entry)

    def entry(self, file_id: str) -> dict[str, Any]:
        for entries in self.children.values():
            for entry in entries:
                if entry["id"] == file_id:
                    return entry
        raise KeyError(file_id)

    # --- DriveApi ---

    def list_children(self, folder_id: str, page_token: str | None) -> Mapping[str, Any]:
        self.listed_folders.append(folder_id)
        entries = self.children.get(folder_id, [])
        start = int(page_token) if page_token else 0
        page = entries[start : start + self.page_size]
        result: dict[str, Any] = {"files": page}
        if start + self.page_size < len(entries):
            result["nextPageToken"] = str(start + self.page_size)
        return result

    def get_file(self, file_id: str) -> Mapping[str, Any]:
        return dict(self.entry(file_id))

    def export(self, file_id: str, mime_type: str) -> bytes:
        assert mime_type == "text/markdown"
        return self.exports[file_id]

    def download(self, file_id: str) -> bytes:
        return self.blobs[file_id]

    def get_presentation(self, presentation_id: str) -> Mapping[str, Any]:
        return self.presentations[presentation_id]

    def close(self) -> None:
        self.closed = True


ROOT = "root-folder-id"
T1 = "2026-08-01T10:00:00.000Z"
T2 = "2026-08-02T10:00:00.000Z"


def _drive() -> FakeDrive:
    drive = FakeDrive()
    drive.add_doc(ROOT, "doc-onboarding", "Onboarding", "# Onboarding\n\nWelcome.\n", modified=T1)
    runbooks = drive.add_folder(ROOT, "folder-runbooks", "Runbooks")
    drive.add_doc(runbooks, "doc-laptop", "Laptop setup", "# Laptop\n\nEnroll.\n", modified=T2)
    drive.add_blob(
        runbooks, "pdf-policy", "policy.pdf", b"%PDF-1.4 fake", mime="application/pdf", modified=T1
    )
    drive.add_slides(ROOT, "deck-kickoff", "Kickoff", _deck(), modified=T2)
    drive.add_raw(
        ROOT,
        {"id": "sheet-1", "name": "Budget", "mimeType": "application/vnd.google-apps.spreadsheet",
         "modifiedTime": T1, "version": "2"},
    )
    drive.add_raw(ROOT, {"id": "short-1", "name": "Onboarding", "mimeType": SHORTCUT_MIME})
    return drive


def _source(drive: FakeDrive) -> GoogleDriveDocumentSource:
    return GoogleDriveDocumentSource(api=drive)


def _cfg(**overrides: Any) -> SourceConfig:
    values: dict[str, Any] = {"name": "drive", "path": f"gdrive://{ROOT}", "file_pattern": "*.md"}
    values.update(overrides)
    return SourceConfig(**values)


def _run(text: str, **extra: Any) -> dict[str, Any]:
    return {"textRun": {"content": text}, **extra}


def _para(level: int | None = None) -> dict[str, Any]:
    marker: dict[str, Any] = {}
    if level is not None:
        marker["bullet"] = {"nestingLevel": level}
    return {"paragraphMarker": marker}


def _shape(
    object_id: str, paragraphs: list[tuple[int | None, str]], *, placeholder: str | None = None,
    y: float = 0.0, x: float = 0.0,
) -> dict[str, Any]:
    elements: list[dict[str, Any]] = []
    for level, text in paragraphs:
        elements.append(_para(level))
        elements.append(_run(text + "\n"))
    shape: dict[str, Any] = {"text": {"textElements": elements}}
    if placeholder:
        shape["placeholder"] = {"type": placeholder}
    return {
        "objectId": object_id,
        "shape": shape,
        "transform": {"translateY": y, "translateX": x, "scaleX": 1, "scaleY": 1},
    }


def _row(*cells: str) -> dict[str, Any]:
    return {
        "tableCells": [
            {"text": {"textElements": [_para(), _run(cell + "\n")]}} for cell in cells
        ]
    }


def _deck() -> dict[str, Any]:
    return {
        "title": "Q3 Kickoff",
        "slides": [
            {
                "objectId": "s1",
                "pageElements": [
                    # Body listed before the title in z-order; reading order
                    # must come from position, not API order.
                    _shape(
                        "b1",
                        [(0, "Ship the Drive source"), (1, "Docs first"), (0, "Then Slides")],
                        placeholder="BODY",
                        y=200,
                    ),
                    _shape("t1", [(None, "Goals")], placeholder="TITLE", y=10),
                    _shape(
                        "st1", [(None, "What we said we would do")], placeholder="SUBTITLE", y=60
                    ),
                ],
                "slideProperties": {
                    "notesPage": {
                        "notesProperties": {"speakerNotesObjectId": "n1"},
                        "pageElements": [
                            _shape(
                                "n1",
                                [(None, "Keep it to five minutes."), (None, "Then questions.")],
                                placeholder="BODY",
                            ),
                        ],
                    }
                },
            },
            {
                "objectId": "s2",
                "pageElements": [
                    {
                        "objectId": "g1",
                        "elementGroup": {
                            "children": [_shape("gt", [(None, "Risks")], placeholder="TITLE", y=5)]
                        },
                    },
                    {
                        "objectId": "tbl",
                        "transform": {"translateY": 100, "translateX": 0},
                        "table": {
                            "tableRows": [
                                _row("Risk", "Owner"),
                                _row("Quota", "Alex"),
                            ]
                        },
                    },
                ],
            },
            {"objectId": "s3", "pageElements": []},
        ],
    }


# ─── path parsing and routing ────────────────────────────────────────────────


def test_parse_gdrive_path() -> None:
    assert parse_gdrive_path("gdrive://1AbC_def-GHI") == "1AbC_def-GHI"


@pytest.mark.parametrize("bad", ["gdrive://", "gdrive://a/b", "gs://a", "gdrive:// x"])
def test_parse_gdrive_path_invalid(bad: str) -> None:
    with pytest.raises(SourceError, match="expected gdrive://<folderId>"):
        parse_gdrive_path(bad)


def test_scheme_routing() -> None:
    assert isinstance(get_document_source("gdrive://abc"), GoogleDriveDocumentSource)


# ─── discovery and identity ──────────────────────────────────────────────────


def test_discover_walks_folders_synthesizes_extensions_and_sorts(tmp_path: Path) -> None:
    refs = _source(_drive()).discover(_cfg(), tmp_path)
    assert [r.relative_path for r in refs] == [
        "Kickoff.md",
        "Onboarding.md",
        "Runbooks/Laptop setup.md",
    ]
    onboarding = refs[1]
    assert onboarding.document_id == compute_document_id("drive", "Onboarding.md")
    assert onboarding.source_uri == "gdrive://doc-onboarding#v1"
    assert onboarding.source_metadata is not None
    assert onboarding.source_metadata["identity"] == "change_token"
    assert onboarding.source_metadata["mime_type"] == DOC_MIME


def test_native_files_carry_a_change_token_and_binaries_an_md5(tmp_path: Path) -> None:
    drive = _drive()
    docs = _source(drive).discover(_cfg(), tmp_path)
    pdfs = _source(drive).discover(_cfg(file_pattern="*.pdf"), tmp_path)
    assert docs[1].content_hash == f"mtime:{T1}"
    [pdf] = pdfs
    assert pdf.relative_path == "Runbooks/policy.pdf"
    assert pdf.content_hash == "md5:" + hashlib.md5(b"%PDF-1.4 fake").hexdigest()
    assert pdf.source_metadata is not None and pdf.source_metadata["identity"] == "md5"


def test_an_edit_moves_the_change_token_and_nothing_else(tmp_path: Path) -> None:
    drive = _drive()
    before = {r.relative_path: r.content_hash for r in _source(drive).discover(_cfg(), tmp_path)}
    drive.entry("doc-onboarding")["modifiedTime"] = T2
    after = {r.relative_path: r.content_hash for r in _source(drive).discover(_cfg(), tmp_path)}
    changed = [path for path in after if after[path] != before[path]]
    assert changed == ["Onboarding.md"]


def test_content_hash_for_prefers_md5_then_token() -> None:
    assert content_hash_for({"md5Checksum": "abc", "modifiedTime": T1}) == "md5:abc"
    assert content_hash_for({"modifiedTime": T1}) == f"mtime:{T1}"
    assert content_hash_for({"version": "4"}) == "ver:4"


def test_unsupported_native_types_and_shortcuts_are_skipped(tmp_path: Path) -> None:
    refs = _source(_drive()).discover(_cfg(file_pattern="*"), tmp_path)
    names = [r.relative_path for r in refs]
    assert "Budget" not in names and "Budget.md" not in names
    assert names.count("Onboarding.md") == 1


def test_non_recursive_lists_the_top_level_only(tmp_path: Path) -> None:
    drive = _drive()
    refs = _source(drive).discover(_cfg(recursive=False), tmp_path)
    assert [r.relative_path for r in refs] == ["Kickoff.md", "Onboarding.md"]
    assert "folder-runbooks" not in drive.listed_folders


def test_duplicate_names_in_one_folder_are_refused_not_guessed(tmp_path: Path) -> None:
    drive = _drive()
    drive.add_doc(ROOT, "doc-onboarding-2", "Onboarding", "# other\n", modified=T1)
    with pytest.raises(SourceError, match=r"two files share the path 'Onboarding\.md'"):
        _source(drive).discover(_cfg(), tmp_path)


def test_max_objects_counts_matching_documents(tmp_path: Path) -> None:
    drive = _drive()
    with pytest.raises(SourceError, match="more than 2 documents match"):
        _source(drive).discover(_cfg(max_objects=2), tmp_path)
    # The PDF and the spreadsheet do not count against a `*.md` source.
    assert len(_source(drive).discover(_cfg(max_objects=3), tmp_path)) == 3


def test_the_scan_ceiling_spans_pages_and_folders(tmp_path: Path) -> None:
    drive = FakeDrive(page_size=3)
    for i in range(40):
        drive.add_raw(ROOT, {"id": f"s{i}", "name": f"Sheet {i}",
                             "mimeType": "application/vnd.google-apps.spreadsheet"})
    with pytest.raises(SourceError, match="scanned more than 30 entries"):
        _source(drive).discover(_cfg(max_objects=3), tmp_path)


def test_pagination_is_followed_to_the_end(tmp_path: Path) -> None:
    drive = FakeDrive(page_size=2)
    for i in range(5):
        drive.add_doc(ROOT, f"d{i}", f"Doc {i}", "x", modified=T1)
    assert len(_source(drive).discover(_cfg(), tmp_path)) == 5


def test_a_source_filter_prunes_folders_it_cannot_match(tmp_path: Path) -> None:
    drive = _drive()
    drive.add_folder(ROOT, "folder-archive", "Archive")
    drive.add_doc("folder-archive", "doc-old", "Old", "x", modified=T1)
    refs = _source(drive).discover(_cfg(), tmp_path, source_filter=["Runbooks/*"])
    # The listing hint narrows the walk; the caller still applies the filter.
    assert "folder-runbooks" in drive.listed_folders
    assert "folder-archive" not in drive.listed_folders
    assert {r.relative_path for r in refs} >= {"Runbooks/Laptop setup.md"}


# ─── fetch: pin and verify ───────────────────────────────────────────────────


def _ref(drive: FakeDrive, tmp_path: Path, relative: str, pattern: str = "*.md") -> Any:
    refs = _source(drive).discover(_cfg(file_pattern=pattern), tmp_path)
    return next(r for r in refs if r.relative_path == relative)


def test_fetch_exports_a_doc_as_markdown(tmp_path: Path) -> None:
    drive = _drive()
    ref = _ref(drive, tmp_path, "Onboarding.md")
    local = _source(drive).fetch(ref, tmp_path)
    assert local.name == f"{ref.document_id}.md"
    assert local.read_text(encoding="utf-8") == "# Onboarding\n\nWelcome.\n"
    assert not list(tmp_path.glob(".*.partial"))


def test_fetch_renders_a_deck_one_heading_per_slide(tmp_path: Path) -> None:
    drive = _drive()
    ref = _ref(drive, tmp_path, "Kickoff.md")
    text = _source(drive).fetch(ref, tmp_path).read_text(encoding="utf-8")
    assert text.startswith("# Q3 Kickoff\n\n## 1. Goals\n")
    assert "## 2. Risks" in text and "## Slide 3" in text


def test_fetch_downloads_a_binary_and_verifies_its_md5(tmp_path: Path) -> None:
    drive = _drive()
    ref = _ref(drive, tmp_path, "Runbooks/policy.pdf", "*.pdf")
    assert _source(drive).fetch(ref, tmp_path).read_bytes() == b"%PDF-1.4 fake"
    drive.blobs["pdf-policy"] = b"%PDF-1.4 tampered"
    with pytest.raises(SourceError, match="do not match the listed checksum"):
        _source(drive).fetch(ref, tmp_path)
    assert not list(tmp_path.glob(".*.partial"))


def test_fetch_refuses_a_file_that_moved_after_discovery(tmp_path: Path) -> None:
    drive = _drive()
    ref = _ref(drive, tmp_path, "Onboarding.md")
    drive.entry("doc-onboarding")["modifiedTime"] = T2
    with pytest.raises(SourceError, match="changed after discovery"):
        _source(drive).fetch(ref, tmp_path)
    assert not list(tmp_path.glob("*")), "nothing partial is left behind"


def test_scan_reports_the_newest_file(tmp_path: Path) -> None:
    scan = _source(_drive()).scan(_cfg(), tmp_path)
    assert scan.file_count == 3
    assert scan.newest_name in {"Kickoff.md", "Runbooks/Laptop setup.md"}
    assert scan.newest_epoch is not None


def test_close_releases_the_api_and_is_idempotent() -> None:
    drive = _drive()
    source = _source(drive)
    source.close()
    source.close()
    assert drive.closed


# ─── the REST client: retries and body-free errors ──────────────────────────


class _Response:
    def __init__(self, status: int, payload: Any = None, content: bytes = b"") -> None:
        self.status_code = status
        self._payload = payload
        self.content = content

    def json(self) -> Any:
        return self._payload


class _Session:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    def get(self, url: str, params: dict[str, Any], timeout: int) -> _Response:
        self.calls.append((url, params))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        self.closed = True


def test_rest_client_retries_retryable_statuses_with_backoff() -> None:
    session = _Session([_Response(429), _Response(503), _Response(200, {"files": []})])
    naps: list[float] = []
    api = RestDriveApi(session, sleep=naps.append)
    assert api.list_children("f", None) == {"files": []}
    assert naps == [0.5, 1.0]
    url, params = session.calls[0]
    assert url.endswith("/drive/v3/files")
    assert params["q"] == "'f' in parents and trashed = false"
    assert params["includeItemsFromAllDrives"] == "true"


def test_rest_client_surfaces_status_but_never_a_body() -> None:
    session = _Session([_Response(403, {"error": {"message": "secret document title"}})])
    with pytest.raises(DriveApiError, match=r"metadata read failed \[HTTP 403\]") as info:
        RestDriveApi(session, sleep=lambda _: None).get_file("x")
    assert "secret" not in str(info.value)
    assert info.value.status == 403


def test_rest_client_wraps_transport_errors_and_gives_up_after_bounded_retries() -> None:
    session = _Session([OSError("connection reset")])
    with pytest.raises(DriveApiError, match=r"export failed \[transport error\]"):
        RestDriveApi(session, sleep=lambda _: None).export("x", "text/markdown")
    exhausted = _Session([_Response(500)] * 4)
    with pytest.raises(DriveApiError, match=r"download failed \[HTTP 500\]"):
        RestDriveApi(exhausted, sleep=lambda _: None).download("x")
    assert len(exhausted.calls) == 4


def test_missing_adc_is_a_configuration_error_with_the_login_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError

    def no_credentials(**kwargs: Any) -> Any:
        raise DefaultCredentialsError("none")

    monkeypatch.setattr(google.auth, "default", no_credentials)
    with pytest.raises(ConfigError, match="gcloud auth application-default login --scopes="):
        GoogleDriveDocumentSource()._get_api(None)


# ─── Slides rendering ────────────────────────────────────────────────────────


def test_render_presentation_reads_in_position_order_with_notes_and_tables() -> None:
    text = render_presentation(_deck())
    assert text == (
        "# Q3 Kickoff\n\n"
        "## 1. Goals\n\n"
        "*What we said we would do*\n\n"
        "- Ship the Drive source\n"
        "  - Docs first\n"
        "- Then Slides\n\n"
        "> Notes:\n"
        "> Keep it to five minutes.\n"
        "> Then questions.\n\n"
        "## 2. Risks\n\n"
        "| Risk | Owner |\n"
        "| --- | --- |\n"
        "| Quota | Alex |\n\n"
        "## Slide 3\n"
    )


def test_render_presentation_handles_an_empty_deck() -> None:
    assert render_presentation({}) == ""
    assert render_presentation({"title": "Empty", "slides": []}) == "# Empty\n"


def test_render_presentation_mixes_paragraphs_and_bullets() -> None:
    deck = {
        "title": "T",
        "slides": [
            {"pageElements": [_shape("b", [(None, "Intro line"), (0, "one"), (None, "Outro")])]}
        ],
    }
    assert render_presentation(deck) == "# T\n\n## Slide 1\n\nIntro line\n\n- one\n\nOutro\n"


# ─── end to end: the incremental loop against the fake Drive ────────────────


@pytest.fixture
def drive_project(tmp_path: Path) -> Path:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "stel_project.yml").write_text(
        "name: drive_demo\nversion: '0.1.0'\nprofile: drive_demo\n", encoding="utf-8"
    )
    (project / "profiles.yml").write_text(
        "drive_demo:\n  target: dev\n  outputs:\n    dev:\n      warehouse:\n"
        "        type: duckdb\n        path: ./target/db.duckdb\n        schema: drive\n",
        encoding="utf-8",
    )
    (project / "sources").mkdir()
    (project / "sources" / "docs.yml").write_text(
        f"version: 2\nsources:\n  - name: drive_docs\n    path: gdrive://{ROOT}\n"
        "    file_pattern: '*.md'\n",
        encoding="utf-8",
    )
    (project / "models").mkdir()
    (project / "models" / "drive_pages.yml").write_text(
        "version: 2\nmodels:\n  - name: drive_pages\n    source: ref('drive_docs')\n"
        "    extraction:\n      backend: markdown\n      options:\n"
        "        include_body: true\n"
        "    materialization: incremental\n",
        encoding="utf-8",
    )
    return project


def _pages(project: Path) -> dict[str, str]:
    con = duckdb.connect(str(project / "target" / "db.duckdb"), read_only=True)
    try:
        return dict(con.execute('SELECT source_path, body FROM "db".drive.drive_pages').fetchall())
    finally:
        con.close()


def test_incremental_run_against_the_fake_drive(
    drive_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    drive = _drive()
    monkeypatch.setattr(GoogleDriveDocumentSource, "_get_api", lambda self, project=None: drive)

    [first] = run_project(drive_project)
    assert first.documents_processed == 3 and first.rows_written == 3
    pages = _pages(drive_project)
    assert pages["Onboarding.md"] == "# Onboarding\n\nWelcome.\n"
    assert pages["Kickoff.md"].startswith("# Q3 Kickoff")

    [second] = run_project(drive_project)
    assert second.documents_processed == 0 and second.documents_skipped == 3

    # A content edit moves the change token; only that document re-extracts.
    drive.entry("doc-laptop")["modifiedTime"] = "2026-08-03T10:00:00.000Z"
    drive.exports["doc-laptop"] = b"# Laptop\n\nEnroll, then encrypt.\n"
    [third] = run_project(drive_project)
    assert third.documents_processed == 1 and third.documents_skipped == 2
    assert _pages(drive_project)["Runbooks/Laptop setup.md"].endswith("Enroll, then encrypt.\n")

    # A deleted file prunes its rows.
    drive.children["folder-runbooks"] = [
        e for e in drive.children["folder-runbooks"] if e["id"] != "doc-laptop"
    ]
    [fourth] = run_project(drive_project)
    assert fourth.documents_deleted == 1
    assert "Runbooks/Laptop setup.md" not in _pages(drive_project)


# ─── optional integration (needs real Drive credentials) ────────────────────

_FOLDER = os.environ.get("STEL_GDRIVE_TEST_FOLDER")


@pytest.mark.skipif(
    not _FOLDER, reason="set STEL_GDRIVE_TEST_FOLDER to a shared folder id to run against Drive"
)
def test_integration_discover_and_fetch(tmp_path: Path) -> None:
    source = GoogleDriveDocumentSource()
    try:
        refs = source.discover(
            SourceConfig(name="it", path=f"gdrive://{_FOLDER}", file_pattern="*.md"), tmp_path
        )
        assert refs, "the test folder needs at least one Doc or Slides deck"
        local = source.fetch(refs[0], tmp_path)
        assert local.read_text(encoding="utf-8").strip()
    finally:
        source.close()
