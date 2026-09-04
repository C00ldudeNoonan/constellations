"""Google Drive document source: `path: gdrive://<folderId>` (issue #514).

The one first-party SaaS source ADR-0006 allows, because Drive is
file-grained and fits the object-source shape. Discovery walks the folder
tree through the Drive REST API and never downloads a byte; `fetch()`
materializes exactly one document the runner decided to process.

Identity, in two kinds:

- Uploaded binaries (PDF, DOCX, images) carry an md5 in the listing, so their
  `content_hash` is `md5:<checksum>` exactly as for `gs://`, and the
  downloaded bytes are verified against it.
- Native Docs and Slides have no content hash in any listing. Their
  `content_hash` is `mtime:<modifiedTime>` — a **change token**, named as
  such: every content edit moves it (it never under-triggers on content) and
  a no-op save also moves it (it can over-trigger). Drive's monotonic
  `version` rides in `source_metadata`.

Native files are fetched in the format they are rendered to, and their
`relative_path` carries that extension: a Doc exports as markdown
(`<name>.md`), a Slides deck renders to markdown one heading per slide
(`<name>.md`). `file_pattern: "*.md"` therefore selects both; `"*.pdf"`
selects uploaded PDFs. Sheets, Forms, Drawings and other native types are
skipped and counted.

Auth is Application Default Credentials with the Drive read-only scope —
`gcloud auth application-default login --scopes=...` locally, or
GOOGLE_APPLICATION_CREDENTIALS pointing at a service-account JSON in CI —
the same posture as `gs://`. `google-auth` and `requests` ship as the
`gdrive` extra; the four REST endpoints need no discovery client.
"""
from __future__ import annotations

import fnmatch
import hashlib
import importlib
import logging
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from ..config.loader import ConfigError
from ..config.source import SourceConfig
from ..versioning import compute_document_id
from .base import DocumentRef, DocumentSource, SourceError, SourceScan
from .gcs import _static_filter_prefixes

log = logging.getLogger(__name__)

GDRIVE_SCHEME = "gdrive://"
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
_DRIVE_API = "https://www.googleapis.com/drive/v3"
_SLIDES_API = "https://slides.googleapis.com/v1"

FOLDER_MIME = "application/vnd.google-apps.folder"
SHORTCUT_MIME = "application/vnd.google-apps.shortcut"
DOC_MIME = "application/vnd.google-apps.document"
SLIDES_MIME = "application/vnd.google-apps.presentation"
_NATIVE_PREFIX = "application/vnd.google-apps."
# Native types and the extension of the format they are fetched as.
NATIVE_EXTENSIONS: dict[str, str] = {DOC_MIME: ".md", SLIDES_MIME: ".md"}

_INSTALL_HINT = (
    "Google Drive sources require google-auth and requests. "
    "Install them with: pip install 'stel[gdrive]'"
)
_LOGIN_HINT = (
    "Run `gcloud auth application-default login "
    f"--scopes={DRIVE_SCOPE},https://www.googleapis.com/auth/cloud-platform` "
    "locally, or set GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON "
    "file that the folder is shared with."
)
_LIST_FIELDS = (
    "nextPageToken,files(id,name,mimeType,modifiedTime,md5Checksum,size,version)"
)
_GET_FIELDS = "id,name,mimeType,modifiedTime,md5Checksum,size,version"
_PAGE_SIZE = 1000
_SCAN_CEILING_MULTIPLIER = 10
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 4


def parse_gdrive_path(path: str) -> str:
    """`gdrive://<folderId>` → folder id. A shared drive's id is a folder id."""
    rest = path.removeprefix(GDRIVE_SCHEME)
    if rest == path or not rest or "/" in rest or rest != rest.strip():
        raise SourceError(
            f"Invalid Google Drive source path {path!r}: expected gdrive://<folderId>"
        )
    return rest


class DriveApiError(SourceError):
    """A Drive or Slides request failed. Carries the HTTP status and the
    operation only — never a response body, which may quote the document."""

    def __init__(self, operation: str, status: int | None) -> None:
        label = f"HTTP {status}" if status is not None else "transport error"
        super().__init__(f"Google Drive {operation} failed [{label}]")
        self.status = status


class DriveApi(Protocol):
    """The four calls the source makes. The REST client implements it against
    Google; tests implement it in memory."""

    def list_children(self, folder_id: str, page_token: str | None) -> Mapping[str, Any]: ...

    def get_file(self, file_id: str) -> Mapping[str, Any]: ...

    def export(self, file_id: str, mime_type: str) -> bytes: ...

    def download(self, file_id: str) -> bytes: ...

    def get_presentation(self, presentation_id: str) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


class RestDriveApi:
    """Drive v3 and Slides v1 over an authorized requests session."""

    def __init__(self, session: Any, *, sleep: Callable[[float], None] = time.sleep) -> None:
        self._session = session
        self._sleep = sleep

    def close(self) -> None:
        self._session.close()

    def _request(
        self, operation: str, url: str, params: Mapping[str, Any]
    ) -> Any:
        """One GET with bounded backoff on the statuses Drive documents as
        retryable. The body of a failure is never read into an error."""
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = self._session.get(url, params=dict(params), timeout=120)
            except Exception as e:
                raise DriveApiError(operation, None) from e
            status = int(response.status_code)
            if status < 400:
                return response
            if status not in _RETRY_STATUSES or attempt == _MAX_ATTEMPTS:
                raise DriveApiError(operation, status)
            self._sleep(0.5 * 2 ** (attempt - 1))
        raise AssertionError("unreachable: the retry loop returns or raises")

    def list_children(self, folder_id: str, page_token: str | None) -> Mapping[str, Any]:
        params: dict[str, Any] = {
            "q": f"'{folder_id}' in parents and trashed = false",
            "fields": _LIST_FIELDS,
            "pageSize": _PAGE_SIZE,
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
            "orderBy": "name",
        }
        if page_token:
            params["pageToken"] = page_token
        return self._request("listing", f"{_DRIVE_API}/files", params).json()

    def get_file(self, file_id: str) -> Mapping[str, Any]:
        return self._request(
            "metadata read",
            f"{_DRIVE_API}/files/{file_id}",
            {"fields": _GET_FIELDS, "supportsAllDrives": "true"},
        ).json()

    def export(self, file_id: str, mime_type: str) -> bytes:
        return self._request(
            "export", f"{_DRIVE_API}/files/{file_id}/export", {"mimeType": mime_type}
        ).content

    def download(self, file_id: str) -> bytes:
        return self._request(
            "download",
            f"{_DRIVE_API}/files/{file_id}",
            {"alt": "media", "supportsAllDrives": "true"},
        ).content

    def get_presentation(self, presentation_id: str) -> Mapping[str, Any]:
        return self._request(
            "presentation read", f"{_SLIDES_API}/presentations/{presentation_id}", {}
        ).json()


def _authorized_session(quota_project: str | None) -> Any:
    try:
        google_auth = importlib.import_module("google.auth")
        transport = importlib.import_module("google.auth.transport.requests")
    except ImportError as e:
        raise SourceError(_INSTALL_HINT) from e
    from google.auth.exceptions import DefaultCredentialsError

    try:
        credentials, _ = google_auth.default(
            scopes=[DRIVE_SCOPE], quota_project_id=quota_project
        )
    except DefaultCredentialsError as e:
        raise ConfigError(
            "Google Drive Application Default Credentials were not found or are "
            f"invalid. {_LOGIN_HINT}"
        ) from e
    return transport.AuthorizedSession(credentials)


# --- Listing ------------------------------------------------------------------


@dataclass(frozen=True)
class _Listed:
    file_id: str
    relative_path: str
    mime_type: str
    modified_time: str | None
    md5: str | None
    size: int | None
    version: str | None


def _extension_for(mime_type: str) -> str | None:
    """The relative-path suffix to append. None means the file is not a
    document this source serves (an unsupported native type)."""
    if mime_type in NATIVE_EXTENSIONS:
        return NATIVE_EXTENSIONS[mime_type]
    if mime_type.startswith(_NATIVE_PREFIX):
        return None
    return ""


def _matches(relative: str, pattern: str, recursive: bool) -> bool:
    if not recursive and "/" in relative:
        return False
    target = relative if "/" in pattern else PurePosixPath(relative).name
    return fnmatch.fnmatchcase(target, pattern)


def _folder_may_match(folder_path: str, static_prefixes: Sequence[str]) -> bool:
    """Whether any `--source-filter` glob can select something under a folder,
    so a walk can skip subtrees the filters exclude. `("",)` means no
    narrowing: every folder may match."""
    prefix = f"{folder_path}/"
    return any(
        not static or static.startswith(prefix) or prefix.startswith(static)
        for static in static_prefixes
    )


def content_hash_for(entry: Mapping[str, Any]) -> str:
    """`md5:` for a listed checksum; otherwise the change token, prefixed so
    the two kinds can never collide."""
    md5 = entry.get("md5Checksum")
    if md5:
        return f"md5:{md5}"
    modified = entry.get("modifiedTime")
    if modified:
        return f"mtime:{modified}"
    return f"ver:{entry.get('version')}"


class GoogleDriveDocumentSource(DocumentSource):
    def __init__(self, api: DriveApi | None = None) -> None:
        self._api = api

    def _get_api(self, quota_project: str | None) -> DriveApi:
        if self._api is None:
            self._api = RestDriveApi(_authorized_session(quota_project))
        return self._api

    def close(self) -> None:
        api = self._api
        self._api = None
        if api is not None:
            try:
                api.close()
            except Exception:  # pragma: no cover - best-effort teardown
                pass

    def _walk(
        self,
        source: SourceConfig,
        root_id: str,
        *,
        static_prefixes: Sequence[str],
        scan_ceiling: int,
    ) -> Iterator[tuple[str, Mapping[str, Any]]]:
        """Yield `(folder path, entry)` for every non-folder entry under the
        root, breadth-first, under one scan ceiling across every folder and
        page. Folders the filters exclude are not descended."""
        api = self._get_api(source.project)
        pending: deque[tuple[str, str]] = deque([(root_id, "")])
        scanned = 0
        while pending:
            folder_id, folder_path = pending.popleft()
            page_token: str | None = None
            while True:
                page = api.list_children(folder_id, page_token)
                for entry in page.get("files", ()):
                    scanned += 1
                    if scanned > scan_ceiling:
                        raise SourceError(
                            f"Source '{source.name}': scanned more than {scan_ceiling} "
                            f"entries under {source.path} without reaching the end "
                            "of the folder tree. The folder is too broad — narrow "
                            "it, or raise `max_objects` on the source."
                        )
                    mime = str(entry.get("mimeType", ""))
                    name = str(entry.get("name", ""))
                    if mime == FOLDER_MIME:
                        child_path = f"{folder_path}/{name}" if folder_path else name
                        if source.recursive and _folder_may_match(child_path, static_prefixes):
                            pending.append((str(entry["id"]), child_path))
                        continue
                    yield folder_path, entry
                page_token = page.get("nextPageToken")
                if not page_token:
                    break

    def discover(
        self,
        source: SourceConfig,
        project_dir: Path,
        *,
        source_filter: Sequence[str] = (),
    ) -> list[DocumentRef]:
        root_id = parse_gdrive_path(source.path)
        static_prefixes = _static_filter_prefixes(source_filter)
        scan_ceiling = source.max_objects * _SCAN_CEILING_MULTIPLIER
        log.info("Source '%s': walking Google Drive folder %s", source.name, root_id)
        listed: list[_Listed] = []
        skipped_native = 0
        seen_paths: dict[str, str] = {}
        try:
            for folder_path, entry in self._walk(
                source, root_id, static_prefixes=static_prefixes, scan_ceiling=scan_ceiling
            ):
                mime = str(entry.get("mimeType", ""))
                name = str(entry.get("name", ""))
                if mime == SHORTCUT_MIME:
                    continue
                extension = _extension_for(mime)
                if extension is None:
                    skipped_native += 1
                    continue
                stem = f"{folder_path}/{name}" if folder_path else name
                relative = f"{stem}{extension}"
                if not _matches(relative, source.file_pattern, source.recursive):
                    continue
                file_id = str(entry["id"])
                if relative in seen_paths:
                    raise SourceError(
                        f"Source '{source.name}': two files share the path "
                        f"'{relative}' (ids {seen_paths[relative]} and {file_id}). "
                        "Document identity is the folder path plus the name, so "
                        "rename one of them."
                    )
                seen_paths[relative] = file_id
                listed.append(
                    _Listed(
                        file_id=file_id,
                        relative_path=relative,
                        mime_type=mime,
                        modified_time=entry.get("modifiedTime"),
                        md5=entry.get("md5Checksum"),
                        size=int(entry["size"]) if entry.get("size") is not None else None,
                        version=str(entry["version"]) if entry.get("version") is not None else None,
                    )
                )
                if len(listed) > source.max_objects:
                    raise SourceError(
                        f"Source '{source.name}': more than {source.max_objects} "
                        f"documents match under {source.path}. Narrow the folder or "
                        "`file_pattern`, or raise `max_objects` on the source."
                    )
        except (SourceError, ConfigError):
            raise
        except Exception as e:
            raise DriveApiError(f"listing of folder {root_id}", None) from e
        log.info(
            "Source '%s': %d document(s) matched under folder %s "
            "(%d unsupported native file(s) skipped)",
            source.name,
            len(listed),
            root_id,
            skipped_native,
        )
        refs: list[DocumentRef] = []
        for item in sorted(listed, key=lambda item: item.relative_path):
            entry = {
                "md5Checksum": item.md5,
                "modifiedTime": item.modified_time,
                "version": item.version,
            }
            metadata: dict[str, Any] = {
                "file_id": item.file_id,
                "mime_type": item.mime_type,
                "modified_time": item.modified_time,
                "md5_checksum": item.md5,
                "size": item.size,
                "version": item.version,
                "identity": "md5" if item.md5 else "change_token",
                "project": source.project,
            }
            refs.append(
                DocumentRef(
                    source_name=source.name,
                    relative_path=item.relative_path,
                    document_id=compute_document_id(source.name, item.relative_path),
                    content_hash=content_hash_for(entry),
                    source_uri=f"{GDRIVE_SCHEME}{item.file_id}#v{item.version}",
                    source_metadata={k: v for k, v in metadata.items() if v is not None},
                )
            )
        return refs

    def _verify_unchanged(self, api: DriveApi, ref: DocumentRef, meta: Mapping[str, Any]) -> None:
        """The generation-pin analogue: Drive cannot serve an old version by
        number here, so the fetch re-reads the listing fields and refuses to
        proceed when they moved. A run extracts what it discovered or fails."""
        current = api.get_file(str(meta["file_id"]))
        if content_hash_for(current) != ref.content_hash:
            raise SourceError(
                f"Google Drive file '{ref.relative_path}' changed after discovery; "
                "run again to refresh source state"
            )

    def _bytes_for(self, api: DriveApi, ref: DocumentRef, meta: Mapping[str, Any]) -> bytes:
        file_id = str(meta["file_id"])
        mime = str(meta.get("mime_type", ""))
        if mime == DOC_MIME:
            return api.export(file_id, "text/markdown")
        if mime == SLIDES_MIME:
            return render_presentation(api.get_presentation(file_id)).encode("utf-8")
        payload = api.download(file_id)
        listed_md5 = meta.get("md5_checksum")
        if listed_md5 and hashlib.md5(payload).hexdigest() != str(listed_md5):
            raise SourceError(
                f"Google Drive file '{ref.relative_path}' downloaded bytes do not "
                "match the listed checksum; run again to refresh source state"
            )
        return payload

    def fetch(self, ref: DocumentRef, work_dir: Path) -> Path:
        assert ref.source_metadata is not None
        meta = ref.source_metadata
        api = self._get_api(meta.get("project"))
        suffix = PurePosixPath(ref.relative_path).suffix
        local = work_dir / f"{ref.document_id}{suffix}"
        partial = work_dir / f".{ref.document_id}{suffix}.partial"
        try:
            self._verify_unchanged(api, ref, meta)
            partial.write_bytes(self._bytes_for(api, ref, meta))
        except (SourceError, ConfigError):
            partial.unlink(missing_ok=True)
            raise
        except Exception as e:
            partial.unlink(missing_ok=True)
            raise DriveApiError(f"fetch of {ref.source_uri or ref.relative_path}", None) from e
        partial.replace(local)
        return local

    def scan(self, source: SourceConfig, project_dir: Path) -> SourceScan:
        refs = self.discover(source, project_dir)
        if not refs:
            return SourceScan(
                exists=True,
                file_count=0,
                newest_epoch=None,
                newest_name=None,
                message="no matching files",
            )
        newest_epoch: float | None = None
        newest_name: str | None = None
        for ref in refs:
            modified = (ref.source_metadata or {}).get("modified_time")
            if not modified:
                continue
            epoch = datetime.fromisoformat(str(modified).replace("Z", "+00:00")).timestamp()
            if newest_epoch is None or epoch > newest_epoch:
                newest_epoch = epoch
                newest_name = ref.relative_path
        return SourceScan(
            exists=True, file_count=len(refs), newest_epoch=newest_epoch, newest_name=newest_name
        )


# --- Slides rendering ---------------------------------------------------------

_TITLE_PLACEHOLDERS = frozenset({"TITLE", "CENTERED_TITLE"})
_SUBTITLE_PLACEHOLDERS = frozenset({"SUBTITLE"})


def _text_paragraphs(text: Mapping[str, Any] | None) -> list[tuple[int | None, str]]:
    """A shape's text as `(bullet nesting level or None, paragraph text)`.
    Slides delivers a paragraph marker followed by its runs; the run text
    carries the trailing newline that ends the paragraph."""
    if not text:
        return []
    paragraphs: list[tuple[int | None, str]] = []
    level: int | None = None
    buffer: list[str] = []

    def flush() -> None:
        joined = "".join(buffer).strip()
        if joined:
            paragraphs.append((level, joined))
        buffer.clear()

    for element in text.get("textElements", ()):
        marker = element.get("paragraphMarker")
        if marker is not None:
            flush()
            bullet = marker.get("bullet")
            level = int(bullet.get("nestingLevel", 0)) if bullet is not None else None
            continue
        run = element.get("textRun")
        if run is not None:
            buffer.append(str(run.get("content", "")))
    flush()
    return paragraphs


def _position(element: Mapping[str, Any]) -> tuple[float, float]:
    transform = element.get("transform") or {}
    return (float(transform.get("translateY", 0.0)), float(transform.get("translateX", 0.0)))


def _shapes(elements: Sequence[Mapping[str, Any]]) -> Iterator[Mapping[str, Any]]:
    """Leaf page elements in reading order: groups are flattened, and
    elements sort top-to-bottom then left-to-right. The API's own order is
    z-order, which is not how a slide reads."""
    leaves: list[Mapping[str, Any]] = []
    for element in elements:
        group = element.get("elementGroup")
        if group is not None:
            leaves.extend(_shapes(group.get("children", ())))
        else:
            leaves.append(element)
    return iter(sorted(leaves, key=_position))


def _placeholder_type(element: Mapping[str, Any]) -> str | None:
    shape = element.get("shape") or {}
    placeholder = shape.get("placeholder")
    return str(placeholder.get("type")) if placeholder else None


def _render_paragraphs(paragraphs: Sequence[tuple[int | None, str]]) -> list[str]:
    lines: list[str] = []
    for level, text in paragraphs:
        if level is None:
            if lines and lines[-1] != "":
                lines.append("")
            lines.append(text)
            lines.append("")
        else:
            lines.append(f"{'  ' * level}- {text}")
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def _render_table(table: Mapping[str, Any]) -> list[str]:
    rows: list[list[str]] = []
    for row in table.get("tableRows", ()):
        cells = [
            " ".join(text for _, text in _text_paragraphs(cell.get("text")))
            for cell in row.get("tableCells", ())
        ]
        rows.append(cells)
    if not rows:
        return []
    width = max(len(row) for row in rows)
    lines = ["| " + " | ".join(row + [""] * (width - len(row))) + " |" for row in rows]
    lines.insert(1, "|" + " --- |" * width)
    return lines


def _speaker_notes(slide: Mapping[str, Any]) -> list[str]:
    properties = slide.get("slideProperties") or {}
    notes_page = properties.get("notesPage") or {}
    notes_id = (notes_page.get("notesProperties") or {}).get("speakerNotesObjectId")
    for element in notes_page.get("pageElements", ()):
        if notes_id is not None and element.get("objectId") != notes_id:
            continue
        shape = element.get("shape") or {}
        paragraphs = _text_paragraphs(shape.get("text"))
        if paragraphs:
            return [text for _, text in paragraphs]
    return []


def _render_slide(index: int, slide: Mapping[str, Any]) -> list[str]:
    title: str | None = None
    body: list[str] = []
    for element in _shapes(slide.get("pageElements", ())):
        table = element.get("table")
        if table is not None:
            body.extend(["", *_render_table(table), ""])
            continue
        shape = element.get("shape")
        if shape is None:
            continue
        paragraphs = _text_paragraphs(shape.get("text"))
        if not paragraphs:
            continue
        kind = _placeholder_type(element)
        if kind in _TITLE_PLACEHOLDERS and title is None:
            title = " ".join(text for _, text in paragraphs)
            continue
        if kind in _SUBTITLE_PLACEHOLDERS:
            body.extend(["", *(f"*{text}*" for _, text in paragraphs), ""])
            continue
        body.extend(["", *_render_paragraphs(paragraphs)])
    heading = f"## {index}. {title}" if title else f"## Slide {index}"
    lines = [heading, *body]
    notes = _speaker_notes(slide)
    if notes:
        lines.extend(["", "> Notes:", *(f"> {line}" for line in notes)])
    collapsed: list[str] = []
    for line in lines:
        if line == "" and collapsed and collapsed[-1] == "":
            continue
        collapsed.append(line)
    while collapsed and collapsed[-1] == "":
        collapsed.pop()
    return collapsed


def render_presentation(presentation: Mapping[str, Any]) -> str:
    """Markdown for a Slides deck: `#` deck title, then `##` per slide with
    its title, so heading attribution gives every chunk its slide."""
    title = str(presentation.get("title") or "").strip()
    parts: list[str] = [f"# {title}"] if title else []
    for index, slide in enumerate(presentation.get("slides", ()), start=1):
        parts.append("\n".join(_render_slide(index, slide)))
    return "\n\n".join(parts).rstrip() + ("\n" if parts else "")

