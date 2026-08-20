"""GCS document sources (issue #84): discovery from listings, identity
without download, bounded scanning, generation-pinned fetch, and the full
incremental loop against a fake storage client. Real GCS runs only when
STEL_GCS_TEST_BUCKET is set."""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import duckdb
import pytest
from click.testing import CliRunner
from google.auth.exceptions import DefaultCredentialsError

from stel.cli import cli
from stel.config import ConfigError
from stel.config.source import SourceConfig
from stel.freshness import check_freshness
from stel.runner import run_project
from stel.sources import (
    GCSDocumentSource,
    LocalDocumentSource,
    SourceError,
    get_document_source,
)
from stel.sources.gcs import content_hash_for_blob, parse_gcs_path
from stel.versioning import compute_document_id

# ─── fakes ──────────────────────────────────────────────────────────────────


class _FakeBlob:
    def __init__(
        self,
        name: str,
        *,
        generation: int = 1,
        md5_hash: str | None = None,
        crc32c: str | None = None,
        size: int = 10,
        updated: datetime | None = None,
        content_type: str = "application/octet-stream",
        etag: str | None = None,
        payload: bytes = b"",
    ) -> None:
        self.name = name
        self.generation = generation
        self.md5_hash = md5_hash
        self.crc32c = crc32c
        self.size = size
        self.updated = updated or datetime(2026, 1, 1, tzinfo=UTC)
        self.content_type = content_type
        self.etag = etag
        self.payload = payload

    def download_to_filename(self, filename: str) -> None:
        Path(filename).write_bytes(self.payload)


class _FakeBucket:
    def __init__(self, blobs: list[_FakeBlob]) -> None:
        self._by_name = {b.name: b for b in blobs}
        self.requested_generations: list[int | None] = []

    def blob(self, name: str, generation: int | None = None) -> _FakeBlob:
        self.requested_generations.append(generation)
        return self._by_name[name]


class _FakeStorageClient:
    def __init__(self, blobs: list[_FakeBlob]) -> None:
        self.blobs = blobs
        self.list_calls: list[tuple[str, str | None, int | None]] = []
        self.last_bucket: _FakeBucket | None = None

    def list_blobs(
        self, bucket_name: str, prefix: str | None = None, max_results: int | None = None
    ) -> Any:
        self.list_calls.append((bucket_name, prefix, max_results))
        matching = [
            b for b in self.blobs if prefix is None or b.name.startswith(prefix)
        ]
        return iter(matching[:max_results])

    def bucket(self, name: str) -> _FakeBucket:
        self.last_bucket = _FakeBucket(self.blobs)
        return self.last_bucket


def _gcs_source(client: _FakeStorageClient) -> GCSDocumentSource:
    src = GCSDocumentSource()
    src._client = client
    return src


def _cfg(**kwargs: Any) -> SourceConfig:
    defaults: dict[str, Any] = {
        "name": "filings",
        "path": "gs://bkt/raw/docs",
        "file_pattern": "*.html",
    }
    return SourceConfig(**{**defaults, **kwargs})


# ─── path parsing + routing ─────────────────────────────────────────────────


def test_parse_gcs_path() -> None:
    assert parse_gcs_path("gs://bkt") == ("bkt", "")
    assert parse_gcs_path("gs://bkt/raw/docs") == ("bkt", "raw/docs")


@pytest.mark.parametrize("bad", ["s3://bkt/x", "gs://", "gs:///x", "data/local"])
def test_parse_gcs_path_invalid(bad: str) -> None:
    with pytest.raises(SourceError, match="gs://bucket"):
        parse_gcs_path(bad)


def test_scheme_routing() -> None:
    assert isinstance(get_document_source("gs://bkt/raw"), GCSDocumentSource)
    assert isinstance(get_document_source("data/invoices"), LocalDocumentSource)


def test_explicit_project_is_passed_to_storage_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    projects: list[str | None] = []
    client = _FakeStorageClient([_FakeBlob("raw/docs/a.html")])

    def _client(*, project: str | None = None) -> _FakeStorageClient:
        projects.append(project)
        return client

    monkeypatch.setattr(
        "stel.sources.gcs._storage", lambda: SimpleNamespace(Client=_client)
    )

    refs = GCSDocumentSource().discover(_cfg(project="econ-prod"), tmp_path)

    assert projects == ["econ-prod"]
    assert refs[0].source_metadata is not None
    assert refs[0].source_metadata["project"] == "econ-prod"


def test_missing_adc_is_configuration_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _client(*, project: str | None = None) -> None:
        raise DefaultCredentialsError("credentials unavailable")

    monkeypatch.setattr(
        "stel.sources.gcs._storage", lambda: SimpleNamespace(Client=_client)
    )

    with pytest.raises(ConfigError, match="Application Default Credentials"):
        GCSDocumentSource().discover(_cfg(), tmp_path)


# ─── discovery ──────────────────────────────────────────────────────────────


def test_discover_identity_and_lineage(tmp_path: Path) -> None:
    blob = _FakeBlob(
        "raw/docs/2026/acme-report.html", generation=42, md5_hash="m5==", size=1234
    )
    src = _gcs_source(_FakeStorageClient([blob]))
    refs = src.discover(_cfg(), tmp_path)

    assert len(refs) == 1
    ref = refs[0]
    assert ref.relative_path == "2026/acme-report.html"
    assert ref.document_id == compute_document_id("filings", "2026/acme-report.html")
    assert ref.content_hash == "md5:m5=="
    assert ref.source_uri == "gs://bkt/raw/docs/2026/acme-report.html#42"
    assert ref.source_metadata is not None
    assert ref.source_metadata["generation"] == 42
    assert ref.source_metadata["size"] == 1234
    assert ref.path is None  # nothing downloaded at discovery time


def test_content_hash_fallback_order() -> None:
    assert content_hash_for_blob(_FakeBlob("a", md5_hash="m")) == "md5:m"
    assert content_hash_for_blob(_FakeBlob("a", crc32c="c")) == "crc32c:c"
    assert content_hash_for_blob(_FakeBlob("a", generation=7)) == "gen:7"


def test_changed_generation_changes_hash() -> None:
    before = content_hash_for_blob(_FakeBlob("a", generation=1))
    after = content_hash_for_blob(_FakeBlob("a", generation=2))
    assert before != after


def test_discover_filters_pattern_and_placeholders(tmp_path: Path) -> None:
    blobs = [
        _FakeBlob("raw/docs/"),  # directory placeholder
        _FakeBlob("raw/docs/a.html"),
        _FakeBlob("raw/docs/notes.txt"),
        _FakeBlob("raw/docs/deep/b.html"),
    ]
    src = _gcs_source(_FakeStorageClient(blobs))
    refs = src.discover(_cfg(), tmp_path)
    assert [r.relative_path for r in refs] == ["a.html", "deep/b.html"]

    refs = src.discover(_cfg(recursive=False), tmp_path)
    assert [r.relative_path for r in refs] == ["a.html"]


def test_discover_slash_pattern_matches_full_relative(tmp_path: Path) -> None:
    blobs = [_FakeBlob("raw/docs/2026/a.html"), _FakeBlob("raw/docs/2025/b.html")]
    src = _gcs_source(_FakeStorageClient(blobs))
    refs = src.discover(_cfg(file_pattern="2026/*.html"), tmp_path)
    assert [r.relative_path for r in refs] == ["2026/a.html"]


def test_bounded_listing_raises(tmp_path: Path) -> None:
    blobs = [_FakeBlob(f"raw/docs/{i}.html") for i in range(4)]
    src = _gcs_source(_FakeStorageClient(blobs))
    with pytest.raises(SourceError, match="max_objects"):
        src.discover(_cfg(max_objects=3), tmp_path)
    # and the listing itself was capped, not just post-filtered
    client = src._client
    assert client.list_calls[-1] == ("bkt", "raw/docs/", 4)


def test_sibling_prefix_is_not_ingested(tmp_path: Path) -> None:
    """`gs://bkt/raw/docs` must not match `raw/docs-archive/…` — GCS prefixes are
    raw string matches, so the source normalizes to a directory boundary
    (PR #92 review finding)."""
    blobs = [
        _FakeBlob("raw/docs/a.html"),
        _FakeBlob("raw/docs-archive/b.html"),
        _FakeBlob("raw/docs"),  # object literally named like the prefix
    ]
    src = _gcs_source(_FakeStorageClient(blobs))
    refs = src.discover(_cfg(), tmp_path)
    assert [r.relative_path for r in refs] == ["a.html"]
    # the boundary is enforced at the listing API, not just post-filtered
    assert src._client.list_calls[-1][1] == "raw/docs/"

    # a trailing slash in the configured path behaves identically
    refs = src.discover(_cfg(path="gs://bkt/raw/docs/"), tmp_path)
    assert [r.relative_path for r in refs] == ["a.html"]


# ─── fetch ──────────────────────────────────────────────────────────────────


def test_fetch_downloads_pinned_generation(tmp_path: Path) -> None:
    blob = _FakeBlob("raw/docs/a.html", generation=9, payload=b"<html>hi</html>")
    client = _FakeStorageClient([blob])
    src = _gcs_source(client)
    ref = src.discover(_cfg(), tmp_path)[0]

    local = src.fetch(ref, tmp_path)
    assert local.read_bytes() == b"<html>hi</html>"
    assert local.suffix == ".html"
    assert client.last_bucket is not None
    assert client.last_bucket.requested_generations == [9]


# ─── freshness ──────────────────────────────────────────────────────────────


def test_scan_reports_newest_object(tmp_path: Path) -> None:
    blobs = [
        _FakeBlob("raw/docs/old.html", updated=datetime(2026, 1, 1, tzinfo=UTC)),
        _FakeBlob("raw/docs/new.html", updated=datetime(2026, 6, 1, tzinfo=UTC)),
    ]
    src = _gcs_source(_FakeStorageClient(blobs))
    scan = src.scan(_cfg(), tmp_path)
    assert scan.file_count == 2
    assert scan.newest_name == "new.html"
    assert scan.newest_epoch == datetime(2026, 6, 1, tzinfo=UTC).timestamp()


# ─── end to end: incremental runs against a fake GCS ────────────────────────


@pytest.fixture
def gcs_project(tmp_path: Path) -> Path:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "stel_project.yml").write_text(
        "name: econ\nversion: '0.1.0'\nprofile: econ\n"
    )
    (project / "profiles.yml").write_text(
        "econ:\n  target: dev\n  outputs:\n    dev:\n      warehouse:\n"
        "        type: duckdb\n        path: ./target/db.duckdb\n"
        "        schema: econ\n"
    )
    (project / "sources").mkdir()
    (project / "sources" / "filings.yml").write_text(
        "version: 2\nsources:\n  - name: filings\n    path: gs://bkt/raw\n"
        "    file_pattern: '*.json'\n"
    )
    (project / "models").mkdir()
    (project / "models" / "raw_filings.yml").write_text(
        "version: 2\nmodels:\n  - name: raw_filings\n    source: ref('filings')\n"
        "    extraction:\n      backend: json\n      options:\n"
        "        fields: [ticker, revenue]\n"
        "    materialization: incremental\n"
    )
    return project


def _json_blob(name: str, generation: int, **fields: Any) -> _FakeBlob:
    return _FakeBlob(
        name, generation=generation, payload=json.dumps(fields).encode()
    )


def test_incremental_run_against_fake_gcs(
    gcs_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blobs = [
        _json_blob("raw/acme.json", 1, ticker="ACME", revenue=100),
        _json_blob("raw/msft.json", 1, ticker="MSFT", revenue=200),
    ]
    client = _FakeStorageClient(blobs)
    monkeypatch.setattr(
        GCSDocumentSource, "_make_client", lambda self, project=None: client
    )

    results = run_project(gcs_project)
    raw = results[0]
    assert raw.documents_processed == 2
    assert raw.rows_written == 2

    db = gcs_project / "target" / "db.duckdb"
    con = duckdb.connect(str(db), read_only=True)
    try:
        rows = con.execute(
            'SELECT source_path, ticker, source_uri FROM "db".econ.raw_filings '
            "ORDER BY source_path"
        ).fetchall()
    finally:
        con.close()
    assert rows[0] == ("acme.json", "ACME", "gs://bkt/raw/acme.json#1")
    assert rows[1][2] == "gs://bkt/raw/msft.json#1"

    # unchanged generations → nothing re-downloaded or reprocessed
    results = run_project(gcs_project)
    raw = results[0]
    assert raw.documents_processed == 0
    assert raw.documents_skipped == 2

    # one object rewritten (new generation), one removed
    client.blobs = [
        _json_blob("raw/acme.json", 2, ticker="ACME", revenue=999),
    ]
    results = run_project(gcs_project)
    raw = results[0]
    assert raw.documents_processed == 1
    assert raw.documents_skipped == 0
    assert raw.documents_deleted == 1

    con = duckdb.connect(str(db), read_only=True)
    try:
        rows = con.execute(
            'SELECT source_path, revenue, source_uri FROM "db".econ.raw_filings'
        ).fetchall()
    finally:
        con.close()
    assert rows == [("acme.json", 999, "gs://bkt/raw/acme.json#2")]


def test_freshness_for_gcs_source(
    gcs_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _FakeStorageClient([_json_blob("raw/acme.json", 1, ticker="ACME")])
    monkeypatch.setattr(
        GCSDocumentSource, "_make_client", lambda self, project=None: client
    )

    results = check_freshness(gcs_project)
    assert results[0].status == "pass"
    assert results[0].newest_file == "acme.json"
    assert results[0].file_count == 1


def test_freshness_propagates_source_errors(
    gcs_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken source (here: the max_objects cap) must fail `source
    freshness`, not report a passing no_data (PR #92 review finding)."""
    client = _FakeStorageClient(
        [_json_blob("raw/a.json", 1), _json_blob("raw/b.json", 1)]
    )
    monkeypatch.setattr(
        GCSDocumentSource, "_make_client", lambda self, project=None: client
    )
    filings = gcs_project / "sources" / "filings.yml"
    filings.write_text(
        filings.read_text().replace(
            "    file_pattern: '*.json'\n",
            "    file_pattern: '*.json'\n    max_objects: 1\n",
        )
    )
    with pytest.raises(SourceError, match="max_objects"):
        check_freshness(gcs_project)


def test_missing_gcs_project_is_actionable_exit_2(
    gcs_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _client(*, project: str | None = None) -> None:
        raise OSError(
            "Project was not passed and could not be determined from the environment."
        )

    monkeypatch.setattr(
        "stel.sources.gcs._storage", lambda: SimpleNamespace(Client=_client)
    )

    result = CliRunner().invoke(cli, ["--project-dir", str(gcs_project), "build"])

    assert result.exit_code == 2, result.output
    assert "GOOGLE_CLOUD_PROJECT" in result.output
    assert "project:" in result.output
    assert "Traceback" not in result.output


# ─── optional integration (needs real GCS credentials) ─────────────────────

_GCS_BUCKET = os.environ.get("STEL_GCS_TEST_BUCKET")


@pytest.mark.skipif(
    not _GCS_BUCKET, reason="set STEL_GCS_TEST_BUCKET to run GCS integration"
)
def test_integration_discover_and_fetch(tmp_path: Path) -> None:
    from google.cloud import storage

    prefix = "stel_it_" + os.urandom(3).hex()
    client = storage.Client()
    bucket = client.bucket(_GCS_BUCKET)
    blob = bucket.blob(f"{prefix}/doc.json")
    blob.upload_from_string(json.dumps({"x": 1}))
    try:
        src = GCSDocumentSource()
        cfg = SourceConfig(
            name="it", path=f"gs://{_GCS_BUCKET}/{prefix}", file_pattern="*.json"
        )
        refs = src.discover(cfg, tmp_path)
        assert [r.relative_path for r in refs] == ["doc.json"]
        local = src.fetch(refs[0], tmp_path)
        assert json.loads(local.read_text()) == {"x": 1}
    finally:
        bucket.blob(f"{prefix}/doc.json").delete()


class _ApiErr(Exception):
    def __init__(self, code: int) -> None:
        super().__init__("boom")
        self.code = code


def test_discover_wraps_listing_errors_as_source_error() -> None:
    class _ErrClient(_FakeStorageClient):
        def list_blobs(self, *a: Any, **k: Any) -> Any:
            raise _ApiErr(503)

    source = _gcs_source(_ErrClient([]))
    with pytest.raises(SourceError, match=r"GCS listing.*503"):
        source.discover(_cfg(), Path("."))


def test_fetch_cleans_up_partial_download_on_error(tmp_path: Path) -> None:
    class _FailBlob(_FakeBlob):
        def download_to_filename(self, filename: str) -> None:
            Path(filename).write_bytes(b"partial bytes")  # a partial write...
            raise _ApiErr(500)                              # ...then the transfer fails

    client = _FakeStorageClient([_FailBlob("raw/docs/a.html")])
    source = _gcs_source(client)
    (ref,) = source.discover(_cfg(), tmp_path)

    with pytest.raises(SourceError, match=r"GCS download.*500"):
        source.fetch(ref, tmp_path)
    # No partial file (and no final file) is left behind.
    assert list(tmp_path.iterdir()) == []


def test_fetch_writes_final_file_atomically_on_success(tmp_path: Path) -> None:
    client = _FakeStorageClient([_FakeBlob("raw/docs/a.html", payload=b"hello")])
    source = _gcs_source(client)
    (ref,) = source.discover(_cfg(), tmp_path)

    local = source.fetch(ref, tmp_path)
    assert local.read_bytes() == b"hello"
    # No leftover .partial temp file.
    assert not any(p.name.endswith(".partial") for p in tmp_path.iterdir())


def test_close_releases_client_and_is_idempotent() -> None:
    closed = {"n": 0}

    class _ClosableClient(_FakeStorageClient):
        def close(self) -> None:
            closed["n"] += 1

    source = _gcs_source(_ClosableClient([]))
    source.close()
    source.close()  # idempotent — no client to close the second time
    assert closed["n"] == 1
    assert source._client is None
