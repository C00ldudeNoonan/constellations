"""GCS document sources (issue #84): discovery from listings, identity
without download, bounded scanning, generation-pinned fetch, and the full
incremental loop against a fake storage client. Real GCS runs only when
DBT_ML_GCS_TEST_BUCKET is set."""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pytest

from dbt_ml.config.source import SourceConfig
from dbt_ml.freshness import check_freshness
from dbt_ml.runner import run_project
from dbt_ml.sources import (
    GCSDocumentSource,
    LocalDocumentSource,
    SourceError,
    get_document_source,
)
from dbt_ml.sources.gcs import content_hash_for_blob, parse_gcs_path
from dbt_ml.versioning import compute_document_id

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
        "path": "gs://bkt/raw/sec",
        "file_pattern": "*.html",
    }
    return SourceConfig(**{**defaults, **kwargs})


# ─── path parsing + routing ─────────────────────────────────────────────────


def test_parse_gcs_path() -> None:
    assert parse_gcs_path("gs://bkt") == ("bkt", "")
    assert parse_gcs_path("gs://bkt/raw/sec") == ("bkt", "raw/sec")


@pytest.mark.parametrize("bad", ["s3://bkt/x", "gs://", "gs:///x", "data/local"])
def test_parse_gcs_path_invalid(bad: str) -> None:
    with pytest.raises(SourceError, match="gs://bucket"):
        parse_gcs_path(bad)


def test_scheme_routing() -> None:
    assert isinstance(get_document_source("gs://bkt/raw"), GCSDocumentSource)
    assert isinstance(get_document_source("data/invoices"), LocalDocumentSource)


# ─── discovery ──────────────────────────────────────────────────────────────


def test_discover_identity_and_lineage(tmp_path: Path) -> None:
    blob = _FakeBlob(
        "raw/sec/2026/aapl-10k.html", generation=42, md5_hash="m5==", size=1234
    )
    src = _gcs_source(_FakeStorageClient([blob]))
    refs = src.discover(_cfg(), tmp_path)

    assert len(refs) == 1
    ref = refs[0]
    assert ref.relative_path == "2026/aapl-10k.html"
    assert ref.document_id == compute_document_id("filings", "2026/aapl-10k.html")
    assert ref.content_hash == "md5:m5=="
    assert ref.source_uri == "gs://bkt/raw/sec/2026/aapl-10k.html#42"
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
        _FakeBlob("raw/sec/"),  # directory placeholder
        _FakeBlob("raw/sec/a.html"),
        _FakeBlob("raw/sec/notes.txt"),
        _FakeBlob("raw/sec/deep/b.html"),
    ]
    src = _gcs_source(_FakeStorageClient(blobs))
    refs = src.discover(_cfg(), tmp_path)
    assert [r.relative_path for r in refs] == ["a.html", "deep/b.html"]

    refs = src.discover(_cfg(recursive=False), tmp_path)
    assert [r.relative_path for r in refs] == ["a.html"]


def test_discover_slash_pattern_matches_full_relative(tmp_path: Path) -> None:
    blobs = [_FakeBlob("raw/sec/2026/a.html"), _FakeBlob("raw/sec/2025/b.html")]
    src = _gcs_source(_FakeStorageClient(blobs))
    refs = src.discover(_cfg(file_pattern="2026/*.html"), tmp_path)
    assert [r.relative_path for r in refs] == ["2026/a.html"]


def test_bounded_listing_raises(tmp_path: Path) -> None:
    blobs = [_FakeBlob(f"raw/sec/{i}.html") for i in range(4)]
    src = _gcs_source(_FakeStorageClient(blobs))
    with pytest.raises(SourceError, match="max_objects"):
        src.discover(_cfg(max_objects=3), tmp_path)
    # and the listing itself was capped, not just post-filtered
    client = src._client
    assert client.list_calls[-1] == ("bkt", "raw/sec/", 4)


def test_sibling_prefix_is_not_ingested(tmp_path: Path) -> None:
    """`gs://bkt/raw/sec` must not match `raw/secret/…` — GCS prefixes are
    raw string matches, so the source normalizes to a directory boundary
    (PR #92 review finding)."""
    blobs = [
        _FakeBlob("raw/sec/a.html"),
        _FakeBlob("raw/secret/b.html"),
        _FakeBlob("raw/sec"),  # object literally named like the prefix
    ]
    src = _gcs_source(_FakeStorageClient(blobs))
    refs = src.discover(_cfg(), tmp_path)
    assert [r.relative_path for r in refs] == ["a.html"]
    # the boundary is enforced at the listing API, not just post-filtered
    assert src._client.list_calls[-1][1] == "raw/sec/"

    # a trailing slash in the configured path behaves identically
    refs = src.discover(_cfg(path="gs://bkt/raw/sec/"), tmp_path)
    assert [r.relative_path for r in refs] == ["a.html"]


# ─── fetch ──────────────────────────────────────────────────────────────────


def test_fetch_downloads_pinned_generation(tmp_path: Path) -> None:
    blob = _FakeBlob("raw/sec/a.html", generation=9, payload=b"<html>hi</html>")
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
        _FakeBlob("raw/sec/old.html", updated=datetime(2026, 1, 1, tzinfo=UTC)),
        _FakeBlob("raw/sec/new.html", updated=datetime(2026, 6, 1, tzinfo=UTC)),
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
    (project / "dbt_ml_project.yml").write_text(
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
        _json_blob("raw/aapl.json", 1, ticker="AAPL", revenue=100),
        _json_blob("raw/msft.json", 1, ticker="MSFT", revenue=200),
    ]
    client = _FakeStorageClient(blobs)
    monkeypatch.setattr(GCSDocumentSource, "_make_client", lambda self: client)

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
    assert rows[0] == ("aapl.json", "AAPL", "gs://bkt/raw/aapl.json#1")
    assert rows[1][2] == "gs://bkt/raw/msft.json#1"

    # unchanged generations → nothing re-downloaded or reprocessed
    results = run_project(gcs_project)
    raw = results[0]
    assert raw.documents_processed == 0
    assert raw.documents_skipped == 2

    # one object rewritten (new generation), one removed
    client.blobs = [
        _json_blob("raw/aapl.json", 2, ticker="AAPL", revenue=999),
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
    assert rows == [("aapl.json", 999, "gs://bkt/raw/aapl.json#2")]


def test_freshness_for_gcs_source(
    gcs_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _FakeStorageClient([_json_blob("raw/aapl.json", 1, ticker="AAPL")])
    monkeypatch.setattr(GCSDocumentSource, "_make_client", lambda self: client)

    results = check_freshness(gcs_project)
    assert results[0].status == "pass"
    assert results[0].newest_file == "aapl.json"
    assert results[0].file_count == 1


def test_freshness_propagates_source_errors(
    gcs_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken source (here: the max_objects cap) must fail `source
    freshness`, not report a passing no_data (PR #92 review finding)."""
    client = _FakeStorageClient(
        [_json_blob("raw/a.json", 1), _json_blob("raw/b.json", 1)]
    )
    monkeypatch.setattr(GCSDocumentSource, "_make_client", lambda self: client)
    filings = gcs_project / "sources" / "filings.yml"
    filings.write_text(
        filings.read_text().replace(
            "    file_pattern: '*.json'\n",
            "    file_pattern: '*.json'\n    max_objects: 1\n",
        )
    )
    with pytest.raises(SourceError, match="max_objects"):
        check_freshness(gcs_project)


# ─── optional integration (needs real GCS credentials) ─────────────────────

_GCS_BUCKET = os.environ.get("DBT_ML_GCS_TEST_BUCKET")


@pytest.mark.skipif(
    not _GCS_BUCKET, reason="set DBT_ML_GCS_TEST_BUCKET to run GCS integration"
)
def test_integration_discover_and_fetch(tmp_path: Path) -> None:
    from google.cloud import storage

    prefix = "dbt_ml_it_" + os.urandom(3).hex()
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
