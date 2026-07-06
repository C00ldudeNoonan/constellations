"""GCS document source: `path: gs://bucket/prefix` in a source YAML.

Identity comes from the object *listing*, never from content: document_id
hashes the prefix-relative object name, and content_hash prefers the listed
md5 (stable across metadata-only rewrites), falling back to crc32c, then to
the generation number. Unchanged objects are skipped incrementally without
downloading a byte; `fetch()` downloads exactly the listed generation of a
document the runner decided to process.

Auth is Application Default Credentials — `gcloud auth application-default
login` locally, or GOOGLE_APPLICATION_CREDENTIALS pointing at a
service-account JSON in CI. google-cloud-storage ships as the `gcs` extra.

Listing is bounded by `max_objects` on the source (default 5000) so a typo'd
prefix cannot silently crawl a whole bucket.
"""
from __future__ import annotations

import fnmatch
from pathlib import Path, PurePosixPath
from typing import Any

from ..config.source import SourceConfig
from ..versioning import compute_document_id
from .base import DocumentRef, DocumentSource, SourceError, SourceScan

_INSTALL_HINT = (
    "GCS sources require google-cloud-storage. "
    "Install it with: pip install 'dbt-ml[gcs]'"
)


def _storage() -> Any:
    try:
        from google.cloud import storage  # type: ignore[attr-defined]
    except ImportError as e:
        raise SourceError(_INSTALL_HINT) from e
    return storage


def parse_gcs_path(path: str) -> tuple[str, str]:
    """`gs://bucket[/prefix]` → (bucket, prefix). Prefix may be empty."""
    rest = path.removeprefix("gs://")
    if rest == path or not rest or rest.startswith("/"):
        raise SourceError(
            f"Invalid GCS source path {path!r}: expected gs://bucket[/prefix]"
        )
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix


def _matches(relative: str, pattern: str, recursive: bool) -> bool:
    if not recursive and "/" in relative:
        return False
    target = relative if "/" in pattern else PurePosixPath(relative).name
    return fnmatch.fnmatch(target, pattern)


def content_hash_for_blob(blob: Any) -> str:
    """Prefixed so hashes from different mechanisms can never collide."""
    if getattr(blob, "md5_hash", None):
        return f"md5:{blob.md5_hash}"
    if getattr(blob, "crc32c", None):
        return f"crc32c:{blob.crc32c}"
    return f"gen:{blob.generation}"


class GCSDocumentSource(DocumentSource):
    def __init__(self) -> None:
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._make_client()
        return self._client

    def _make_client(self) -> Any:
        return _storage().Client()

    def discover(self, source: SourceConfig, project_dir: Path) -> list[DocumentRef]:
        bucket_name, prefix = parse_gcs_path(source.path)
        # Directory-boundary semantics: GCS prefixes are raw string matches,
        # so listing `raw/doc` would also return `raw/docs/…`. Normalizing
        # to a trailing slash keeps sibling prefixes out.
        list_prefix = f"{prefix.rstrip('/')}/" if prefix else ""
        listed = list(
            self._get_client().list_blobs(
                bucket_name,
                prefix=list_prefix or None,
                max_results=source.max_objects + 1,
            )
        )
        if len(listed) > source.max_objects:
            raise SourceError(
                f"Source '{source.name}': more than {source.max_objects} objects "
                f"under gs://{bucket_name}/{list_prefix}. Narrow the prefix or "
                "raise `max_objects` on the source."
            )

        refs: list[DocumentRef] = []
        for blob in sorted(listed, key=lambda b: b.name):
            if blob.name.endswith("/"):  # directory placeholder objects
                continue
            if list_prefix and not blob.name.startswith(list_prefix):
                continue
            relative = blob.name.removeprefix(list_prefix)
            if not relative or not _matches(
                relative, source.file_pattern, source.recursive
            ):
                continue
            metadata: dict[str, Any] = {
                "bucket": bucket_name,
                "name": blob.name,
                "generation": blob.generation,
                "size": blob.size,
                "updated": blob.updated.isoformat() if blob.updated else None,
                "content_type": blob.content_type,
                "md5_hash": getattr(blob, "md5_hash", None),
                "crc32c": getattr(blob, "crc32c", None),
                "etag": getattr(blob, "etag", None),
            }
            refs.append(
                DocumentRef(
                    source_name=source.name,
                    relative_path=relative,
                    document_id=compute_document_id(source.name, relative),
                    content_hash=content_hash_for_blob(blob),
                    source_uri=f"gs://{bucket_name}/{blob.name}#{blob.generation}",
                    source_metadata={
                        k: v for k, v in metadata.items() if v is not None
                    },
                )
            )
        return refs

    def fetch(self, ref: DocumentRef, work_dir: Path) -> Path:
        assert ref.source_metadata is not None
        meta = ref.source_metadata
        bucket = self._get_client().bucket(meta["bucket"])
        # Pin the listed generation: a concurrent rewrite between discovery
        # and fetch yields the listed bytes, not silently newer ones.
        blob = bucket.blob(meta["name"], generation=meta.get("generation"))
        suffix = PurePosixPath(ref.relative_path).suffix
        local = work_dir / f"{ref.document_id}{suffix}"
        blob.download_to_filename(str(local))
        return local

    def scan(self, source: SourceConfig, project_dir: Path) -> SourceScan:
        refs = self.discover(source, project_dir)
        if not refs:
            return SourceScan(
                exists=True,
                file_count=0,
                newest_epoch=None,
                newest_name=None,
                message="no matching objects",
            )
        from datetime import datetime

        newest_epoch: float | None = None
        newest_name: str | None = None
        for ref in refs:
            updated = (ref.source_metadata or {}).get("updated")
            if not updated:
                continue
            epoch = datetime.fromisoformat(updated).timestamp()
            if newest_epoch is None or epoch > newest_epoch:
                newest_epoch = epoch
                newest_name = ref.relative_path
        return SourceScan(
            exists=True,
            file_count=len(refs),
            newest_epoch=newest_epoch,
            newest_name=newest_name,
        )
