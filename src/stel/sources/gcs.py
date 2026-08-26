"""GCS document source: `path: gs://bucket/prefix` in a source YAML.

Identity comes from the object *listing*, never from content: document_id
hashes the prefix-relative object name, and content_hash prefers the listed
md5 (stable across metadata-only rewrites), falling back to crc32c, then to
the generation number. Unchanged objects are skipped incrementally without
downloading a byte; `fetch()` downloads exactly the listed generation of a
document the runner decided to process.

Auth is Application Default Credentials — `gcloud auth application-default
login` locally, or GOOGLE_APPLICATION_CREDENTIALS pointing at a
service-account JSON in CI. Set `project:` on the source or
GOOGLE_CLOUD_PROJECT when ADC cannot infer a project. google-cloud-storage
ships as the `gcs` extra.

Listing is bounded by `max_objects` on the source (default 5000) so a typo'd
prefix cannot silently crawl a whole bucket.
"""
from __future__ import annotations

import fnmatch
import importlib
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from ..config.loader import ConfigError
from ..config.source import SourceConfig
from ..versioning import compute_document_id
from .base import DocumentRef, DocumentSource, SourceError, SourceScan

log = logging.getLogger(__name__)

_INSTALL_HINT = (
    "GCS sources require google-cloud-storage. "
    "Install it with: pip install 'stel[gcs]'"
)


def _storage() -> Any:
    try:
        return importlib.import_module("google.cloud.storage")
    except ImportError as e:
        raise SourceError(_INSTALL_HINT) from e


def parse_gcs_path(path: str) -> tuple[str, str]:
    """`gs://bucket[/prefix]` → (bucket, prefix). Prefix may be empty."""
    rest = path.removeprefix("gs://")
    if rest == path or not rest or rest.startswith("/"):
        raise SourceError(
            f"Invalid GCS source path {path!r}: expected gs://bucket[/prefix]"
        )
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix


# How far past `max_objects` a listing may scan before the prefix is judged
# too broad. The cap counts matched documents now, so an unfiltered listing
# still needs a hard stop — otherwise a typo'd prefix crawls a whole bucket
# quietly instead of loudly, which is the protection the old raw-count check
# was really providing (issue #348).
_SCAN_CEILING_MULTIPLIER = 10

_GLOB_METACHARACTERS = ("*", "?", "[")


def _static_filter_prefixes(source_filter: Sequence[str]) -> tuple[str, ...]:
    """The narrowest set of listing prefixes covering every `--source-filter`.

    `--source-filter 'AMAT/*'` against a 30k-object prefix listed all 30k and
    kept ~40. The globs address the source-relative path, so a leading segment
    with no wildcard in it can narrow the listing itself.

    Several filters yield several prefixes, listed separately and unioned
    (issue #378). Collapsing to one shared prefix — or, worse, to `""` when
    they disagree — gave back the whole win the moment a run passed more than
    one filter, which is exactly what a batched backfill does.

    Only whole segments count: each prefix is truncated at the last `/`, so
    `AMAT*` contributes nothing rather than excluding `AMATX/…`. A glob with
    no static segment forces `("",)` — list everything — because nothing
    narrower can cover it.

    Nested prefixes collapse to the shorter one, which also makes the returned
    prefixes mutually disjoint: after the reduction no prefix is a string
    prefix of another, so separate listings cannot return the same object
    twice.
    """
    if not source_filter:
        return ("",)
    statics: set[str] = set()
    for pattern in source_filter:
        cut = min(
            (pattern.find(char) for char in _GLOB_METACHARACTERS if char in pattern),
            default=len(pattern),
        )
        head = pattern[:cut]
        static = head[: head.rfind("/") + 1] if "/" in head else ""
        if not static:
            # This glob can only be served by the full listing, and the full
            # listing covers every other glob too.
            return ("",)
        statics.add(static)
    return tuple(
        sorted(
            candidate
            for candidate in statics
            if not any(
                other != candidate and candidate.startswith(other)
                for other in statics
            )
        )
    )


def _matches(relative: str, pattern: str, recursive: bool) -> bool:
    """Whether a source-relative object path matches a glob, case-sensitively.

    `fnmatch.fnmatch` folds case through `os.path.normcase`, which lowercases
    on Windows — so the same filter against the same bucket selected different
    documents depending on the developer's platform, and `AMAT/*` matched
    `amat/x.html` on Windows only. GCS object names are case-sensitive
    everywhere, so `fnmatchcase` is the honest comparison.

    It is also what keeps the listing pushdown sound (Codex review on #378):
    prefix listing is case-sensitive, so a case-insensitive matcher would
    authoritatively match objects a narrowed prefix can never return.
    """
    if not recursive and "/" in relative:
        return False
    target = relative if "/" in pattern else PurePosixPath(relative).name
    return fnmatch.fnmatchcase(target, pattern)


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

    def _get_client(self, project: str | None = None) -> Any:
        if self._client is None:
            self._client = self._make_client(project)
        return self._client

    def close(self) -> None:
        """Release the cached storage client's HTTP session. Idempotent."""
        client = self._client
        self._client = None
        if client is not None:
            try:
                client.close()
            except Exception:  # pragma: no cover - best-effort teardown
                pass

    @staticmethod
    def _api_error(operation: str, error: Exception) -> SourceError:
        """Map a google-cloud exception to a `SourceError` with a concise,
        response-body-free label. `code` (HTTP status) is surfaced when present
        so a 404/403 (permanent) reads differently from a 429/5xx (transient)."""
        status = getattr(error, "code", None)
        label = type(error).__name__
        if isinstance(status, int) and not isinstance(status, bool):
            label = f"{label} {status}"
        return SourceError(f"GCS {operation} failed [{label}]")

    def _make_client(self, project: str | None = None) -> Any:
        storage = _storage()
        from google.auth.exceptions import DefaultCredentialsError

        try:
            return storage.Client(project=project)
        except DefaultCredentialsError as e:
            raise ConfigError(
                "GCS Application Default Credentials were not found or are invalid. "
                "Run `gcloud auth application-default login` locally, or set "
                "GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON file."
            ) from e
        except OSError as e:
            raise ConfigError(
                "The Google Cloud project for the GCS source could not be determined. "
                "Set `project:` on the source or set GOOGLE_CLOUD_PROJECT. User "
                "Application Default Credentials may not include a default project."
            ) from e

    def _list_prefixes(
        self,
        source: SourceConfig,
        *,
        bucket_name: str,
        listing_prefixes: Sequence[str],
        scan_ceiling: int,
        listing_label: str,
    ) -> Iterator[Any]:
        """Yield every object under each prefix, under one shared scan ceiling.

        Extracted so `discover`'s per-blob branches keep their original depth:
        inlining the prefix loop put them five levels deep, past the four this
        repository allows (Codex review). The ceiling counts across prefixes,
        not per prefix — a filter set that fans out into many listings is
        exactly how an unbounded scan would otherwise slip through.
        """
        scanned = 0
        client = self._get_client(source.project)
        for listing_prefix in listing_prefixes:
            for blob in client.list_blobs(
                bucket_name,
                prefix=listing_prefix or None,
                max_results=scan_ceiling + 1,
            ):
                scanned += 1
                if scanned > scan_ceiling:
                    raise SourceError(
                        f"Source '{source.name}': scanned more than "
                        f"{scan_ceiling} objects under "
                        f"gs://{bucket_name}/{listing_label} without reaching "
                        f"the end of the listing. The prefix is too broad — "
                        "narrow it, or raise `max_objects` on the source."
                    )
                yield blob

    def discover(
        self,
        source: SourceConfig,
        project_dir: Path,
        *,
        source_filter: Sequence[str] = (),
    ) -> list[DocumentRef]:
        bucket_name, prefix = parse_gcs_path(source.path)
        # Directory-boundary semantics: GCS prefixes are raw string matches,
        # so listing `raw/doc` would also return `raw/docs/…`. Normalizing
        # to a trailing slash keeps sibling prefixes out.
        list_prefix = f"{prefix.rstrip('/')}/" if prefix else ""
        # The filter can narrow what is *listed*, but never what `relative` is
        # measured against: document identity is the source-relative path, so
        # stripping a filter-derived prefix would change every document_id.
        listing_prefixes = tuple(
            list_prefix + static for static in _static_filter_prefixes(source_filter)
        )
        log.info(
            "Source '%s': listing %d prefix(es) under gs://%s/%s",
            source.name,
            len(listing_prefixes),
            bucket_name,
            list_prefix,
        )
        # One ceiling across every prefix, not one each: the cap exists to stop
        # a run scanning an unbounded corpus, and a filter set that fans out
        # into many listings should still hit it (issue #378).
        scan_ceiling = source.max_objects * _SCAN_CEILING_MULTIPLIER
        matched: list[Any] = []
        scanned = 0
        # Reads exactly as it did for a single prefix; only a fan-out shows a
        # list, and the prefixes already carry `list_prefix`.
        listing_label = ", ".join(listing_prefixes)
        try:
            for blob in self._list_prefixes(
                source,
                bucket_name=bucket_name,
                listing_prefixes=listing_prefixes,
                scan_ceiling=scan_ceiling,
                listing_label=listing_label,
            ):
                scanned += 1
                if blob.name.endswith("/"):  # directory placeholder objects
                    continue
                if list_prefix and not blob.name.startswith(list_prefix):
                    continue
                relative = blob.name.removeprefix(list_prefix)
                if not relative or not _matches(
                    relative, source.file_pattern, source.recursive
                ):
                    continue
                matched.append(blob)
                # The cap is about the documents this source reads. Counting
                # the raw listing instead let unrelated objects under the same
                # prefix — a sibling pipeline's sidecars — fail a run over
                # files the pattern would have discarded (issue #348).
                if len(matched) > source.max_objects:
                    raise SourceError(
                        f"Source '{source.name}': more than {source.max_objects} "
                        f"documents match under gs://{bucket_name}/"
                        f"{listing_label}. Narrow the prefix or raise "
                        "`max_objects` on the source."
                    )
        except (SourceError, ConfigError):
            raise
        except Exception as e:
            raise self._api_error(
                f"listing gs://{bucket_name}/{listing_label}", e
            ) from e
        log.info(
            "Source '%s': scanned %d object(s), %d matched under gs://%s/%s",
            source.name,
            scanned,
            len(matched),
            bucket_name,
            listing_label,
        )

        refs: list[DocumentRef] = []
        for blob in sorted(matched, key=lambda b: b.name):
            relative = blob.name.removeprefix(list_prefix)
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
                "project": source.project,
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
        bucket = self._get_client(meta.get("project")).bucket(meta["bucket"])
        # Pin the listed generation: a concurrent rewrite between discovery
        # and fetch yields the listed bytes, not silently newer ones.
        blob = bucket.blob(meta["name"], generation=meta.get("generation"))
        suffix = PurePosixPath(ref.relative_path).suffix
        local = work_dir / f"{ref.document_id}{suffix}"
        # Download to a temp path and atomically move into place, cleaning up on
        # any failure — a partial/aborted download must not be left behind as if
        # it were the document (mirrors sources/local.py's contract).
        partial = work_dir / f".{ref.document_id}{suffix}.partial"
        try:
            blob.download_to_filename(str(partial))
        except (SourceError, ConfigError):
            raise
        except Exception as e:
            partial.unlink(missing_ok=True)
            raise self._api_error(f"download of {ref.source_uri or ref.relative_path}", e) from e
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
