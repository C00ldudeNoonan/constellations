"""Document-source seam (issue #84).

Discovery, identity, and content access for source documents live behind
`DocumentSource`, keyed by the URI scheme of `SourceConfig.path` — the same
shape as the warehouse adapter seam. Local filesystem is the reference
implementation; GCS is the first remote one.

The contract that makes incremental processing work without downloading
anything: `discover()` must produce a stable `document_id` (from the
source-relative path) and a `content_hash` that changes when the object's
bytes change — from a *listing*, not from content. Backends still consume a
local `Path`, so `fetch()` materializes one document into a per-run scratch
directory only when the runner has decided it actually needs processing.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config.source import SourceConfig


class SourceError(Exception):
    pass


@dataclass
class DocumentRef:
    source_name: str
    relative_path: str
    document_id: str
    content_hash: str
    path: Path | None = None  # set when the document is already a local file
    source_uri: str | None = None  # e.g. gs://bucket/name#generation
    source_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class SourceScan:
    """Listing summary for freshness checks."""

    exists: bool
    file_count: int
    newest_epoch: float | None
    newest_name: str | None
    message: str = ""


class DocumentSource(ABC):
    @abstractmethod
    def discover(
        self, source: SourceConfig, project_dir: Path
    ) -> list[DocumentRef]:
        """Deterministically list matching documents with stable identity."""

    @abstractmethod
    def fetch(self, ref: DocumentRef, work_dir: Path) -> Path:
        """Make the document's bytes available as a local file. Local sources
        return the original path; remote sources download into `work_dir`
        (a per-run scratch directory the runner cleans up)."""

    @abstractmethod
    def scan(self, source: SourceConfig, project_dir: Path) -> SourceScan:
        """Cheap listing summary for `source freshness`."""
