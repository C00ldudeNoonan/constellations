from __future__ import annotations

from pathlib import Path

from ..config.source import SourceConfig
from ..versioning import compute_content_hash, compute_document_id
from .base import DocumentRef, DocumentSource, SourceScan


class LocalDocumentSource(DocumentSource):
    """Files on disk under `<project_dir>/<source.path>`."""

    def discover(self, source: SourceConfig, project_dir: Path) -> list[DocumentRef]:
        source_dir = (project_dir / source.path).resolve()
        if not source_dir.exists():
            return []
        pattern = (
            f"**/{source.file_pattern}" if source.recursive else source.file_pattern
        )
        files = sorted(p for p in source_dir.glob(pattern) if p.is_file())
        refs: list[DocumentRef] = []
        for p in files:
            # POSIX separators so document_id is stable across OSes (issue #67)
            relative_path = p.relative_to(source_dir).as_posix()
            refs.append(
                DocumentRef(
                    source_name=source.name,
                    relative_path=relative_path,
                    document_id=compute_document_id(source.name, relative_path),
                    content_hash=compute_content_hash(p),
                    path=p,
                )
            )
        return refs

    def fetch(self, ref: DocumentRef, work_dir: Path) -> Path:
        assert ref.path is not None
        return ref.path

    def scan(self, source: SourceConfig, project_dir: Path) -> SourceScan:
        source_dir = (project_dir / source.path).resolve()
        if not source_dir.exists():
            return SourceScan(
                exists=False,
                file_count=0,
                newest_epoch=None,
                newest_name=None,
                message=f"source path does not exist: {source_dir}",
            )
        pattern = (
            f"**/{source.file_pattern}" if source.recursive else source.file_pattern
        )
        files = [p for p in source_dir.glob(pattern) if p.is_file()]
        if not files:
            return SourceScan(
                exists=True,
                file_count=0,
                newest_epoch=None,
                newest_name=None,
                message="no matching files",
            )
        newest = max(files, key=lambda p: p.stat().st_mtime)
        return SourceScan(
            exists=True,
            file_count=len(files),
            newest_epoch=newest.stat().st_mtime,
            newest_name=newest.relative_to(source_dir).as_posix(),
        )
