"""Source freshness: warn / error when source documents are too old.

Works for any document source (local mtimes, GCS `updated` timestamps) via
the DocumentSource.scan() seam.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .config import load_project
from .config.source import SourceConfig
from .profile import apply_source_path_overrides, resolve_profile
from .sources import get_document_source


@dataclass
class FreshnessResult:
    source_name: str
    status: str  # "pass" | "warn" | "fail" | "no_data"
    newest_age_seconds: float | None
    newest_file: str | None
    file_count: int
    message: str = ""


def check_freshness(
    project_dir: Path,
    *,
    target: str | None = None,
    profiles_dir: Path | None = None,
) -> list[FreshnessResult]:
    project, sources, _ = load_project(project_dir)
    resolved = resolve_profile(
        project, project_dir, target=target, profiles_dir=profiles_dir
    )
    sources = apply_source_path_overrides(sources, resolved)
    results: list[FreshnessResult] = []
    for source in sources:
        results.append(_check_one(source, project_dir))
    return results


def _check_one(source: SourceConfig, project_dir: Path) -> FreshnessResult:
    backend = get_document_source(source.path)
    # SourceError (bad path, auth, max_objects cap) propagates: a broken
    # source must fail the command, not report a passing no_data.
    scan = backend.scan(source, project_dir)
    if not scan.exists or scan.file_count == 0 or scan.newest_epoch is None:
        return FreshnessResult(
            source_name=source.name,
            status="no_data",
            newest_age_seconds=None,
            newest_file=None,
            file_count=scan.file_count,
            message=scan.message or "no matching files",
        )

    age = time.time() - scan.newest_epoch
    relative = scan.newest_name

    if source.freshness is None:
        return FreshnessResult(
            source_name=source.name,
            status="pass",
            newest_age_seconds=age,
            newest_file=relative,
            file_count=scan.file_count,
            message="no freshness thresholds configured",
        )

    fresh = source.freshness
    if fresh.error_after and age >= fresh.error_after.to_seconds():
        status = "fail"
        msg = (
            f"newest file is {_fmt_age(age)} old "
            f"(threshold: {fresh.error_after.count} {fresh.error_after.period})"
        )
    elif fresh.warn_after and age >= fresh.warn_after.to_seconds():
        status = "warn"
        msg = (
            f"newest file is {_fmt_age(age)} old "
            f"(threshold: {fresh.warn_after.count} {fresh.warn_after.period})"
        )
    else:
        status = "pass"
        msg = f"newest file is {_fmt_age(age)} old"

    return FreshnessResult(
        source_name=source.name,
        status=status,
        newest_age_seconds=age,
        newest_file=relative,
        file_count=scan.file_count,
        message=msg,
    )


def _fmt_age(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds / 60)}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"
