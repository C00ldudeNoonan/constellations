from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from stel.config.model import SearchFreshnessConfig
from stel.config.source import DurationSpec, FreshnessConfig
from stel.freshness import check_freshness


def _bootstrap(
    tmp_path: Path,
    *,
    freshness: str | None = None,
    create_files: list[str] | None = None,
    mtime_seconds_ago: dict[str, int] | None = None,
) -> Path:
    """Create a minimal project with one source and optional files."""
    (tmp_path / "stel_project.yml").write_text(
        "name: t\n"
        "duckdb:\n  path: ./target/t.duckdb\n  schema: t\n"
    )
    (tmp_path / "sources").mkdir()
    body = (
        "version: 2\n"
        "sources:\n"
        "  - name: s\n"
        "    path: ./data/\n"
        "    file_pattern: '*.json'\n"
    )
    if freshness:
        body += freshness
    (tmp_path / "sources" / "s.yml").write_text(body)
    if create_files:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        now = time.time()
        for f in create_files:
            p = data_dir / f
            p.write_text("{}")
            if mtime_seconds_ago and f in mtime_seconds_ago:
                old = now - mtime_seconds_ago[f]
                os.utime(p, (old, old))
    return tmp_path


def test_no_files_yields_no_data(tmp_path: Path) -> None:
    project = _bootstrap(tmp_path)
    results = check_freshness(project)
    assert len(results) == 1
    assert results[0].status == "no_data"


def test_pass_when_under_warn_threshold(tmp_path: Path) -> None:
    project = _bootstrap(
        tmp_path,
        freshness=(
            "    freshness:\n"
            "      warn_after: { count: 1, period: hour }\n"
        ),
        create_files=["a.json", "b.json"],
        mtime_seconds_ago={"a.json": 60, "b.json": 30},
    )
    results = check_freshness(project)
    assert results[0].status == "pass"
    assert results[0].file_count == 2


def test_warn_when_past_warn_threshold(tmp_path: Path) -> None:
    project = _bootstrap(
        tmp_path,
        freshness=(
            "    freshness:\n"
            "      warn_after: { count: 1, period: minute }\n"
            "      error_after: { count: 1, period: hour }\n"
        ),
        create_files=["a.json"],
        mtime_seconds_ago={"a.json": 120},  # 2 minutes old
    )
    results = check_freshness(project)
    assert results[0].status == "warn"


def test_fail_when_past_error_threshold(tmp_path: Path) -> None:
    project = _bootstrap(
        tmp_path,
        freshness=(
            "    freshness:\n"
            "      warn_after: { count: 1, period: minute }\n"
            "      error_after: { count: 5, period: minute }\n"
        ),
        create_files=["a.json"],
        mtime_seconds_ago={"a.json": 600},  # 10 minutes old
    )
    results = check_freshness(project)
    assert results[0].status == "fail"


def test_no_thresholds_returns_pass(tmp_path: Path) -> None:
    project = _bootstrap(
        tmp_path,
        create_files=["a.json"],
        mtime_seconds_ago={"a.json": 10_000},
    )
    results = check_freshness(project)
    assert results[0].status == "pass"
    assert "no freshness thresholds" in results[0].message


def test_missing_source_directory(tmp_path: Path) -> None:
    project = _bootstrap(tmp_path)
    results = check_freshness(project)
    assert results[0].status == "no_data"
    assert "does not exist" in results[0].message


def test_a_nonpositive_duration_is_refused_by_configuration() -> None:
    """A threshold of zero or less is refused where it is declared.

    Every reader of a `DurationSpec` compares an age against it, so a
    non-positive count means "instantly overdue" -- which nobody declares on
    purpose. It is rejected by the config model rather than by whichever
    reader gets there first, so a bad threshold fails `stel compile` instead
    of the MCP server that later has to interpret the compiled artifact
    (PR #685 review).
    """
    assert DurationSpec(count=1, period="hour").to_seconds() == 3600
    for count in (0, -1):
        with pytest.raises(ValidationError, match="greater than 0"):
            DurationSpec(count=count, period="hour")
        # Both readers of the shape, so neither can regain it on its own.
        with pytest.raises(ValidationError, match="greater than 0"):
            FreshnessConfig(warn_after={"count": count, "period": "hour"})
        with pytest.raises(ValidationError, match="greater than 0"):
            SearchFreshnessConfig(stale_after={"count": count, "period": "day"})
