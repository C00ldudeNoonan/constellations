"""`stel plan --pending` reports rows waiting upstream (issue #751)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from stel.adapters import create_adapter
from stel.cli import cli
from stel.config.loader import load_project
from stel.pending import CHANGED_REASON, PENDING_SCHEMA_VERSION
from stel.profile import resolve_profile
from stel.runner import run_project

# Runs a whole project, so it belongs to the `e2e` tier (issue #518).
pytestmark = pytest.mark.e2e

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def rag_project(tmp_path: Path) -> Path:
    pytest.importorskip("lancedb")
    dst = tmp_path / "rag_chunks_pipeline"
    shutil.copytree(
        EXAMPLES / "rag_chunks_pipeline",
        dst,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    return dst


def _pending(project: Path, *args: str) -> tuple[int, dict[str, Any]]:
    result = CliRunner().invoke(
        cli,
        ["--project-dir", str(project), "plan", "--pending", "--json", *args],
    )
    assert result.exit_code in (0, 3), result.output
    return result.exit_code, json.loads(result.output)


def test_pending_exit_code_and_report_follow_the_state_table(rag_project: Path) -> None:
    run_project(rag_project)

    code, report = _pending(rag_project, "--select", "chunk_embeddings")
    assert code == 0
    assert report["metadata"]["schema_version"] == PENDING_SCHEMA_VERSION
    assert report["metadata"]["has_pending"] is False
    (settled,) = report["models"]
    assert (settled["status"], settled["new_rows"]) == ("none", 0)

    project, _, _ = load_project(rag_project)
    resolved = resolve_profile(project, rag_project, target=None, profiles_dir=None)
    with create_adapter(resolved.warehouse, project_dir=rag_project) as adapter:
        state = f"{adapter.schema_ref}.{adapter.quote_ident('stel_state')}"
        adapter.execute(
            f"DELETE FROM {state} WHERE model_name = 'chunk_embeddings' AND record_key = "
            f"(SELECT MIN(record_key) FROM {state} WHERE model_name = 'chunk_embeddings')"
        )

    code, report = _pending(rag_project, "--select", "chunk_embeddings")
    assert code == 3
    (waiting,) = report["models"]
    assert (waiting["status"], waiting["new_rows"]) == ("pending", 1)
    assert waiting["changed_rows"] is None
    assert waiting["changed_reason"] == CHANGED_REASON


def test_pending_reports_kinds_it_cannot_read_as_unsupported(rag_project: Path) -> None:
    run_project(rag_project)

    _, report = _pending(rag_project, "--select", "chunk_search")

    (model,) = report["models"]
    assert model["status"] == "unsupported"
    assert model["new_rows"] is None


def test_dry_run_requires_pending(rag_project: Path) -> None:
    result = CliRunner().invoke(
        cli, ["--project-dir", str(rag_project), "plan", "--dry-run"]
    )
    assert result.exit_code == 2
    assert "--pending" in result.output


def test_include_changed_counts_rows_whose_input_fingerprint_moved(
    rag_project: Path,
) -> None:
    """Issue #751: `changed` needs the row fingerprint, so it is opt-in, and it
    uses the runner's own fingerprint so a settled project reports zero."""
    run_project(rag_project)

    code, report = _pending(
        rag_project, "--select", "chunk_embeddings", "--include-changed"
    )
    assert code == 0
    assert report["models"][0]["changed_rows"] == 0

    project, _, _ = load_project(rag_project)
    resolved = resolve_profile(project, rag_project, target=None, profiles_dir=None)
    with create_adapter(resolved.warehouse, project_dir=rag_project) as adapter:
        state = f"{adapter.schema_ref}.{adapter.quote_ident('stel_state')}"
        adapter.execute(
            f"UPDATE {state} SET input_fingerprint = 'stale' "
            "WHERE model_name = 'chunk_embeddings' AND record_key = "
            f"(SELECT MIN(record_key) FROM {state} WHERE model_name = 'chunk_embeddings')"
        )

    code, report = _pending(
        rag_project, "--select", "chunk_embeddings", "--include-changed"
    )
    assert code == 3
    (changed,) = report["models"]
    assert (changed["status"], changed["new_rows"], changed["changed_rows"]) == (
        "pending",
        0,
        1,
    )
    assert changed["changed_reason"] is None

    # Without the flag the same state is invisible, and says why.
    code, report = _pending(rag_project, "--select", "chunk_embeddings")
    assert code == 0
    assert report["models"][0]["changed_rows"] is None
    assert "--include-changed" in report["models"][0]["changed_reason"]


def test_include_changed_agrees_with_the_chunk_runner_fingerprint(
    rag_project: Path,
) -> None:
    """A fingerprint that drifted from `chunk_input_hash` would report every
    settled document as changed."""
    run_project(rag_project)

    code, report = _pending(
        rag_project, "--select", "document_chunks", "--include-changed"
    )

    assert code == 0
    (model,) = report["models"]
    assert (model["status"], model["changed_rows"]) == ("none", 0)


def test_include_changed_requires_pending(rag_project: Path) -> None:
    result = CliRunner().invoke(
        cli, ["--project-dir", str(rag_project), "plan", "--include-changed"]
    )
    assert result.exit_code == 2
    assert "--pending" in result.output
