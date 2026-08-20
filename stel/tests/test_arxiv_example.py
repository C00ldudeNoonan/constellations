"""End-to-end test of the arxiv_papers example + its quality checks, plus the
synthetic arXiv generator. No network, no LLM."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from stel.checks import run_project_tests
from stel.runner import run_project
from stel.synth import generate_arxiv_papers


@pytest.fixture
def arxiv_project(tmp_path: Path) -> Path:
    repo = Path(__file__).resolve().parents[1]
    src = repo / "examples" / "arxiv_papers"
    dst = tmp_path / "arxiv"
    shutil.copytree(
        src, dst, ignore=shutil.ignore_patterns("data", "target", "__pycache__")
    )
    return dst


def test_synth_arxiv_shape(tmp_path: Path) -> None:
    paths = generate_arxiv_papers(5, tmp_path, seed=1)
    assert len(paths) == 5
    rec = json.loads(paths[0].read_text())
    assert set(rec) >= {
        "arxiv_id", "title", "authors", "n_authors",
        "primary_category", "published", "abstract",
    }
    # title is embedded in the abstract → grounded_in passes on clean data
    assert rec["title"] in rec["abstract"]


def test_synth_arxiv_deterministic(tmp_path: Path) -> None:
    a = generate_arxiv_papers(4, tmp_path / "a", seed=7)
    b = generate_arxiv_papers(4, tmp_path / "b", seed=7)
    for pa, pb in zip(a, b, strict=True):
        assert pa.read_text() == pb.read_text()


def test_arxiv_pipeline_all_checks_pass(arxiv_project: Path) -> None:
    generate_arxiv_papers(30, arxiv_project / "data" / "papers", seed=3)
    run_project(arxiv_project)
    results = run_project_tests(arxiv_project)
    assert len(results) >= 12
    failed = [r for r in results if not r.passed]
    assert failed == [], f"unexpected failures: {[(r.test_name, r.message) for r in failed]}"


def test_arxiv_grounded_in_catches_hallucinated_title(arxiv_project: Path) -> None:
    papers_dir = arxiv_project / "data" / "papers"
    generate_arxiv_papers(10, papers_dir, seed=3)

    # Simulate a hallucinated extraction: title absent from the abstract.
    target = papers_dir / "paper_00000.json"
    rec = json.loads(target.read_text())
    rec["title"] = "A Title That Does Not Appear In The Abstract Anywhere"
    target.write_text(json.dumps(rec))

    run_project(arxiv_project)
    results = run_project_tests(arxiv_project)
    grounded = next(r for r in results if r.test_name == "grounded_in")
    assert grounded.status == "fail"
    assert "not grounded" in grounded.message
