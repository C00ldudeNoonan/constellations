"""An imported ontology drives real incremental plans (issue #767 / ALE-419)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import rdflib

from stel.plan import plan_project
from stel.runner import build_project

pytestmark = pytest.mark.e2e

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "ontology_import"


def test_documented_example_builds_offline_and_plans_file_changes(tmp_path: Path) -> None:
    project = tmp_path / "example"
    shutil.copytree(EXAMPLE, project, ignore=shutil.ignore_patterns("target", "__pycache__"))
    result = build_project(project)
    assert not result.skipped
    assert all(not model.errors for model in result.run_results)
    assert plan_project(project).models[0].status == "unchanged"

    # Re-serializing the file reorders every statement and drops its comments;
    # identity follows the allow-list, not the file's layout.
    ontology = project / "ontologies" / "fibo_subset.ttl"
    graph = rdflib.Graph().parse(ontology)
    # N-Triples is a subset of Turtle, so the `.ttl` suffix still holds.
    ontology.write_text(graph.serialize(format="nt"), encoding="utf-8")
    assert plan_project(project).models[0].status == "unchanged"

    text = ontology.read_text(encoding="utf-8")
    assert text.count('"Germany"@en') == 1
    ontology.write_text(
        text.replace('"Germany"@en', '"Federal Republic of Germany"@en'), encoding="utf-8"
    )
    assert plan_project(project).models[0].status == "changed"
    rebuilt = build_project(project)
    assert all(not model.errors for model in rebuilt.run_results)
    assert plan_project(project).models[0].status == "unchanged"
