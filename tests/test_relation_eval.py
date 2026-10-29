"""The committed offline example is a real regression gate (#731, ALE-425)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import duckdb
import pytest

from stel.manifest import build_manifest
from stel.runner import build_project, run_project

pytestmark = pytest.mark.e2e


@pytest.fixture
def project(tmp_path):
    source = Path(__file__).resolve().parents[1] / "examples" / "economic_nlp"
    destination = tmp_path / "economic_nlp"
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("target", "__pycache__"))
    return destination


def _metrics(project):
    with duckdb.connect(str(project / "target" / "stel.duckdb"), read_only=True) as con:
        return {
            (row[0], row[1]): row[2]
            for row in con.execute(
                "SELECT metric, label, value FROM economic_nlp.relation_quality"
            ).fetchall()
        }


def test_offline_example_runs_the_extractor_and_a_broken_rule_lowers_f1(project):
    """The acceptance test must detect a broken extractor, not just report rows (#731)."""
    result = build_project(project, select="+relation_quality")
    assert {item.model_name for item in result.run_results} == {
        "relation_eval_mentions",
        "relation_eval_gold",
        "relation_eval_assertions",
        "relation_eval_predictions",
        "relation_quality",
    }
    assert len(result.test_results) == 3
    assert all(item.status == "pass" for item in result.test_results)
    before = _metrics(project)
    assert before["f1", None] == 1
    assert before["support", "references_geography"] == 2
    assert before["support", "references_topic"] == 2

    model_path = project / "models" / "relation_eval.yml"
    model_path.write_text(
        model_path.read_text(encoding="utf-8").replace(
            "object_label: GPE", "object_label: NEVER_MATCH"
        ),
        encoding="utf-8",
    )
    broken = build_project(project, select="+relation_quality")
    metrics = _metrics(project)
    assert metrics["f1", None] == pytest.approx(2 / 3)
    assert metrics["f1", None] < before["f1", None]
    assert metrics["recall", "references_geography"] == 0
    assert [item.status for item in broken.test_results].count("fail") == 2


def test_relation_config_is_serialized_in_manifest(project):
    manifest = build_manifest(project)
    model = next(item for item in manifest["models"] if item["name"] == "relation_quality")
    assert model["kind"] == "eval"
    assert model["eval"]["kind"] == "relations"
    assert model["eval"]["document_field"] == "evaluation_document"
    assert model["depends_on"] == ["ref('relation_eval_predictions')", "ref('relation_eval_gold')"]
    # The artifact is ordinary JSON and carries no fixture row content.
    assert "agency:FRB" not in json.dumps(model)


def test_incremental_relation_eval_replaces_removed_label_metrics(project):
    model_path = project / "models" / "relation_eval.yml"
    model_path.write_text(
        model_path.read_text(encoding="utf-8").replace(
            "  - name: relation_quality\n",
            "  - name: relation_quality\n    materialization: incremental\n",
        ),
        encoding="utf-8",
    )
    run_project(project, select="+relation_quality")
    assert _metrics(project)["support", "references_topic"] == 2
    model_path.write_text(
        model_path.read_text(encoding="utf-8")
        .replace("object_label: TOPIC", "object_label: NEVER_MATCH")
        .replace(
            "labels: [references_geography, references_topic]", "labels: [references_geography]"
        ),
        encoding="utf-8",
    )
    for name in ("fed_topic.json", "bea_topic.json"):
        (project / "eval_fixtures" / "gold" / name).unlink()
    run_project(project, select="+relation_quality")
    metrics = _metrics(project)
    assert ("support", "references_topic") not in metrics
    assert metrics["f1", None] == 1
    with duckdb.connect(str(project / "target" / "stel.duckdb"), read_only=True) as con:
        count = con.execute("SELECT COUNT(*) FROM economic_nlp.relation_quality").fetchone()
    assert count == (len(metrics),)
