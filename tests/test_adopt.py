"""`stel adopt` and the identity fields `stel plan` names (issue #750).

A release can move a model's code_version while changing no row it publishes.
The plan has to say which input moved, so the operator can tell that from a
real change, and adoption has to be a restamp: state only, logged, no rewrite.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from stel.adapters import (
    StateRecord,
    StateScope,
    WarehouseAdapter,
    create_adapter,
    parse_warehouse_config,
)
from stel.cli import cli
from stel.plan import ModelPlan, plan_project
from stel.runner import run_project
from stel.synth import generate_invoices

pytestmark = pytest.mark.e2e

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
_FIELDS = "fields: [invoice_id, vendor, issue_date, line_items, total, currency]"
_FIELDS_NARROWED = "fields: [invoice_id, vendor, issue_date, total, currency]"


@pytest.fixture
def invoice_project(tmp_path: Path) -> Path:
    project = tmp_path / "invoice_pipeline"
    shutil.copytree(
        EXAMPLES / "invoice_pipeline",
        project,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    generate_invoices(5, project / "data" / "invoices", seed=1)
    return project


def _raw(project: Path) -> ModelPlan:
    return {m.name: m for m in plan_project(project).models}["raw_invoices"]


def _change_config(project: Path) -> None:
    path = project / "models" / "raw_invoices.yml"
    text = path.read_text(encoding="utf-8")
    assert text.count(_FIELDS) == 1
    path.write_text(text.replace(_FIELDS, _FIELDS_NARROWED), encoding="utf-8")


def _warehouse(project: Path) -> WarehouseAdapter:
    config = parse_warehouse_config(
        {"type": "duckdb", "path": str(project / "target" / "stel.duckdb")}
    )
    return create_adapter(config, project_dir=project)


def _adopt(project: Path, *args: str) -> str:
    result = CliRunner().invoke(
        cli, ["--project-dir", str(project), "adopt", *args], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    return result.output


def test_plan_names_the_identity_field_a_config_change_moved(invoice_project: Path) -> None:
    run_project(invoice_project)
    _change_config(invoice_project)

    raw = _raw(invoice_project)

    assert raw.status == "changed"
    assert raw.identity_recorded is True
    assert [change.field for change in raw.identity_changes] == ["extraction.options"]
    assert "moved: extraction.options (" in raw.reason
    assert raw.to_dict()["identity_changes"][0]["field"] == "extraction.options"


def test_plan_does_not_guess_when_the_old_identity_was_never_recorded(
    invoice_project: Path,
) -> None:
    """Rows published by a stel that kept no record cannot be compared; saying
    nothing moved would be a false all-clear, so the plan says it is unknown."""
    run_project(invoice_project)
    with _warehouse(invoice_project) as adapter:
        adapter.execute(f'DROP TABLE {adapter.schema_ref}."stel_code_identity"')
    _change_config(invoice_project)

    raw = _raw(invoice_project)

    assert raw.identity_recorded is False
    assert raw.identity_changes == ()
    assert "not recorded" in raw.reason


def test_adopt_restamps_state_and_rewrites_nothing(invoice_project: Path) -> None:
    run_project(invoice_project)
    _change_config(invoice_project)
    assert _raw(invoice_project).status == "changed"
    with _warehouse(invoice_project) as adapter:
        before = adapter.rows(f"SELECT * FROM {adapter.schema_ref}.raw_invoices ORDER BY 1")
        old_versions = set(adapter.state_code_version_counts(StateScope("raw_invoices")))

    output = _adopt(invoice_project, "--select", "raw_invoices")

    assert "restamped 5 rows" in output
    raw = _raw(invoice_project)
    assert (raw.status, raw.stale_rows) == ("unchanged", 0)
    with _warehouse(invoice_project) as adapter:
        assert adapter.rows(f"SELECT * FROM {adapter.schema_ref}.raw_invoices ORDER BY 1") == before
        counts = adapter.state_code_version_counts(StateScope("raw_invoices"))
        assert counts == {raw.code_version: 5}
        logged = adapter.rows(
            "SELECT model_name, old_code_version, new_code_version, rows_adopted, "
            f"adopted_by FROM {adapter.schema_ref}.stel_adoption_log"
        )
    assert len(logged) == 1
    model, old, new, rows, who = logged[0]
    assert (model, new, rows) == ("raw_invoices", raw.code_version, 5)
    assert old in old_versions
    assert who


def test_adopt_dry_run_changes_nothing(invoice_project: Path) -> None:
    run_project(invoice_project)
    _change_config(invoice_project)

    output = _adopt(invoice_project, "--select", "raw_invoices", "--dry-run")

    assert "would restamp 5 rows" in output
    assert _raw(invoice_project).stale_rows == 5
    with _warehouse(invoice_project) as adapter:
        assert adapter.table_column_names("stel_adoption_log") is None


def test_adopting_a_current_model_says_so(invoice_project: Path) -> None:
    run_project(invoice_project)

    output = _adopt(invoice_project, "--select", "raw_invoices")

    assert "nothing to adopt" in output
    with _warehouse(invoice_project) as adapter:
        assert adapter.table_column_names("stel_adoption_log") is None


def test_adopt_requires_a_selection(invoice_project: Path) -> None:
    """Adopting is a claim about named models, never a project-wide default."""
    result = CliRunner().invoke(cli, ["--project-dir", str(invoice_project), "adopt"])

    assert result.exit_code != 0
    assert "--select" in result.output


def test_adopt_leaves_other_scopes_alone(invoice_project: Path) -> None:
    run_project(invoice_project)
    _change_config(invoice_project)
    with _warehouse(invoice_project) as adapter:
        other = StateScope("someone_else")
        adapter.upsert_state(other, [StateRecord("k", "fp", "stale-version")])

    _adopt(invoice_project, "--select", "raw_invoices")

    with _warehouse(invoice_project) as adapter:
        assert adapter.state_code_version_counts(other) == {"stale-version": 1}
