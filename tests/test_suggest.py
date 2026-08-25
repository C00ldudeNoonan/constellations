"""Context suggestions rendered as a reviewable diff (issue #361)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from stel.suggest import (
    SuggestionError,
    SuggestionRow,
    plan_suggestions,
    render_suggestion,
    schema_files,
)

SCHEMA = """version: 2

models:
  # Revenue rollup. Owned by finance.
  - name: fct_orders
    columns:
      - name: order_id
        tests:
          - unique
      - name: amount
        description: "Gross amount in cents"
  - name: dim_customers
    description: "One row per customer"
"""


def _row(
    *,
    dbt_model: str = "fct_orders",
    suggested_description: str = "One row per order, excluding cancellations",
    evidence_count: int = 4,
    dbt_column: str | None = None,
) -> SuggestionRow:
    return SuggestionRow(
        dbt_model=dbt_model,
        suggested_description=suggested_description,
        evidence_count=evidence_count,
        evidence_sessions=("s1", "s2", "s3", "s4"),
        dbt_column=dbt_column,
    )


def _dbt_project(tmp_path: Path, schema: str = SCHEMA) -> Path:
    project = tmp_path / "dbt"
    (project / "models" / "marts").mkdir(parents=True)
    (project / "models" / "marts" / "schema.yml").write_text(schema, encoding="utf-8")
    return project


# ─── the contract ───────────────────────────────────────────────────────────


def test_a_suggestion_without_provenance_is_still_rejected_on_its_other_fields() -> None:
    with pytest.raises(SuggestionError, match="suggested_description"):
        _row(suggested_description="   ")
    with pytest.raises(SuggestionError, match="evidence_count"):
        _row(evidence_count=0)


def test_model_and_column_names_are_validated_not_trusted() -> None:
    """These cross into a YAML key and a file lookup, so a relation cannot
    smuggle a path or a structural character through them."""
    for bad in ("../../etc/passwd", "a b", "x:y", "-leading"):
        with pytest.raises((SuggestionError, ValueError)):
            _row(dbt_model=bad)


# ─── what it refuses to do ──────────────────────────────────────────────────


def test_an_existing_description_is_never_overwritten(tmp_path: Path) -> None:
    """The suggestion can add context. It must never replace a human's."""
    project = _dbt_project(tmp_path)
    pending, outcomes = plan_suggestions(
        [_row(dbt_model="dim_customers")], project, min_evidence=1
    )

    assert pending == {}
    assert outcomes[0].applied is False
    assert outcomes[0].reason == "already documented"
    assert "One row per customer" in (
        project / "models" / "marts" / "schema.yml"
    ).read_text(encoding="utf-8")


def test_a_column_that_already_has_a_description_is_left_alone(tmp_path: Path) -> None:
    project = _dbt_project(tmp_path)
    pending, outcomes = plan_suggestions(
        [_row(dbt_column="amount")], project, min_evidence=1
    )
    assert pending == {}
    assert outcomes[0].reason == "already documented"


def test_thin_evidence_is_not_worth_a_pull_request(tmp_path: Path) -> None:
    """One session is an anecdote. The threshold is what stops the loop from
    proposing a change every time an agent happens to open a file."""
    project = _dbt_project(tmp_path)
    pending, outcomes = plan_suggestions([_row(evidence_count=1)], project)

    assert pending == {}
    assert "below the evidence threshold" in outcomes[0].reason


def test_a_model_the_dbt_project_does_not_declare_is_reported_not_invented(
    tmp_path: Path,
) -> None:
    project = _dbt_project(tmp_path)
    pending, outcomes = plan_suggestions(
        [_row(dbt_model="not_a_model")], project, min_evidence=1
    )
    assert pending == {}
    assert outcomes[0].reason == "model not declared in any models/**/*.yml"


def test_a_directory_without_models_is_refused_as_not_a_dbt_project(
    tmp_path: Path,
) -> None:
    with pytest.raises(SuggestionError, match="does not look like a dbt project"):
        plan_suggestions([_row()], tmp_path)


def test_only_models_yml_is_ever_in_reach(tmp_path: Path) -> None:
    """Widening the search would put dbt_project.yml and seeds in reach of an
    automated edit."""
    project = _dbt_project(tmp_path)
    (project / "dbt_project.yml").write_text("name: x\n", encoding="utf-8")
    (project / "seeds").mkdir()
    (project / "seeds" / "schema.yml").write_text("version: 2\n", encoding="utf-8")

    found = schema_files(project)
    assert [path.name for path in found] == ["schema.yml"]
    assert found[0].parent.name == "marts"


# ─── what it does ───────────────────────────────────────────────────────────


def test_a_missing_model_description_is_filled_in_place(tmp_path: Path) -> None:
    project = _dbt_project(tmp_path)
    pending, outcomes = plan_suggestions([_row()], project, min_evidence=1)

    assert outcomes[0].applied is True
    (updated,) = pending.values()
    document = yaml.safe_load(updated)
    entry = next(m for m in document["models"] if m["name"] == "fct_orders")
    assert entry["description"] == "One row per order, excluding cancellations"
    # The rest of the file is untouched, comment included — a proposal buried
    # in a reformatted file is not reviewable.
    assert "# Revenue rollup. Owned by finance." in updated
    assert updated.count("- name: order_id") == 1


def test_a_missing_column_description_is_filled_under_its_column(
    tmp_path: Path,
) -> None:
    project = _dbt_project(tmp_path)
    pending, _ = plan_suggestions(
        [_row(dbt_column="order_id", suggested_description="Natural key")],
        project,
        min_evidence=1,
    )

    (updated,) = pending.values()
    document = yaml.safe_load(updated)
    entry = next(m for m in document["models"] if m["name"] == "fct_orders")
    column = next(c for c in entry["columns"] if c["name"] == "order_id")
    assert column["description"] == "Natural key"
    # The sibling column keeps its own description, and its tests survive.
    assert entry["columns"][0]["tests"] == ["unique"]


def test_prose_that_looks_like_yaml_cannot_restructure_the_document(
    tmp_path: Path,
) -> None:
    """A description is ordinary prose and may contain colons and quotes. It
    must not be able to introduce keys into the file it lands in."""
    project = _dbt_project(tmp_path)
    hostile = 'Totals: "gross", not net\n  tests:\n    - unique'
    pending, _ = plan_suggestions(
        [_row(suggested_description=hostile)], project, min_evidence=1
    )

    (updated,) = pending.values()
    document = yaml.safe_load(updated)
    entry = next(m for m in document["models"] if m["name"] == "fct_orders")
    assert entry["description"] == hostile
    assert "tests" not in entry


def test_two_suggestions_against_one_file_both_land(tmp_path: Path) -> None:
    """Each edit is planned against the previous one's result, not the file on
    disk, so the second does not silently drop the first."""
    project = _dbt_project(tmp_path)
    pending, outcomes = plan_suggestions(
        [
            _row(),
            _row(dbt_column="order_id", suggested_description="Natural key"),
        ],
        project,
        min_evidence=1,
    )

    assert [outcome.applied for outcome in outcomes] == [True, True]
    (updated,) = pending.values()
    entry = next(
        m for m in yaml.safe_load(updated)["models"] if m["name"] == "fct_orders"
    )
    assert entry["description"]
    assert entry["columns"][0]["description"] == "Natural key"


def test_planning_writes_nothing(tmp_path: Path) -> None:
    """The artifact is a patch. Applying it is a separate, explicit decision."""
    project = _dbt_project(tmp_path)
    before = (project / "models" / "marts" / "schema.yml").read_text(encoding="utf-8")

    plan_suggestions([_row()], project, min_evidence=1)

    assert (project / "models" / "marts" / "schema.yml").read_text(
        encoding="utf-8"
    ) == before


def test_a_flow_mapping_entry_is_refused_rather_than_guessed_at(
    tmp_path: Path,
) -> None:
    project = _dbt_project(
        tmp_path, "version: 2\nmodels: [{name: fct_orders}]\n"
    )
    updated, reason = render_suggestion(
        (project / "models" / "marts" / "schema.yml").read_text(encoding="utf-8"),
        _row(),
    )
    assert updated is None
    assert "by hand" in reason


# ─── the command edge ───────────────────────────────────────────────────────


def test_the_cli_prints_a_diff_and_writes_nothing_without_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`stel suggest dbt` hands back a patch. Applying it stays a decision."""
    from click.testing import CliRunner

    from stel.cli import cli
    from stel.cli_services import suggest as suggest_service

    project = _dbt_project(tmp_path)
    before = (project / "models" / "marts" / "schema.yml").read_text(encoding="utf-8")
    monkeypatch.setattr(
        suggest_service, "read_suggestions", lambda *a, **k: [_row()]
    )

    result = CliRunner().invoke(
        cli,
        [
            "suggest",
            "dbt",
            "--from",
            "analytics.suggestions",
            "--dbt-project",
            str(project),
            "--min-evidence",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "+    description:" in result.output
    assert "Re-run with --write to apply." in result.output
    assert (project / "models" / "marts" / "schema.yml").read_text(
        encoding="utf-8"
    ) == before


def test_the_cli_applies_only_with_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from click.testing import CliRunner

    from stel.cli import cli
    from stel.cli_services import suggest as suggest_service

    project = _dbt_project(tmp_path)
    monkeypatch.setattr(
        suggest_service, "read_suggestions", lambda *a, **k: [_row()]
    )

    result = CliRunner().invoke(
        cli,
        [
            "suggest",
            "dbt",
            "--from",
            "analytics.suggestions",
            "--dbt-project",
            str(project),
            "--min-evidence",
            "1",
            "--write",
        ],
    )

    assert result.exit_code == 0, result.output
    entry = next(
        m
        for m in yaml.safe_load(
            (project / "models" / "marts" / "schema.yml").read_text(encoding="utf-8")
        )["models"]
        if m["name"] == "fct_orders"
    )
    assert entry["description"] == "One row per order, excluding cancellations"
