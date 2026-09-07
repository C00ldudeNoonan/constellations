"""`stel ls --orphans` lists what an experiment leaves behind (issue #531)."""
from __future__ import annotations

import json
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
from stel.orphans import find_orphans, format_orphans
from stel.runner import run_project
from stel.synth import generate_invoices

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def invoice_project(tmp_path: Path) -> Path:
    dst = tmp_path / "invoice_pipeline"
    shutil.copytree(
        EXAMPLES / "invoice_pipeline",
        dst,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    generate_invoices(5, dst / "data" / "invoices", seed=1)
    return dst


def _warehouse(project_dir: Path) -> WarehouseAdapter:
    config = parse_warehouse_config(
        {"type": "duckdb", "path": str(project_dir / "target" / "stel.duckdb")}
    )
    return create_adapter(config, project_dir=project_dir)


def _names(project_dir: Path) -> tuple[list[str], list[str]]:
    report = find_orphans(project_dir, target=None, profiles_dir=None)
    return (
        [table.name for table in report.tables],
        [scope.name for scope in report.state_scopes],
    )


# ─── the adapter read ───────────────────────────────────────────────────────


def test_adapter_lists_every_scope_with_size_and_activity(tmp_path: Path) -> None:
    config = parse_warehouse_config({"type": "duckdb", "path": str(tmp_path / "w.duckdb")})
    with create_adapter(config, project_dir=tmp_path) as adapter:
        assert adapter.list_state_scopes() == []
        adapter.upsert_state(
            StateScope("docs"),
            [StateRecord("a", "fp", "v1"), StateRecord("b", "fp", "v2")],
        )
        adapter.upsert_state(StateScope("other"), [StateRecord("z", "fp", "v9")])
        summaries = adapter.list_state_scopes()
    by_name = {summary.scope.model_name: summary for summary in summaries}
    assert set(by_name) == {"docs", "other"}
    assert (by_name["docs"].rows, by_name["docs"].code_versions) == (2, 2)
    assert (by_name["other"].rows, by_name["other"].code_versions) == (1, 1)
    assert by_name["docs"].last_run_at is not None
    assert by_name["docs"].scope.stage == "materialization"


# ─── what counts as an orphan ───────────────────────────────────────────────


def test_a_fully_claimed_project_has_no_orphans(invoice_project: Path) -> None:
    run_project(invoice_project)
    report = find_orphans(invoice_project, target=None, profiles_dir=None)
    assert report.is_empty()
    assert report.schema
    assert "No orphans" in format_orphans(report)[0]


def test_a_deleted_model_file_orphans_its_table(invoice_project: Path) -> None:
    run_project(invoice_project)
    (invoice_project / "models" / "monthly_totals.yml").unlink()
    tables, scopes = _names(invoice_project)
    assert tables == ["monthly_totals"]
    # A full model keeps no incremental state, so nothing is stranded there.
    assert scopes == []


def test_a_renamed_model_orphans_its_table_and_state(invoice_project: Path) -> None:
    run_project(invoice_project)
    for filename in ("raw_invoices.yml", "invoice_summary.yml", "monthly_totals.yml"):
        path = invoice_project / "models" / filename
        path.write_text(
            path.read_text(encoding="utf-8").replace("raw_invoices", "raw_invoices_v2"),
            encoding="utf-8",
        )
    tables, scopes = _names(invoice_project)
    assert tables == ["raw_invoices"]
    assert scopes == ["raw_invoices"]
    report = find_orphans(invoice_project, target=None, profiles_dir=None)
    stranded = report.state_scopes[0]
    assert stranded.summary.rows == 5
    assert stranded.summary.code_versions == 1


def test_a_stranded_scope_with_no_table_is_still_listed(invoice_project: Path) -> None:
    """State can outlive its table (an interrupted first run, a manual drop);
    the scope alone is still something to know about."""
    run_project(invoice_project)
    with _warehouse(invoice_project) as adapter:
        adapter.upsert_state(StateScope("ghost"), [StateRecord("k", "fp", "v1")])
    tables, scopes = _names(invoice_project)
    assert tables == []
    assert scopes == ["ghost"]


def test_dropping_a_for_each_axis_value_orphans_that_variant(tmp_path: Path) -> None:
    """The experiment-ending case: two variants ran, one axis value goes away."""
    project = tmp_path / "variants"
    shutil.copytree(
        EXAMPLES / "invoice_pipeline",
        project,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    generate_invoices(3, project / "data" / "invoices", seed=1)
    # The downstream models reference the template name, which a for_each
    # expansion removes; the experiment is on the extraction step alone.
    (project / "models" / "invoice_summary.yml").unlink()
    (project / "models" / "monthly_totals.yml").unlink()
    model = project / "models" / "raw_invoices.yml"
    text = model.read_text(encoding="utf-8")
    assert text.count("    source: ref('vendor_invoices')\n") == 1
    model.write_text(
        text.replace(
            "    source: ref('vendor_invoices')\n",
            "    source: ref('vendor_invoices')\n    for_each:\n      variant: [a, b]\n",
        ),
        encoding="utf-8",
    )
    results = run_project(project)
    assert sorted(r.model_name for r in results) == [
        "raw_invoices__variant_a",
        "raw_invoices__variant_b",
    ]
    assert find_orphans(project, target=None, profiles_dir=None).is_empty()

    model.write_text(
        model.read_text(encoding="utf-8").replace("variant: [a, b]", "variant: [a]"),
        encoding="utf-8",
    )
    tables, scopes = _names(project)
    assert tables == ["raw_invoices__variant_b"]
    assert scopes == ["raw_invoices__variant_b"]


def test_internal_tables_are_never_orphans(invoice_project: Path) -> None:
    run_project(invoice_project)
    with _warehouse(invoice_project) as adapter:
        # The state table itself, plus a test-failure and a staging table
        # from a crashed run: stel's own, never reported.
        adapter.execute(
            f"CREATE TABLE {adapter.schema_ref}.stel_test_failures__x AS SELECT 1 AS a"
        )
        adapter.execute(
            f"CREATE TABLE {adapter.schema_ref}.stel_staging__y AS SELECT 1 AS a"
        )
    assert find_orphans(invoice_project, target=None, profiles_dir=None).is_empty()


# ─── the CLI ────────────────────────────────────────────────────────────────


def test_cli_lists_orphans_in_both_forms(invoice_project: Path) -> None:
    run_project(invoice_project)
    (invoice_project / "models" / "monthly_totals.yml").unlink()
    runner = CliRunner()
    project = ["--project-dir", str(invoice_project)]

    result = runner.invoke(cli, [*project, "ls", "--orphans"])
    assert result.exit_code == 0, result.output
    assert "monthly_totals" in result.output
    assert "Nothing was removed" in result.output

    result = runner.invoke(cli, [*project, "ls", "--orphans", "--output", "json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [t["name"] for t in payload["tables"]] == ["monthly_totals"]
    assert payload["state_scopes"] == []
    assert payload["schema"]


def test_cli_rejects_selectors_with_orphans(invoice_project: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--project-dir", str(invoice_project), "ls", "--orphans", "--select", "x"]
    )
    assert result.exit_code == 2
    assert "do not apply" in result.output


def test_plain_ls_is_unchanged(invoice_project: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--project-dir", str(invoice_project), "ls"])
    assert result.exit_code == 0, result.output
    assert "raw_invoices" in result.output
