"""`stel migrate` and the guards that make it mandatory (#313).

The rename moved stel's persisted warehouse objects from `dbt_ml_*` to
`stel_*`. Every failure mode here has the same shape and the same cost: from
inside a run, a warehouse whose state table is under the old name is
indistinguishable from a brand-new project, so the run reports every document
as new and reprocesses the corpus at provider cost — green, silent, and
expensive. So the tests below check two things in pairs: that the migration
carries the rows over, and that nothing is allowed to proceed until it has.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl
import pytest
from click.testing import CliRunner

from stel.adapters import (
    LegacyWarehouseNamesError,
    MigrationConflictError,
    StateRecord,
    StateScope,
    apply_name_migration,
    create_adapter,
    parse_warehouse_config,
    plan_name_migration,
)
from stel.adapters.base import (
    LEGACY_SERVING_LEASE_TABLE,
    LEGACY_SERVING_LEDGER_TABLE,
    LEGACY_STATE_TABLE,
    LEGACY_TEST_FAILURES_TABLE_PREFIX,
    SERVING_LEASE_TABLE,
    SERVING_LEDGER_TABLE,
    STATE_TABLE,
    TEST_FAILURES_TABLE_PREFIX,
    WarehouseAdapter,
)
from stel.cli import cli
from stel.config import ConfigError
from stel.config.identifiers import (
    DEFAULT_SCHEMA_NAME,
    LEGACY_DUCKDB_FILENAME,
    LEGACY_SCHEMA_NAME,
)
from stel.config.loader import load_project
from stel.config.profile import WarehouseConfig
from stel.runner import run_project

# ─── helpers ────────────────────────────────────────────────────────────────


def _config(path: Path, schema: str | None = None) -> WarehouseConfig:
    """Omitting `schema` is the point of several tests: it is what leaves the
    default unset, which is what the legacy-schema guard keys on."""
    raw: dict[str, object] = {"type": "duckdb", "path": str(path)}
    if schema is not None:
        raw["schema"] = schema
    return parse_warehouse_config(raw)


def _open(path: Path, schema: str | None = None) -> WarehouseAdapter:
    """Connect past the guards, the way `stel migrate` does."""
    adapter = create_adapter(_config(path, schema))
    adapter.migration_mode = True
    return adapter


def _seed_legacy_warehouse(path: Path, schema: str = LEGACY_SCHEMA_NAME) -> None:
    """Build the warehouse a pre-#313 stel would have left behind.

    The state table is written through the state API and then rewound to its
    old name, so it carries the real v2 shape the adapter validates on connect
    rather than a stand-in that would fail for the wrong reason.
    """
    with create_adapter(_config(path, schema)) as adapter:
        adapter.upsert_state(
            StateScope("raw_invoices"), [StateRecord("doc-1", "fp-1", "code-v1")]
        )
        for table in (
            LEGACY_SERVING_LEDGER_TABLE,
            LEGACY_SERVING_LEASE_TABLE,
            LEGACY_TEST_FAILURES_TABLE_PREFIX + "items__not_null__total",
            "raw_invoices",
        ):
            adapter.materialize_full(table, pl.DataFrame({"marker": [table]}))
    with _open(path, schema) as adapter:
        adapter.rename_table(STATE_TABLE, LEGACY_STATE_TABLE)


def _marker(adapter: WarehouseAdapter, table: str) -> str:
    return str(adapter.read_table(table, limit=1)["marker"][0])


# ─── planning ───────────────────────────────────────────────────────────────


def test_plan_covers_every_persisted_table_and_the_failure_prefix(
    tmp_path: Path,
) -> None:
    db = tmp_path / "w.duckdb"
    _seed_legacy_warehouse(db)
    with _open(db, LEGACY_SCHEMA_NAME) as adapter:
        planned = {(r.old, r.new) for r in plan_name_migration(adapter)}
    assert planned == {
        (LEGACY_STATE_TABLE, STATE_TABLE),
        (LEGACY_SERVING_LEDGER_TABLE, SERVING_LEDGER_TABLE),
        (LEGACY_SERVING_LEASE_TABLE, SERVING_LEASE_TABLE),
        (
            LEGACY_TEST_FAILURES_TABLE_PREFIX + "items__not_null__total",
            TEST_FAILURES_TABLE_PREFIX + "items__not_null__total",
        ),
    }


def test_plan_leaves_user_models_and_staging_debris_alone(tmp_path: Path) -> None:
    """Staging tables are in-flight debris from a crashed run, not state worth
    carrying; models are not ours to touch at all."""
    db = tmp_path / "w.duckdb"
    _seed_legacy_warehouse(db)
    with _open(db, LEGACY_SCHEMA_NAME) as adapter:
        adapter.materialize_full("dbt_ml_staging__x__abc", pl.DataFrame({"a": [1]}))
        touched = {r.old for r in plan_name_migration(adapter)}
    assert "raw_invoices" not in touched
    assert "dbt_ml_staging__x__abc" not in touched


def test_plan_is_empty_on_an_already_migrated_warehouse(tmp_path: Path) -> None:
    db = tmp_path / "w.duckdb"
    _seed_legacy_warehouse(db)
    with _open(db, LEGACY_SCHEMA_NAME) as adapter:
        apply_name_migration(adapter, plan_name_migration(adapter))
        assert plan_name_migration(adapter) == []


def test_both_spellings_present_refuses_to_choose(tmp_path: Path) -> None:
    db = tmp_path / "w.duckdb"
    _seed_legacy_warehouse(db)
    with _open(db, LEGACY_SCHEMA_NAME) as adapter:
        adapter.materialize_full(
            SERVING_LEDGER_TABLE, pl.DataFrame({"marker": ["new"]})
        )
        with pytest.raises(MigrationConflictError) as excinfo:
            plan_name_migration(adapter)
    message = str(excinfo.value)
    assert LEGACY_SERVING_LEDGER_TABLE in message and SERVING_LEDGER_TABLE in message
    # An interrupted migration is recoverable only if nothing was destroyed.
    with _open(db, LEGACY_SCHEMA_NAME) as adapter:
        assert _marker(adapter, LEGACY_SERVING_LEDGER_TABLE) == (
            LEGACY_SERVING_LEDGER_TABLE
        )
        assert _marker(adapter, SERVING_LEDGER_TABLE) == "new"


# ─── applying ───────────────────────────────────────────────────────────────


def test_migration_moves_the_rows_rather_than_recreating_the_table(
    tmp_path: Path,
) -> None:
    db = tmp_path / "w.duckdb"
    with _open(db, "sch") as adapter:
        adapter._ensure_schema()
        adapter.materialize_full(
            LEGACY_STATE_TABLE,
            pl.DataFrame({"model_name": ["a", "b"], "record_key": ["1", "2"]}),
        )
    with _open(db, "sch") as adapter:
        apply_name_migration(adapter, plan_name_migration(adapter))
    with _open(db, "sch") as adapter:
        present = set(adapter.list_all_tables())
        assert STATE_TABLE in present
        assert LEGACY_STATE_TABLE not in present
        assert sorted(adapter.read_table(STATE_TABLE, limit=10)["model_name"]) == [
            "a",
            "b",
        ]


def test_incremental_state_survives_the_migration(tmp_path: Path) -> None:
    """The whole point: the fingerprints written before the rename must still
    be readable after it, or every model reprocesses."""
    db = tmp_path / "w.duckdb"
    scope = StateScope("raw_invoices")
    with create_adapter(_config(db, "sch")) as adapter:
        adapter.upsert_state(scope, [StateRecord("doc-1", "fp-1", "code-v1")])
        before = adapter.fetch_state(scope)
    # Put that state back under its pre-#313 name.
    with _open(db, "sch") as adapter:
        adapter.rename_table(STATE_TABLE, LEGACY_STATE_TABLE)

    with _open(db, "sch") as adapter:
        apply_name_migration(adapter, plan_name_migration(adapter))
    with create_adapter(_config(db, "sch")) as adapter:
        assert adapter.fetch_state(scope) == before


# ─── the guards ─────────────────────────────────────────────────────────────


def test_connecting_to_a_legacy_state_table_refuses_and_names_the_command(
    tmp_path: Path,
) -> None:
    db = tmp_path / "w.duckdb"
    _seed_legacy_warehouse(db, "sch")
    with pytest.raises(LegacyWarehouseNamesError) as excinfo:
        with create_adapter(_config(db, "sch")):
            pass
    message = str(excinfo.value)
    assert LEGACY_STATE_TABLE in message
    assert "stel migrate" in message


def test_the_refusal_creates_nothing(tmp_path: Path) -> None:
    """A guard that half-initialized the schema on its way out would leave the
    conflict it just refused to resolve."""
    db = tmp_path / "w.duckdb"
    _seed_legacy_warehouse(db, "sch")
    with pytest.raises(LegacyWarehouseNamesError):
        with create_adapter(_config(db, "sch")):
            pass
    with _open(db, "sch") as adapter:
        assert STATE_TABLE not in set(adapter.list_all_tables())


def test_migration_clears_the_guard(tmp_path: Path) -> None:
    db = tmp_path / "w.duckdb"
    _seed_legacy_warehouse(db, "sch")
    with _open(db, "sch") as adapter:
        apply_name_migration(adapter, plan_name_migration(adapter))
    with create_adapter(_config(db, "sch")) as adapter:
        assert adapter.list_tables() == ["raw_invoices"]
        assert adapter.fetch_state(StateScope("raw_invoices")).keys() == {"doc-1"}


def test_defaulted_schema_beside_a_populated_legacy_one_refuses(tmp_path: Path) -> None:
    db = tmp_path / "w.duckdb"
    _seed_legacy_warehouse(db, LEGACY_SCHEMA_NAME)
    with pytest.raises(LegacyWarehouseNamesError) as excinfo:
        with create_adapter(_config(db)):  # no `schema:`, so the new default
            pass
    message = str(excinfo.value)
    assert f"schema: {LEGACY_SCHEMA_NAME}" in message
    assert DEFAULT_SCHEMA_NAME in message


def test_an_explicit_schema_is_never_second_guessed(tmp_path: Path) -> None:
    """Someone who wrote `schema:` chose where their data lives, so the changed
    default is not a surprise to them — even when they chose the new default's
    own value."""
    db = tmp_path / "w.duckdb"
    _seed_legacy_warehouse(db, LEGACY_SCHEMA_NAME)
    with create_adapter(_config(db, DEFAULT_SCHEMA_NAME)) as adapter:
        assert adapter.list_tables() == []


def test_a_genuinely_fresh_warehouse_is_not_blocked(tmp_path: Path) -> None:
    with create_adapter(_config(tmp_path / "fresh.duckdb")) as adapter:
        assert adapter.list_tables() == []


def test_the_legacy_schema_guard_ignores_a_schema_holding_no_stel_objects(
    tmp_path: Path,
) -> None:
    """A `dbt_ml` schema someone else built is not evidence of a pre-rename
    stel project, and blocking on it would be a false positive with no fix."""
    db = tmp_path / "w.duckdb"
    with _open(db, LEGACY_SCHEMA_NAME) as adapter:
        adapter._ensure_schema()
        adapter.materialize_full("someone_elses_table", pl.DataFrame({"a": [1]}))
    with create_adapter(_config(db)) as adapter:
        assert adapter.list_tables() == []


# ─── the inline zero-config path ────────────────────────────────────────────


def _inline_project(tmp_path: Path) -> Path:
    project_dir = tmp_path / "proj"
    (project_dir / "target").mkdir(parents=True)
    (project_dir / "sources").mkdir()
    (project_dir / "models").mkdir()
    (project_dir / "stel_project.yml").write_text(
        "name: inline_project\nversion: 0.1.0\n", encoding="utf-8"
    )
    return project_dir


def test_inline_project_refuses_to_open_a_new_database_beside_the_legacy_file(
    tmp_path: Path,
) -> None:
    project_dir = _inline_project(tmp_path)
    (project_dir / "target" / LEGACY_DUCKDB_FILENAME).write_text("", encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        load_project(project_dir)
    assert LEGACY_DUCKDB_FILENAME in str(excinfo.value)
    assert "path:" in str(excinfo.value)


def test_inline_project_with_no_legacy_file_loads(tmp_path: Path) -> None:
    project, _, _ = load_project(_inline_project(tmp_path))
    assert project.duckdb.path.name != LEGACY_DUCKDB_FILENAME


def test_an_explicit_inline_path_is_never_second_guessed(tmp_path: Path) -> None:
    project_dir = _inline_project(tmp_path)
    (project_dir / "target" / LEGACY_DUCKDB_FILENAME).write_text("", encoding="utf-8")
    (project_dir / "stel_project.yml").write_text(
        "name: inline_project\nversion: 0.1.0\n"
        f"duckdb:\n  path: ./target/{LEGACY_DUCKDB_FILENAME}\n",
        encoding="utf-8",
    )
    project, _, _ = load_project(project_dir)
    assert project.duckdb.path.name == LEGACY_DUCKDB_FILENAME


# ─── the command ────────────────────────────────────────────────────────────


def _example_on_a_legacy_warehouse(tmp_path: Path, example_project_dir: Path) -> Path:
    """A real example project, run to completion, then rewound to the names a
    pre-#313 stel would have written."""
    from stel.synth import generate_invoices

    project_dir = tmp_path / "project"
    shutil.copytree(
        example_project_dir,
        project_dir,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    generate_invoices(5, project_dir / "data" / "invoices", seed=1)
    run_project(project_dir)

    # Only the table names are rewound. The example pins `schema:` the way
    # every shipped example does, so the schema guard is not what is under
    # test here — the state-table guard is.
    db = project_dir / "target" / "stel.duckdb"
    with _open(db, DEFAULT_SCHEMA_NAME) as adapter:
        adapter.rename_table(STATE_TABLE, LEGACY_STATE_TABLE)
    return project_dir


def test_dry_run_reports_the_plan_and_changes_nothing(
    tmp_path: Path, example_project_dir: Path
) -> None:
    project_dir = _example_on_a_legacy_warehouse(tmp_path, example_project_dir)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--project-dir", str(project_dir), "migrate", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert f"{LEGACY_STATE_TABLE} -> {STATE_TABLE}" in result.output

    db = project_dir / "target" / "stel.duckdb"
    with _open(db, DEFAULT_SCHEMA_NAME) as adapter:
        assert LEGACY_STATE_TABLE in set(adapter.list_all_tables())


def test_migrate_then_rerun_skips_instead_of_reprocessing(
    tmp_path: Path, example_project_dir: Path
) -> None:
    """The rehearsal the whole phase exists for: after migrating, a re-run must
    report the models as up to date rather than rebuilding them."""
    project_dir = _example_on_a_legacy_warehouse(tmp_path, example_project_dir)
    runner = CliRunner()

    blocked = runner.invoke(cli, ["--project-dir", str(project_dir), "run"])
    assert blocked.exit_code != 0
    assert "stel migrate" in blocked.output

    migrated = runner.invoke(cli, ["--project-dir", str(project_dir), "migrate"])
    assert migrated.exit_code == 0, migrated.output

    results = {r.model_name: r for r in run_project(project_dir)}
    assert results, "expected the example project to have models"
    extraction = results["raw_invoices"]
    assert not extraction.errors, extraction.errors
    # The fingerprints written before the rename are still being read: every
    # document matches, so none is processed again.
    assert extraction.documents_processed == 0, results
    assert extraction.documents_skipped > 0, results


def test_migrate_on_a_current_warehouse_says_so(
    tmp_path: Path, example_project_dir: Path
) -> None:
    project_dir = tmp_path / "project"
    shutil.copytree(
        example_project_dir,
        project_dir,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    result = CliRunner().invoke(cli, ["--project-dir", str(project_dir), "migrate"])
    assert result.exit_code == 0, result.output
    assert "Nothing to migrate" in result.output
