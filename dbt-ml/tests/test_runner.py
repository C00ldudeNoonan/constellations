from __future__ import annotations

import json
import shutil
from pathlib import Path

import duckdb
import pytest

from dbt_ml.config import ConfigError
from dbt_ml.config.source import SourceConfig
from dbt_ml.manifest import write_run_results
from dbt_ml.runner import RunError, build_project, clean_project, run_project
from dbt_ml.sources import LocalDocumentSource
from dbt_ml.synth import generate_invoices, generate_support_tickets
from dbt_ml.versioning import compute_document_id


def test_build_runs_and_tests_in_order(fresh_project: Path) -> None:
    generate_invoices(10, fresh_project / "data" / "invoices", seed=1)

    result = build_project(fresh_project)
    run_names = {r.model_name for r in result.run_results}
    assert run_names == {"raw_invoices", "invoice_summary", "monthly_totals"}
    assert result.skipped == []
    assert all(t.status == "pass" for t in result.test_results)
    assert any(t.model_name == "raw_invoices" for t in result.test_results)


def test_build_skips_downstream_on_test_failure(fresh_project: Path) -> None:
    generate_invoices(5, fresh_project / "data" / "invoices", seed=1)
    raw_yml = fresh_project / "models" / "raw_invoices.yml"
    raw_yml.write_text(
        raw_yml.read_text().replace(
            "tests:", "tests:\n      - min_rows: 100000", 1
        )
    )

    result = build_project(fresh_project)
    assert "raw_invoices" in {r.model_name for r in result.run_results}
    assert set(result.skipped) == {"invoice_summary", "monthly_totals"}
    assert any(
        t.model_name == "raw_invoices" and t.status == "fail"
        for t in result.test_results
    )
    # downstream models never ran
    assert "invoice_summary" not in {r.model_name for r in result.run_results}


@pytest.fixture
def fresh_project(tmp_path: Path, example_project_dir: Path) -> Path:
    """Copy the example project into a tmp dir so each test gets a clean slate."""
    dst = tmp_path / "project"
    shutil.copytree(
        example_project_dir,
        dst,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    return dst


def _query(db_path: Path, sql: str) -> list[tuple]:
    con = duckdb.connect(str(db_path))
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def _write_ticket(
    path: Path,
    ticket_id: str,
    summary: str,
    priority: str = "medium",
) -> None:
    path.write_text(
        json.dumps(
            {
                "ticket_id": ticket_id,
                "summary": summary,
                "priority": priority,
            }
        )
    )


def test_end_to_end_run(fresh_project: Path) -> None:
    invoices_dir = fresh_project / "data" / "invoices"
    generate_invoices(10, invoices_dir, seed=1)

    results = run_project(fresh_project)
    by_name = {r.model_name: r for r in results}
    assert by_name["raw_invoices"].documents_processed == 10
    assert by_name["raw_invoices"].documents_skipped == 0
    assert by_name["raw_invoices"].rows_written == 10
    assert by_name["invoice_summary"].kind == "transform"

    db = fresh_project / "target" / "dbt_ml.duckdb"
    assert db.exists()
    rows = _query(db, 'SELECT COUNT(*) FROM "dbt_ml".dbt_ml.raw_invoices')
    assert rows[0][0] == 10


def test_warehouse_options_portable_and_state_preserving(fresh_project: Path) -> None:
    """BigQuery-shaped warehouse_options on a DuckDB target are ignored
    (dev/prod portability, issue #91) and never invalidate incremental state."""
    generate_invoices(5, fresh_project / "data" / "invoices", seed=1)
    run_project(fresh_project, select="raw_invoices")

    raw_yml = fresh_project / "models" / "raw_invoices.yml"
    updated = raw_yml.read_text().replace(
        "    materialization: incremental",
        "    materialization: incremental\n"
        "    warehouse_options:\n"
        "      partition_by:\n"
        "        field: issue_date\n"
        "      cluster_by: [vendor]",
        1,
    )
    assert "warehouse_options" in updated
    raw_yml.write_text(updated)

    (result,) = run_project(fresh_project, select="raw_invoices")
    assert result.errors == []
    assert result.documents_processed == 0
    assert result.documents_skipped == 5


def test_second_run_is_incremental(fresh_project: Path) -> None:
    invoices_dir = fresh_project / "data" / "invoices"
    generate_invoices(5, invoices_dir, seed=1)
    run_project(fresh_project)

    results = run_project(fresh_project)
    raw = next(r for r in results if r.model_name == "raw_invoices")
    assert raw.documents_processed == 0
    assert raw.documents_skipped == 5


def test_default_backend_change_reprocesses_incremental_documents(
    tmp_path: Path,
) -> None:
    (tmp_path / "dbt_ml_project.yml").write_text(
        "name: effective_backend\n"
        "extraction:\n  default_backend: json\n"
    )
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "docs.yml").write_text(
        "version: 2\nsources:\n"
        "  - name: docs\n    path: data/docs\n    file_pattern: '*.json'\n"
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "raw.yml").write_text(
        "version: 2\nmodels:\n"
        "  - name: raw_docs\n    source: ref('docs')\n"
        "    extraction: {}\n    materialization: incremental\n"
        "    on_schema_change: append_new_columns\n"
    )
    data_dir = tmp_path / "data" / "docs"
    data_dir.mkdir(parents=True)
    (data_dir / "doc.json").write_text("{}")

    first = run_project(tmp_path)
    first_raw = next(result for result in first if result.model_name == "raw_docs")
    assert first_raw.documents_processed == 1

    project_file = tmp_path / "dbt_ml_project.yml"
    project_file.write_text(
        project_file.read_text().replace("default_backend: json", "default_backend: markdown")
    )
    second = run_project(tmp_path)
    second_raw = next(result for result in second if result.model_name == "raw_docs")

    assert second_raw.documents_processed == 1
    assert second_raw.documents_skipped == 0


def test_changed_doc_is_reprocessed(fresh_project: Path) -> None:
    invoices_dir = fresh_project / "data" / "invoices"
    generate_invoices(5, invoices_dir, seed=1)
    run_project(fresh_project)

    # Mutate one doc's content
    target = invoices_dir / "invoice_00002.json"
    data = json.loads(target.read_text())
    data["vendor"] = "MUTATED_VENDOR"
    target.write_text(json.dumps(data))

    results = run_project(fresh_project)
    raw = next(r for r in results if r.model_name == "raw_invoices")
    assert raw.documents_processed == 1
    assert raw.documents_skipped == 4

    db = fresh_project / "target" / "dbt_ml.duckdb"
    rows = _query(
        db,
        'SELECT vendor FROM "dbt_ml".dbt_ml.raw_invoices '
        "WHERE source_path = 'invoice_00002.json'",
    )
    assert rows[0][0] == "MUTATED_VENDOR"


def test_removed_doc_is_pruned_on_incremental(fresh_project: Path) -> None:
    invoices_dir = fresh_project / "data" / "invoices"
    generate_invoices(5, invoices_dir, seed=1)
    run_project(fresh_project)

    (invoices_dir / "invoice_00002.json").unlink()

    results = run_project(fresh_project)
    raw = next(r for r in results if r.model_name == "raw_invoices")
    assert raw.documents_processed == 0
    assert raw.documents_skipped == 4
    assert raw.documents_deleted == 1

    db = fresh_project / "target" / "dbt_ml.duckdb"
    rows = _query(db, 'SELECT COUNT(*) FROM "dbt_ml".dbt_ml.raw_invoices')
    assert rows[0][0] == 4
    gone = _query(
        db,
        'SELECT COUNT(*) FROM "dbt_ml".dbt_ml.raw_invoices '
        "WHERE source_path = 'invoice_00002.json'",
    )
    assert gone[0][0] == 0
    state = _query(
        db,
        "SELECT COUNT(*) FROM \"dbt_ml\".dbt_ml.dbt_ml_state "
        "WHERE model_name = 'raw_invoices'",
    )
    assert state[0][0] == 4


def test_full_refresh_reprocesses_all(fresh_project: Path) -> None:
    invoices_dir = fresh_project / "data" / "invoices"
    generate_invoices(5, invoices_dir, seed=1)
    run_project(fresh_project)

    results = run_project(fresh_project, full_refresh=True)
    raw = next(r for r in results if r.model_name == "raw_invoices")
    assert raw.documents_processed == 5
    assert raw.documents_skipped == 0


def test_incremental_transform_is_rejected(fresh_project: Path) -> None:
    generate_invoices(3, fresh_project / "data" / "invoices", seed=1)
    summary_yml = fresh_project / "models" / "invoice_summary.yml"
    text = summary_yml.read_text()
    summary_yml.write_text(text.replace("materialization: full", "materialization: incremental"))

    with pytest.raises(ConfigError, match="only supports `materialization: full`"):
        run_project(fresh_project, select="invoice_summary")


def test_transform_aggregates_dependency(fresh_project: Path) -> None:
    invoices_dir = fresh_project / "data" / "invoices"
    generate_invoices(20, invoices_dir, seed=1)
    run_project(fresh_project)

    db = fresh_project / "target" / "dbt_ml.duckdb"
    rows = _query(
        db,
        'SELECT SUM(invoice_count), SUM(total_spend) FROM "dbt_ml".dbt_ml.invoice_summary',
    )
    raw_rows = _query(
        db, 'SELECT COUNT(*), SUM(total) FROM "dbt_ml".dbt_ml.raw_invoices'
    )
    assert rows[0][0] == raw_rows[0][0]
    assert rows[0][1] == pytest.approx(raw_rows[0][1])


def test_llm_transform_reports_provider_provenance(fresh_project: Path) -> None:
    profiles = fresh_project / "profiles.yml"
    profiles.write_text(
        profiles.read_text().replace(
            "        schema: dbt_ml",
            "        schema: dbt_ml\n"
            "      llm:\n"
            "        provider: anthropic\n"
            "        model: provenance-model",
        )
    )
    transform_yml = fresh_project / "models" / "invoice_summary.yml"
    transform_yml.write_text(
        transform_yml.read_text().replace(
            "      module: transforms.summarize",
            "      module: transforms.summarize\n      uses_llm: true",
        )
    )
    generate_invoices(3, fresh_project / "data" / "invoices", seed=1)

    results = run_project(fresh_project)
    summary = next(r for r in results if r.model_name == "invoice_summary")

    assert summary.provider == "anthropic"
    assert summary.provider_model == "provenance-model"
    assert summary.provider_implementation is not None
    payload = json.loads(write_run_results(fresh_project, results).read_text())
    row = next(r for r in payload["results"] if r["model_name"] == "invoice_summary")
    assert row["provider"] == "anthropic"
    assert row["provider_model"] == "provenance-model"
    assert row["provider_implementation"] == summary.provider_implementation


def test_llm_transform_provider_failure_is_isolated_and_sanitized(
    fresh_project: Path,
) -> None:
    transform_yml = fresh_project / "models" / "invoice_summary.yml"
    transform_yml.write_text(
        transform_yml.read_text().replace(
            "      module: transforms.summarize",
            "      module: transforms.summarize\n      uses_llm: true",
        )
    )
    (fresh_project / "transforms" / "summarize.py").write_text(
        "from dbt_ml.providers import ProviderConfigurationError\n\n"
        "def run(deps, ctx):\n"
        "    raise ProviderConfigurationError('PRIVATE_LLM_ENV is not set')\n"
    )
    generate_invoices(3, fresh_project / "data" / "invoices", seed=1)

    result = build_project(fresh_project)
    summary = next(
        r for r in result.run_results if r.model_name == "invoice_summary"
    )

    assert summary.errors == [
        "ProviderConfigurationError: provider configuration is invalid"
    ]
    assert "PRIVATE_LLM_ENV" not in json.dumps(summary.errors)
    assert "monthly_totals" in {r.model_name for r in result.run_results}


def test_run_with_select(fresh_project: Path) -> None:
    generate_invoices(5, fresh_project / "data" / "invoices", seed=2)
    results = run_project(fresh_project, select="raw_invoices")
    assert [r.model_name for r in results] == ["raw_invoices"]


def test_run_with_select_descendants(fresh_project: Path) -> None:
    generate_invoices(5, fresh_project / "data" / "invoices", seed=2)
    results = run_project(fresh_project, select="raw_invoices+")
    assert {r.model_name for r in results} == {
        "raw_invoices",
        "invoice_summary",
        "monthly_totals",
    }


def test_run_with_exclude(fresh_project: Path) -> None:
    generate_invoices(5, fresh_project / "data" / "invoices", seed=2)
    results = run_project(fresh_project, exclude="invoice_summary")
    assert "invoice_summary" not in {r.model_name for r in results}
    assert {r.model_name for r in results} == {"raw_invoices", "monthly_totals"}


def test_run_with_threads_produces_same_results(fresh_project: Path) -> None:
    """Parallel extraction must yield the same rows as serial."""
    generate_invoices(20, fresh_project / "data" / "invoices", seed=4)

    results_serial = run_project(fresh_project)
    raw_serial = next(r for r in results_serial if r.model_name == "raw_invoices")
    assert raw_serial.rows_written == 20

    results_parallel = run_project(fresh_project, threads=4, full_refresh=True)
    raw_parallel = next(r for r in results_parallel if r.model_name == "raw_invoices")
    assert raw_parallel.rows_written == 20

    db = fresh_project / "target" / "dbt_ml.duckdb"
    rows = _query(db, 'SELECT COUNT(*) FROM "dbt_ml".dbt_ml.raw_invoices')
    assert rows[0][0] == 20


def test_threaded_run_parallelizes_independent_branches(fresh_project: Path) -> None:
    """invoice_summary and monthly_totals are independent siblings of raw_invoices;
    running the DAG with threads>1 must produce the same tables as a serial run."""
    generate_invoices(20, fresh_project / "data" / "invoices", seed=4)

    serial = run_project(fresh_project)
    serial_rows = {r.model_name: r.rows_written for r in serial}

    parallel = run_project(fresh_project, threads=4, full_refresh=True)
    parallel_rows = {r.model_name: r.rows_written for r in parallel}

    assert parallel_rows == serial_rows
    assert set(parallel_rows) == {"raw_invoices", "invoice_summary", "monthly_totals"}

    db = fresh_project / "target" / "dbt_ml.duckdb"
    summary = _query(db, 'SELECT COUNT(*) FROM "dbt_ml".dbt_ml.invoice_summary')
    monthly = _query(db, 'SELECT COUNT(*) FROM "dbt_ml".dbt_ml.monthly_totals')
    assert summary[0][0] > 0
    assert monthly[0][0] > 0


def test_clean_preserves_duckdb(fresh_project: Path) -> None:
    invoices_dir = fresh_project / "data" / "invoices"
    generate_invoices(2, invoices_dir, seed=1)
    run_project(fresh_project)
    db = fresh_project / "target" / "dbt_ml.duckdb"
    assert db.exists()

    clean_project(fresh_project)
    assert db.exists()


def test_classic_ml_tfidf_end_to_end(tmp_path: Path) -> None:
    src = Path(__file__).resolve().parents[1] / "examples" / "classic_text_ml"
    project = tmp_path / "classic_text_ml"
    shutil.copytree(src, project, ignore=shutil.ignore_patterns("data", "target"))
    generate_support_tickets(8, project / "data" / "tickets", seed=7)

    results = run_project(project)
    by_name = {r.model_name: r for r in results}
    ml_result = by_name["ticket_tfidf"]
    assert ml_result.kind == "ml"
    assert ml_result.rows_written > 0
    assert ml_result.artifact_version is not None
    assert ml_result.training_input is not None
    assert ml_result.training_input["refs"] == ["raw_tickets"]
    assert ml_result.training_input["row_count"] == 8
    assert ml_result.metrics["vocabulary_size"] > 0
    assert set(ml_result.metrics) == {"row_count", "vocabulary_size"}

    artifact = project / "target" / "artifacts" / "ticket_tfidf"
    metadata_path = artifact / "metadata.json"
    assert metadata_path.exists()
    assert (artifact / "vocabulary.json").exists()
    from dbt_ml.classic_ml import ARTIFACT_SCHEMA_VERSION

    metadata = json.loads(metadata_path.read_text())
    assert metadata["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert metadata["artifact_type"] == "classic_ml"
    assert metadata["artifact_version"] == ml_result.artifact_version
    assert metadata["artifact_files_hash"]
    assert metadata["code_version"]
    assert metadata["config_hash"]
    assert metadata["runtime"]["provider"] == "builtin.tfidf"

    registry_path = project / "target" / "artifacts" / "registry.json"
    registry = json.loads(registry_path.read_text())
    assert registry["artifacts"]["ticket_tfidf"]["artifact_version"] == ml_result.artifact_version

    db = project / "target" / "dbt_ml.duckdb"
    rows = _query(
        db,
        'SELECT COUNT(*), COUNT(DISTINCT row_id) FROM '
        '"dbt_ml".classic_text_ml.ticket_tfidf',
    )
    assert rows[0][0] == ml_result.rows_written
    assert rows[0][1] == 8

    run_results_path = write_run_results(project, results)
    payload = json.loads(run_results_path.read_text())
    emitted = next(r for r in payload["results"] if r["model_name"] == "ticket_tfidf")
    assert emitted["artifact_version"] == ml_result.artifact_version
    assert emitted["training_input"]["row_count"] == 8
    assert emitted["metrics"]["vocabulary_size"] == ml_result.metrics["vocabulary_size"]
    assert emitted["artifact_metadata"]["artifact_files_hash"] == metadata["artifact_files_hash"]


def test_classic_ml_tfidf_fit_then_predict(tmp_path: Path) -> None:
    src = Path(__file__).resolve().parents[1] / "examples" / "classic_text_ml"
    project = tmp_path / "classic_text_ml"
    shutil.copytree(src, project, ignore=shutil.ignore_patterns("data", "target"))
    (project / "models" / "ticket_tfidf.yml").write_text(
        "\n".join(
            [
                "version: 2",
                "models:",
                "  - name: ticket_tfidf_fit",
                "    depends_on: [ref('raw_tickets')]",
                "    ml:",
                "      task: features",
                "      mode: fit",
                "      provider: builtin.tfidf",
                "      text_field: summary",
                "      artifact:",
                "        path: target/artifacts/ticket_tfidf",
                "      options:",
                "        min_df: 1",
                "  - name: ticket_tfidf_predict",
                "    depends_on: [ref('raw_tickets'), ref('ticket_tfidf_fit')]",
                "    ml:",
                "      task: features",
                "      mode: predict",
                "      provider: builtin.tfidf",
                "      text_field: summary",
                "      artifact:",
                "        path: target/artifacts/ticket_tfidf",
            ]
        )
    )
    generate_support_tickets(5, project / "data" / "tickets", seed=11)

    results = run_project(project)
    by_name = {r.model_name: r for r in results}
    fit = by_name["ticket_tfidf_fit"]
    predict = by_name["ticket_tfidf_predict"]
    assert fit.rows_written == 1
    assert predict.rows_written > 0
    assert fit.artifact_version == predict.artifact_version
    assert fit.training_input == predict.training_input


def test_classic_ml_naive_bayes_classifier_end_to_end(tmp_path: Path) -> None:
    src = Path(__file__).resolve().parents[1] / "examples" / "classic_text_ml"
    project = tmp_path / "classic_text_ml"
    shutil.copytree(src, project, ignore=shutil.ignore_patterns("data", "target"))
    tickets = project / "data" / "tickets"
    tickets.mkdir(parents=True)
    _write_ticket(tickets / "ticket_1.json", "T-1", "urgent outage blocked", "high")
    _write_ticket(tickets / "ticket_2.json", "T-2", "critical outage urgent", "high")
    _write_ticket(tickets / "ticket_3.json", "T-3", "billing question invoice", "low")
    _write_ticket(tickets / "ticket_4.json", "T-4", "password reset question", "low")
    (project / "models" / "ticket_tfidf.yml").write_text(
        "\n".join(
            [
                "version: 2",
                "models:",
                "  - name: ticket_priority_classifier",
                "    depends_on: [ref('raw_tickets')]",
                "    ml:",
                "      task: classifier",
                "      mode: fit_transform",
                "      provider: builtin.naive_bayes",
                "      text_field: summary",
                "      label_field: priority",
                "      options:",
                "        min_df: 1",
                "        alpha: 1.0",
                "    materialization: full",
            ]
        )
    )

    results = run_project(project)
    classifier = next(r for r in results if r.model_name == "ticket_priority_classifier")
    assert classifier.kind == "ml"
    assert classifier.rows_written == 4
    assert classifier.artifact_version is not None
    assert classifier.metrics["class_count"] == 2
    assert classifier.metrics["vocabulary_size"] > 0
    assert classifier.metrics["accuracy"] == 1.0
    assert classifier.artifact_metadata is not None
    assert classifier.artifact_metadata["provider"] == "builtin.naive_bayes"

    artifact = project / "target" / "artifacts" / "ticket_priority_classifier"
    assert (artifact / "metadata.json").exists()
    assert (artifact / "model.json").exists()
    model_payload = json.loads((artifact / "model.json").read_text())
    assert model_payload["classes"] == ["high", "low"]

    db = project / "target" / "dbt_ml.duckdb"
    rows = _query(
        db,
        'SELECT COUNT(*), SUM(CASE WHEN correct THEN 1 ELSE 0 END) '
        'FROM "dbt_ml".classic_text_ml.ticket_priority_classifier',
    )
    assert rows == [(4, 4)]


def test_classic_ml_naive_bayes_fit_then_predict(tmp_path: Path) -> None:
    src = Path(__file__).resolve().parents[1] / "examples" / "classic_text_ml"
    project = tmp_path / "classic_text_ml"
    shutil.copytree(src, project, ignore=shutil.ignore_patterns("data", "target"))
    tickets = project / "data" / "tickets"
    tickets.mkdir(parents=True)
    _write_ticket(tickets / "ticket_1.json", "T-1", "urgent outage blocked", "high")
    _write_ticket(tickets / "ticket_2.json", "T-2", "critical outage urgent", "high")
    _write_ticket(tickets / "ticket_3.json", "T-3", "billing question invoice", "low")
    _write_ticket(tickets / "ticket_4.json", "T-4", "password reset question", "low")
    (project / "models" / "ticket_tfidf.yml").write_text(
        "\n".join(
            [
                "version: 2",
                "models:",
                "  - name: ticket_priority_fit",
                "    depends_on: [ref('raw_tickets')]",
                "    ml:",
                "      task: classifier",
                "      mode: fit",
                "      provider: builtin.naive_bayes",
                "      text_field: summary",
                "      label_field: priority",
                "      artifact:",
                "        path: target/artifacts/ticket_priority",
                "      options:",
                "        min_df: 1",
                "  - name: ticket_priority_predict",
                "    depends_on: [ref('raw_tickets'), ref('ticket_priority_fit')]",
                "    ml:",
                "      task: classifier",
                "      mode: predict",
                "      provider: builtin.naive_bayes",
                "      text_field: summary",
                "      label_field: priority",
                "      artifact:",
                "        path: target/artifacts/ticket_priority",
            ]
        )
    )

    results = run_project(project)
    by_name = {r.model_name: r for r in results}
    fit = by_name["ticket_priority_fit"]
    predict = by_name["ticket_priority_predict"]
    assert fit.rows_written == 1
    assert predict.rows_written == 4
    assert fit.artifact_version == predict.artifact_version
    assert fit.training_input == predict.training_input
    assert predict.metrics["accuracy"] == 1.0


def test_classic_ml_predict_missing_artifact_reports_lifecycle_error(tmp_path: Path) -> None:
    src = Path(__file__).resolve().parents[1] / "examples" / "classic_text_ml"
    project = tmp_path / "classic_text_ml"
    shutil.copytree(src, project, ignore=shutil.ignore_patterns("data", "target"))
    (project / "models" / "ticket_tfidf.yml").write_text(
        "\n".join(
            [
                "version: 2",
                "models:",
                "  - name: ticket_tfidf_predict",
                "    depends_on: [ref('raw_tickets')]",
                "    ml:",
                "      task: features",
                "      mode: predict",
                "      provider: builtin.tfidf",
                "      text_field: summary",
                "      artifact:",
                "        path: target/artifacts/missing_tfidf",
            ]
        )
    )
    generate_support_tickets(2, project / "data" / "tickets", seed=12)

    with pytest.raises(RunError, match="missing artifact metadata"):
        run_project(project)


def test_classic_ml_predict_stale_artifact_payload_reports_lifecycle_error(
    tmp_path: Path,
) -> None:
    src = Path(__file__).resolve().parents[1] / "examples" / "classic_text_ml"
    project = tmp_path / "classic_text_ml"
    shutil.copytree(src, project, ignore=shutil.ignore_patterns("data", "target"))
    generate_support_tickets(4, project / "data" / "tickets", seed=13)
    run_project(project)

    vocab_path = project / "target" / "artifacts" / "ticket_tfidf" / "vocabulary.json"
    vocab = json.loads(vocab_path.read_text())
    vocab["terms"].append("synthetic_stale_term")
    vocab_path.write_text(json.dumps(vocab, indent=2, sort_keys=True))
    (project / "models" / "ticket_tfidf.yml").write_text(
        "\n".join(
            [
                "version: 2",
                "models:",
                "  - name: ticket_tfidf_predict",
                "    depends_on: [ref('raw_tickets')]",
                "    ml:",
                "      task: features",
                "      mode: predict",
                "      provider: builtin.tfidf",
                "      text_field: summary",
                "      artifact:",
                "        path: target/artifacts/ticket_tfidf",
            ]
        )
    )

    with pytest.raises(RunError, match="stale artifact payload"):
        run_project(project)


def test_classic_ml_predict_incompatible_provider_reports_lifecycle_error(
    tmp_path: Path,
) -> None:
    src = Path(__file__).resolve().parents[1] / "examples" / "classic_text_ml"
    project = tmp_path / "classic_text_ml"
    shutil.copytree(src, project, ignore=shutil.ignore_patterns("data", "target"))
    generate_support_tickets(4, project / "data" / "tickets", seed=14)
    run_project(project)
    (project / "models" / "ticket_tfidf.yml").write_text(
        "\n".join(
            [
                "version: 2",
                "models:",
                "  - name: ticket_count_predict",
                "    depends_on: [ref('raw_tickets')]",
                "    ml:",
                "      task: features",
                "      mode: predict",
                "      provider: builtin.count",
                "      text_field: summary",
                "      artifact:",
                "        path: target/artifacts/ticket_tfidf",
            ]
        )
    )

    with pytest.raises(RunError, match="incompatible artifact provider"):
        run_project(project)


def test_classic_ml_count_vectorizer_options(tmp_path: Path) -> None:
    src = Path(__file__).resolve().parents[1] / "examples" / "classic_text_ml"
    project = tmp_path / "classic_text_ml"
    shutil.copytree(src, project, ignore=shutil.ignore_patterns("data", "target"))
    tickets = project / "data" / "tickets"
    tickets.mkdir(parents=True)
    _write_ticket(tickets / "ticket_1.json", "T-1", "alpha alpha beta the")
    _write_ticket(tickets / "ticket_2.json", "T-2", "beta gamma the")
    (project / "models" / "ticket_tfidf.yml").write_text(
        "\n".join(
            [
                "version: 2",
                "models:",
                "  - name: ticket_count",
                "    depends_on: [ref('raw_tickets')]",
                "    ml:",
                "      task: features",
                "      mode: fit_transform",
                "      provider: builtin.count",
                "      text_field: summary",
                "      options:",
                "        binary: true",
                "        stop_words: [the]",
                "    materialization: full",
            ]
        )
    )

    results = run_project(project)
    count = next(r for r in results if r.model_name == "ticket_count")
    assert count.rows_written == 4
    assert count.metrics["vocabulary_size"] == 3

    db = project / "target" / "dbt_ml.duckdb"
    rows = _query(
        db,
        'SELECT term, SUM(count), SUM(value) FROM "dbt_ml".classic_text_ml.ticket_count '
        "GROUP BY term ORDER BY term",
    )
    assert rows == [
        ("alpha", 1, 1.0),
        ("beta", 2, 2.0),
        ("gamma", 1, 1.0),
    ]
    vocab_path = project / "target" / "artifacts" / "ticket_count" / "vocabulary.json"
    vocab = json.loads(vocab_path.read_text())
    assert vocab["terms"] == ["alpha", "beta", "gamma"]


def test_classic_ml_hashing_vectorizer(tmp_path: Path) -> None:
    src = Path(__file__).resolve().parents[1] / "examples" / "classic_text_ml"
    project = tmp_path / "classic_text_ml"
    shutil.copytree(src, project, ignore=shutil.ignore_patterns("data", "target"))
    tickets = project / "data" / "tickets"
    tickets.mkdir(parents=True)
    _write_ticket(tickets / "ticket_1.json", "T-1", "alpha beta")
    _write_ticket(tickets / "ticket_2.json", "T-2", "alpha")
    (project / "models" / "ticket_tfidf.yml").write_text(
        "\n".join(
            [
                "version: 2",
                "models:",
                "  - name: ticket_hashing",
                "    depends_on: [ref('raw_tickets')]",
                "    ml:",
                "      task: features",
                "      mode: fit_transform",
                "      provider: builtin.hashing",
                "      text_field: summary",
                "      options:",
                "        n_features: 8",
                "        alternate_sign: false",
                "    materialization: full",
            ]
        )
    )

    results = run_project(project)
    hashing = next(r for r in results if r.model_name == "ticket_hashing")
    assert hashing.rows_written > 0
    assert hashing.metrics["hash_buckets"] == 8
    assert hashing.metrics["vocabulary_size"] == 0

    db = project / "target" / "dbt_ml.duckdb"
    rows = _query(
        db,
        'SELECT MIN(hash_bucket), MAX(hash_bucket), COUNT(DISTINCT term) '
        'FROM "dbt_ml".classic_text_ml.ticket_hashing',
    )
    assert rows[0][0] >= 0
    assert rows[0][1] < 8
    assert rows[0][2] > 0
    metadata = json.loads(
        (project / "target" / "artifacts" / "ticket_hashing" / "metadata.json").read_text()
    )
    assert metadata["files"] == ["metadata.json"]


def test_classic_ml_tfidf_character_ngrams(tmp_path: Path) -> None:
    src = Path(__file__).resolve().parents[1] / "examples" / "classic_text_ml"
    project = tmp_path / "classic_text_ml"
    shutil.copytree(src, project, ignore=shutil.ignore_patterns("data", "target"))
    tickets = project / "data" / "tickets"
    tickets.mkdir(parents=True)
    _write_ticket(tickets / "ticket_1.json", "T-1", "abc abc")
    _write_ticket(tickets / "ticket_2.json", "T-2", "abd")
    (project / "models" / "ticket_tfidf.yml").write_text(
        "\n".join(
            [
                "version: 2",
                "models:",
                "  - name: ticket_char_tfidf",
                "    depends_on: [ref('raw_tickets')]",
                "    ml:",
                "      task: features",
                "      mode: fit_transform",
                "      provider: builtin.tfidf",
                "      text_field: summary",
                "      options:",
                "        analyzer: char",
                "        ngram_range: [3, 3]",
                "        min_df: 1",
                "    materialization: full",
            ]
        )
    )

    run_project(project)

    db = project / "target" / "dbt_ml.duckdb"
    rows = _query(
        db,
        'SELECT term, COUNT(*) FROM "dbt_ml".classic_text_ml.ticket_char_tfidf '
        "WHERE term = 'abc' GROUP BY term",
    )
    assert rows == [("abc", 1)]


def test_discover_source_uses_posix_relative_paths(tmp_path: Path) -> None:
    """document_id hashes the relative path, so separators must be `/` on
    every OS — otherwise state written on Windows invalidates on Linux (#67)."""
    nested = tmp_path / "data" / "batch_a"
    nested.mkdir(parents=True)
    (nested / "doc.json").write_text("{}")

    refs = LocalDocumentSource().discover(
        SourceConfig(name="docs", path="data", file_pattern="*.json"), tmp_path
    )
    assert [r.relative_path for r in refs] == ["batch_a/doc.json"]
    assert refs[0].document_id == compute_document_id("docs", "batch_a/doc.json")


def test_target_source_path_override_drives_discovery(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    dev_root = tmp_path / "dev_docs"
    prod_root = tmp_path / "prod_docs"
    (project / "dbt_ml_project.yml").write_text(
        "name: docs\nversion: '0.1.0'\nprofile: docs\n"
    )
    (project / "profiles.yml").write_text(
        f"docs:\n"
        f"  target: dev\n"
        f"  outputs:\n"
        f"    dev:\n"
        f"      warehouse:\n"
        f"        type: duckdb\n"
        f"        path: ./target/dev/db.duckdb\n"
        f"        schema: docs\n"
        f"      source_paths:\n"
        f"        docs_src: {dev_root.as_posix()}\n"
        f"    prod:\n"
        f"      warehouse:\n"
        f"        type: duckdb\n"
        f"        path: ./target/prod/db.duckdb\n"
        f"        schema: docs\n"
        f"      source_paths:\n"
        f"        docs_src: {prod_root.as_posix()}\n"
    )
    (project / "sources").mkdir()
    (project / "sources" / "docs.yml").write_text(
        "version: 2\nsources:\n"
        "  - name: docs_src\n"
        "    path: data/prod\n"
        "    file_pattern: '*.json'\n"
    )
    (project / "models").mkdir()
    (project / "models" / "raw_docs.yml").write_text(
        "version: 2\nmodels:\n"
        "  - name: raw_docs\n"
        "    source: ref('docs_src')\n"
        "    extraction:\n"
        "      backend: json\n"
        "      options:\n"
        "        fields: [title]\n"
    )
    for target_name, root in (("dev", dev_root), ("prod", prod_root)):
        root.mkdir(parents=True)
        (root / "doc.json").write_text(json.dumps({"title": target_name}))

    run_project(project, target="dev")
    run_project(project, target="prod")

    dev_rows = _query(
        project / "target" / "dev" / "db.duckdb",
        'SELECT document_id, source_path, title FROM "db".docs.raw_docs',
    )
    prod_rows = _query(
        project / "target" / "prod" / "db.duckdb",
        'SELECT document_id, source_path, title FROM "db".docs.raw_docs',
    )

    assert dev_rows == [(prod_rows[0][0], "doc.json", "dev")]
    assert prod_rows == [(dev_rows[0][0], "doc.json", "prod")]


def _drop_currency_field(project: Path, *, on_schema_change: str | None = None) -> None:
    """Remove `currency` from raw_invoices' extraction so the next run's
    staging frame is missing a column the table already has."""
    raw_yml = project / "models" / "raw_invoices.yml"
    text = raw_yml.read_text().replace(
        "fields: [invoice_id, vendor, issue_date, line_items, total, currency]",
        "fields: [invoice_id, vendor, issue_date, line_items, total]",
    )
    text = text.replace(
        "      - name: currency\n        data_type: string\n",
        "",
    ).replace("      - name: currency\n", "")
    if on_schema_change is not None:
        text = text.replace(
            "materialization: incremental",
            f"materialization: incremental\n    on_schema_change: {on_schema_change}",
        )
    raw_yml.write_text(text)


def test_incremental_schema_change_fails_actionably(fresh_project: Path) -> None:
    generate_invoices(5, fresh_project / "data" / "invoices", seed=1)
    run_project(fresh_project)

    _drop_currency_field(fresh_project)
    with pytest.raises(RunError, match="full-refresh"):
        run_project(fresh_project, select="raw_invoices")


def test_incremental_schema_change_append_policy(fresh_project: Path) -> None:
    generate_invoices(5, fresh_project / "data" / "invoices", seed=1)
    run_project(fresh_project)

    _drop_currency_field(fresh_project, on_schema_change="append_new_columns")
    results = run_project(fresh_project, select="raw_invoices")
    raw = next(r for r in results if r.model_name == "raw_invoices")
    # config change bumps code_version, so every doc reprocesses
    assert raw.documents_processed == 5

    db = fresh_project / "target" / "dbt_ml.duckdb"
    rows = _query(
        db,
        'SELECT COUNT(*), COUNT(currency) FROM "dbt_ml".dbt_ml.raw_invoices',
    )
    assert rows[0] == (5, 0)


def test_full_refresh_clears_schema_drift(fresh_project: Path) -> None:
    generate_invoices(5, fresh_project / "data" / "invoices", seed=1)
    run_project(fresh_project)

    _drop_currency_field(fresh_project)
    results = run_project(fresh_project, select="raw_invoices", full_refresh=True)
    raw = next(r for r in results if r.model_name == "raw_invoices")
    assert raw.documents_processed == 5

    db = fresh_project / "target" / "dbt_ml.duckdb"
    cols = _query(
        db,
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'dbt_ml' AND table_name = 'raw_invoices'",
    )
    assert ("currency",) not in cols
