"""`stel plan` reports what a run would reprocess without running it (issue #529)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from stel.adapters import StateRecord, StateScope, create_adapter, parse_warehouse_config
from stel.cli import cli
from stel.plan import ModelPlan, plan_project
from stel.runner import run_project
from stel.synth import generate_invoices

# Runs a whole project or opens a retrieval store, so it belongs to the
# `e2e` tier (issue #518). `test_test_tiers.py` fails if a file that
# does either is missing this.
pytestmark = pytest.mark.e2e

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _copy_example(tmp_path: Path, name: str) -> Path:
    dst = tmp_path / name
    shutil.copytree(
        EXAMPLES / name,
        dst,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    return dst


@pytest.fixture
def invoice_project(tmp_path: Path) -> Path:
    project = _copy_example(tmp_path, "invoice_pipeline")
    generate_invoices(5, project / "data" / "invoices", seed=1)
    return project


@pytest.fixture
def rag_project(tmp_path: Path) -> Path:
    pytest.importorskip("lancedb")
    return _copy_example(tmp_path, "rag_chunks_pipeline")


def _by_name(project_dir: Path, *, select: str | None = None) -> dict[str, ModelPlan]:
    plan = plan_project(project_dir, select=select)
    return {model.name: model for model in plan.models}


def _replace_in_model(project_dir: Path, filename: str, old: str, new: str) -> None:
    path = project_dir / "models" / filename
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1, old
    path.write_text(text.replace(old, new), encoding="utf-8")


# ─── the adapter read ───────────────────────────────────────────────────────


def test_adapter_counts_state_rows_by_code_version(tmp_path: Path) -> None:
    config = parse_warehouse_config({"type": "duckdb", "path": str(tmp_path / "w.duckdb")})
    scope = StateScope("docs")
    other = StateScope("other")
    with create_adapter(config, project_dir=tmp_path) as adapter:
        assert adapter.state_code_version_counts(scope) == {}
        adapter.upsert_state(
            scope,
            [
                StateRecord("a", "fp-a", "v1"),
                StateRecord("b", "fp-b", "v1"),
                StateRecord("c", "fp-c", "v2"),
            ],
        )
        adapter.upsert_state(other, [StateRecord("z", "fp-z", "v9")])
        assert adapter.state_code_version_counts(scope) == {"v1": 2, "v2": 1}
        assert adapter.state_code_version_counts(other) == {"v9": 1}


# ─── classification ─────────────────────────────────────────────────────────


def test_never_run_project_is_new_or_full(invoice_project: Path) -> None:
    plans = _by_name(invoice_project)
    assert plans["raw_invoices"].status == "new"
    assert plans["raw_invoices"].state_rows == 0
    # Full-materialization models are rebuilt every run; state never applies.
    assert plans["invoice_summary"].status == "full"
    assert plans["monthly_totals"].status == "full"
    # No paid kind here, and a full model is never estimated: 0 would read as free.
    assert all(m.estimated_provider_calls is None for m in plans.values())


def test_unchanged_after_a_run(invoice_project: Path) -> None:
    run_project(invoice_project)
    raw = _by_name(invoice_project)["raw_invoices"]
    assert raw.status == "unchanged"
    assert raw.state_rows == 5
    assert raw.stale_rows == 0
    assert raw.rows_to_reprocess == 0
    assert raw.caused_by == ()


def test_config_change_counts_stale_rows_and_reaches_downstream(
    invoice_project: Path,
) -> None:
    run_project(invoice_project)
    # A projected field is part of the extraction identity.
    _replace_in_model(
        invoice_project,
        "raw_invoices.yml",
        "fields: [invoice_id, vendor, issue_date, line_items, total, currency]",
        "fields: [invoice_id, vendor, issue_date, total, currency]",
    )
    plans = _by_name(invoice_project)
    raw = plans["raw_invoices"]
    assert raw.status == "changed"
    assert (raw.state_rows, raw.stale_rows, raw.rows_to_reprocess) == (5, 5, 5)
    assert raw.reprocess_is_upper_bound is False
    for name in ("invoice_summary", "monthly_totals"):
        assert plans[name].status == "full"
        assert plans[name].caused_by == ("raw_invoices",)
        assert "raw_invoices" in plans[name].reason


def test_cadence_change_is_not_a_change(invoice_project: Path) -> None:
    run_project(invoice_project)
    _replace_in_model(
        invoice_project,
        "raw_invoices.yml",
        "      backend: json\n",
        "      backend: json\n      flush_every: 7\n      publish_every: 3\n",
    )
    assert _by_name(invoice_project)["raw_invoices"].status == "unchanged"


def test_partial_reprocess_after_a_run_that_mixed_versions(
    invoice_project: Path,
) -> None:
    """Rows published under two code versions are counted per version, so a
    run that got half-way through a reprocess reports only the remainder."""
    run_project(invoice_project)
    _replace_in_model(
        invoice_project,
        "raw_invoices.yml",
        "fields: [invoice_id, vendor, issue_date, line_items, total, currency]",
        "fields: [invoice_id, vendor, issue_date, total, currency]",
    )
    current = _by_name(invoice_project)["raw_invoices"].code_version
    config = parse_warehouse_config(
        {"type": "duckdb", "path": str(invoice_project / "target" / "stel.duckdb")}
    )
    # Simulate the interrupted reprocess: two documents already carry the new
    # code_version in state.
    with create_adapter(config, project_dir=invoice_project) as adapter:
        state = adapter.fetch_state(StateScope("raw_invoices"))
        keys = sorted(state)[:2]
        adapter.upsert_state(
            StateScope("raw_invoices"),
            [StateRecord(key, state[key].input_fingerprint, current) for key in keys],
        )
    raw = _by_name(invoice_project)["raw_invoices"]
    assert raw.status == "changed"
    assert (raw.state_rows, raw.stale_rows, raw.rows_to_reprocess) == (5, 3, 3)


def test_selection_bounds_the_plan_and_the_cascade(invoice_project: Path) -> None:
    run_project(invoice_project)
    _replace_in_model(
        invoice_project,
        "raw_invoices.yml",
        "fields: [invoice_id, vendor, issue_date, line_items, total, currency]",
        "fields: [invoice_id, vendor, issue_date, total, currency]",
    )
    only_downstream = _by_name(invoice_project, select="monthly_totals")
    assert set(only_downstream) == {"monthly_totals"}
    # raw_invoices is not planned, so it cannot be the cause of anything here:
    # a run with this selection would not run it either.
    assert only_downstream["monthly_totals"].caused_by == ()


# ─── the cascade through a chunk -> embed / llm -> search chain ─────────────


def test_chunk_change_cascades_to_embed_llm_and_search(rag_project: Path) -> None:
    run_project(rag_project)
    before = _by_name(rag_project)
    # The golden-set model is `materialization: full`; everything else has
    # published state that matches its code.
    assert {m.status for m in before.values()} <= {"unchanged", "full"}
    assert before["document_chunks"].status == "unchanged"
    chunk_rows = before["document_chunks"].state_rows
    assert chunk_rows > 0

    _replace_in_model(
        rag_project, "document_chunks.yml", "chunk_size: 800", "chunk_size: 400"
    )
    plans = _by_name(rag_project)

    assert plans["document_registry"].status == "unchanged"
    chunks = plans["document_chunks"]
    assert chunks.status == "changed"
    assert chunks.rows_to_reprocess == chunk_rows
    assert chunks.reprocess_is_upper_bound is False
    assert chunks.estimated_provider_calls is None

    for name in ("chunk_embeddings", "chunk_facts", "chunk_entities", "chunk_search"):
        downstream = plans[name]
        assert downstream.status == "cascade", name
        assert downstream.caused_by == ("document_chunks",), name
        assert downstream.reprocess_is_upper_bound is True, name
        assert downstream.rows_to_reprocess == downstream.state_rows, name
        assert downstream.state_rows > 0, name

    embed = plans["chunk_embeddings"]
    # The deterministic provider does not split a batch: one request per
    # batch of 128, rounded up.
    assert embed.estimated_provider_calls == -(-embed.state_rows // 128)
    assert (embed.provider, embed.provider_model) == ("deterministic", "deterministic-v1")

    facts = plans["chunk_facts"]
    assert facts.estimated_provider_calls == facts.state_rows
    assert facts.provider == "deterministic"

    # The search index publishes into its own serving scope; nothing here
    # opened the store.
    assert plans["chunk_search"].estimated_provider_calls is None
    assert plans["chunk_search"].kind == "search"


def test_embed_only_change_does_not_reach_upstream(rag_project: Path) -> None:
    run_project(rag_project)
    # The model name is part of the embedding identity; the deterministic
    # provider accepts any name, so this compiles and re-keys nothing upstream.
    _replace_in_model(
        rag_project,
        "chunk_embeddings.yml",
        "model: deterministic-v1",
        "model: deterministic-v2",
    )
    plans = _by_name(rag_project, select="chunk_embeddings+")
    assert set(plans) == {"chunk_embeddings", "chunk_search"}
    assert plans["chunk_embeddings"].status == "changed"
    assert plans["chunk_search"].status == "cascade"


# ─── the CLI ────────────────────────────────────────────────────────────────


def test_cli_table_and_json(invoice_project: Path) -> None:
    run_project(invoice_project)
    runner = CliRunner()

    result = runner.invoke(cli, ["--project-dir", str(invoice_project), "plan"])
    assert result.exit_code == 0, result.output
    assert "raw_invoices" in result.output
    assert "unchanged" in result.output
    assert "No source was discovered and no provider was called" in result.output

    result = runner.invoke(cli, ["--project-dir", str(invoice_project), "plan", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["metadata"]["schema_version"] == 1
    assert payload["metadata"]["invocation"] == "plan"
    assert payload["metadata"]["target"]["adapter_type"] == "duckdb"
    assert payload["metadata"]["counts"]["total"] == 3
    assert payload["metadata"]["counts"]["unchanged"] == 1
    raw = next(m for m in payload["models"] if m["name"] == "raw_invoices")
    assert raw["status"] == "unchanged"
    assert raw["caused_by"] == []
    on_disk = (invoice_project / "target" / "plan.json").read_text(encoding="utf-8")
    assert result.output.strip() == on_disk.strip()


def test_cli_rejects_state_selector_without_manifest(invoice_project: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--project-dir", str(invoice_project), "plan", "--select", "state:modified"]
    )
    # A selector that cannot be resolved is a configuration error, exit 2,
    # exactly as `run` reports it.
    assert result.exit_code == 2
    assert "state" in result.output


def test_clean_removes_the_plan_artifact(invoice_project: Path) -> None:
    runner = CliRunner()
    assert runner.invoke(cli, ["--project-dir", str(invoice_project), "plan"]).exit_code == 0
    artifact = invoice_project / "target" / "plan.json"
    assert artifact.exists()
    assert runner.invoke(cli, ["--project-dir", str(invoice_project), "clean"]).exit_code == 0
    assert not artifact.exists()
