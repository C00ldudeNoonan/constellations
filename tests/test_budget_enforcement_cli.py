"""End-to-end proof that budgets actually stop a run (issue #310).

The unit tests in `test_llm_batch_safety.py` verify the accounting: a ledger
raises when its running total passes the cap. That is a different claim from
the one operators depend on — that `stel run` on a project over its cap exits
non-zero, stops before the next provider call, and keeps the work it already
committed. A cost control that is never observed tripping end to end is
untested where it matters: a refactor that stops consulting the cap on some
path leaves every accounting unit test green, and the first real signal is a
provider bill.

Every test here drives the real CLI over a real project with the offline
`deterministic` provider, so the whole chain — config, runner, backend, ledger,
exit code — is exercised for free. Each names the regression it prevents.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from stel.cli import cli
from stel.synth import generate_invoice_texts

_DOCUMENT_COUNT = 5

# No cache_path: a cache would serve already-extracted documents for free and
# make `api_calls` a measure of cache misses rather than of provider calls,
# which is precisely what the caps meter.
_OFFLINE_PROFILE = """\
llm_invoice_pipeline:
  target: dev
  outputs:
    dev:
      warehouse:
        type: duckdb
        path: ./target/stel.duckdb
        schema: llm_invoices
      llm:
        provider: deterministic
        model: deterministic-v1
        pricing:
          input_usd_per_mtok: 1000000.0
          output_usd_per_mtok: 1000000.0
"""


@pytest.fixture
def budget_project(tmp_path: Path) -> Path:
    """The LLM invoice example, repointed at the offline provider.

    `flush_every: 1` makes each document its own publication unit so a
    mid-corpus stop has committed work to keep — the property tested below.
    """
    repo = Path(__file__).resolve().parents[1]
    dst = tmp_path / "budget_project"
    shutil.copytree(
        repo / "examples" / "llm_invoice_pipeline",
        dst,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    (dst / "profiles.yml").write_text(_OFFLINE_PROFILE)
    _set_flush_every(dst, "raw_invoices_llm", 1)
    generate_invoice_texts(_DOCUMENT_COUNT, dst / "data" / "invoices_text", 1)
    return dst


def _set_flush_every(project: Path, model: str, value: int) -> None:
    """Set `extraction.flush_every`, the model's publication unit."""
    path = project / "models" / f"{model}.yml"
    path.write_text(
        path.read_text().replace(
            "    extraction:", f"    extraction:\n      flush_every: {value}", 1
        )
    )


def _set_budget(project: Path, model: str, caps: str) -> None:
    """Insert an `extraction.options.budget` block into a model file."""
    path = project / "models" / f"{model}.yml"
    path.write_text(
        path.read_text().replace(
            "      options:", f"      options:\n        budget:\n{caps}", 1
        )
    )


def _add_second_model(project: Path) -> None:
    """A second llm extraction model over the same source, for scope tests."""
    first = project / "models" / "raw_invoices_llm.yml"
    second = project / "models" / "raw_invoices_llm_b.yml"
    second.write_text(first.read_text().replace("raw_invoices_llm", "raw_invoices_llm_b"))


def _run(project: Path, *args: str) -> tuple[int, str, dict[str, Any]]:
    """Invoke `stel run`, returning (exit code, output, run_results payload)."""
    result = CliRunner().invoke(cli, ["--project-dir", str(project), "run", *args])
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        raise result.exception
    payload: dict[str, Any] = json.loads(
        (project / "target" / "run_results.json").read_text()
    )
    return result.exit_code, result.output, payload


def _model_result(payload: dict[str, Any], name: str) -> dict[str, Any]:
    results: list[dict[str, Any]] = payload["results"]
    return next(r for r in results if r["model_name"] == name)


def test_a_run_over_its_cap_exits_nonzero_and_reports_budget_exceeded(
    budget_project: Path,
) -> None:
    """Prevents: the cap trips internally but the CLI still exits 0, so an
    orchestrator branches on success and schedules the next stage."""
    _set_budget(budget_project, "raw_invoices_llm", "          max_documents: 1")

    exit_code, _, payload = _run(budget_project)

    assert exit_code == 1
    result = _model_result(payload, "raw_invoices_llm")
    assert result["status"] == "budget_exceeded"
    assert result["rows_written"] == 0
    errors: list[str] = result["errors"]
    assert any("max_documents" in str(e) for e in errors)


def test_a_run_under_its_cap_does_not_trip(budget_project: Path) -> None:
    """The complement, so a budget that always raises cannot pass as working:
    a cap comfortably above what the corpus needs must not stop anything."""
    _set_budget(
        budget_project, "raw_invoices_llm", f"          max_documents: {_DOCUMENT_COUNT + 1}"
    )

    exit_code, _, payload = _run(budget_project)

    assert exit_code == 0
    result = _model_result(payload, "raw_invoices_llm")
    assert result["status"] == "success"
    assert result["rows_written"] == _DOCUMENT_COUNT


def test_the_stop_does_not_overshoot_the_api_call_cap(budget_project: Path) -> None:
    """Prevents: the cap is consulted only after a batch completes, so a run
    bills a whole batch past its ceiling before noticing. `ensure_headroom()`
    gates the *next* call, so with single-threaded execution the observed call
    count must land exactly on the cap, never above it."""
    _set_budget(budget_project, "raw_invoices_llm", "          max_api_calls: 2")

    exit_code, _, payload = _run(budget_project, "--threads", "1")

    assert exit_code == 1
    result = _model_result(payload, "raw_invoices_llm")
    assert result["status"] == "budget_exceeded"
    metrics: dict[str, Any] = result["metrics"]
    assert metrics["api_calls"] == 2


def test_a_token_cap_stops_the_run(budget_project: Path) -> None:
    """Token spend is only measurable after a response, so its cap stops the
    call that *would follow* the overrun. Prevents: token caps silently
    ignored because only the document/call caps are wired into the loop."""
    _set_budget(budget_project, "raw_invoices_llm", "          max_input_tokens: 1")

    exit_code, _, payload = _run(budget_project, "--threads", "1")

    assert exit_code == 1
    result = _model_result(payload, "raw_invoices_llm")
    assert result["status"] == "budget_exceeded"
    errors: list[str] = result["errors"]
    assert any("max_input_tokens" in str(e) for e in errors)


def test_a_cost_cap_stops_the_run(budget_project: Path) -> None:
    """The cap operators actually reach for. The offline provider reports no
    cost, so this also proves the profile-pricing estimator feeds the ledger —
    prevents a cost cap that can never trip because nothing charges it."""
    _set_budget(budget_project, "raw_invoices_llm", "          max_cost_usd: 0.5")

    exit_code, _, payload = _run(budget_project, "--threads", "1")

    assert exit_code == 1
    result = _model_result(payload, "raw_invoices_llm")
    assert result["status"] == "budget_exceeded"
    errors: list[str] = result["errors"]
    assert any("max_cost_usd" in str(e) for e in errors)


def test_a_run_stopped_mid_corpus_keeps_its_committed_work_and_resumes(
    budget_project: Path,
) -> None:
    """The documented behavior and the easiest to regress: an incremental model
    that trips mid-run keeps the rows it already published, with their state, so
    a rerun continues instead of restarting. This is also what makes a
    partitioned backfill affordable — a stop must not throw away paid work."""
    _set_budget(budget_project, "raw_invoices_llm", "          max_api_calls: 2")

    exit_code, _, payload = _run(budget_project, "--threads", "1")
    assert exit_code == 1
    stopped = _model_result(payload, "raw_invoices_llm")
    assert stopped["status"] == "budget_exceeded"
    committed = stopped["rows_written"]
    assert committed == 2, "flush_every: 1 should publish each completed document"

    # Rerun with headroom: the committed documents are already in state, so the
    # second run must pick up the remainder rather than reprocessing the corpus.
    _raise_cap(budget_project, "raw_invoices_llm", "max_api_calls", _DOCUMENT_COUNT)
    exit_code, _, payload = _run(budget_project, "--threads", "1")

    assert exit_code == 0
    resumed = _model_result(payload, "raw_invoices_llm")
    assert resumed["status"] == "success"
    assert resumed["documents_processed"] == _DOCUMENT_COUNT - 2
    assert resumed["documents_skipped"] == 2
    metrics: dict[str, Any] = resumed["metrics"]
    assert metrics["api_calls"] == _DOCUMENT_COUNT - 2


def test_a_model_cap_does_not_trip_on_another_models_spend(
    budget_project: Path,
) -> None:
    """Prevents: a per-model ledger accidentally shared across models, so one
    expensive model exhausts every other model's private cap."""
    _add_second_model(budget_project)
    # A cap that exactly covers this model's own corpus: it must survive the
    # sibling model spending the same amount again.
    _set_budget(
        budget_project, "raw_invoices_llm", f"          max_api_calls: {_DOCUMENT_COUNT}"
    )

    exit_code, _, payload = _run(budget_project, "--threads", "1")

    assert exit_code == 0
    for name in ("raw_invoices_llm", "raw_invoices_llm_b"):
        result = _model_result(payload, name)
        assert result["status"] == "success", f"{name} must not trip on the other's spend"
        assert result["rows_written"] == _DOCUMENT_COUNT


def test_the_run_wide_cap_sums_across_models(budget_project: Path) -> None:
    """The run-scope ledger in `profiles.yml` is shared by every model in the
    invocation. Prevents: it being applied per model, which silently multiplies
    the operator's declared ceiling by the number of models in the project."""
    _add_second_model(budget_project)
    profiles = budget_project / "profiles.yml"
    profiles.write_text(
        profiles.read_text()
        + f"        budget:\n          max_api_calls: {_DOCUMENT_COUNT + 1}\n"
    )

    exit_code, _, payload = _run(budget_project, "--threads", "1")

    assert exit_code == 1
    # Whichever model runs first fits under the shared cap and the other
    # exhausts it. Asserting on the pair rather than on names keeps this about
    # the shared ledger and not about DAG ordering between two peer models.
    statuses = sorted(
        _model_result(payload, name)["status"]
        for name in ("raw_invoices_llm", "raw_invoices_llm_b")
    )
    assert statuses == ["budget_exceeded", "success"]


def _raise_cap(project: Path, model: str, limit: str, value: int) -> None:
    path = project / "models" / f"{model}.yml"
    text = path.read_text()
    old = next(line for line in text.splitlines() if limit in line)
    path.write_text(text.replace(old, f"          {limit}: {value}", 1))
