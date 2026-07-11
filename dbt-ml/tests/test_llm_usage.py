"""Token/cost accounting for LLM extraction (issue #75, part 1).

Runs examples/llm_invoice_pipeline with the API mocked to return usage, and
asserts per-model totals land on ModelRunResult.metrics, in run_results.json,
and in the `run` summary output.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from dbt_ml.backends import llm_backend
from dbt_ml.cli import cli
from dbt_ml.manifest import write_manifest, write_run_results
from dbt_ml.runner import run_project
from dbt_ml.synth import generate_invoice_texts

_FIELDS = {
    "vendor": "Mocked Vendor",
    "invoice_id": "INV-1",
    "issue_date": "2026-01-01",
    "currency": "USD",
    "total": 10.0,
}

_CALL_USAGE = {
    "input_tokens": 1000,
    "output_tokens": 100,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
}


@pytest.fixture(autouse=True)
def _default_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")


@pytest.fixture
def llm_project(tmp_path: Path) -> Path:
    repo = Path(__file__).resolve().parents[1]
    dst = tmp_path / "llm_proj"
    shutil.copytree(
        repo / "examples" / "llm_invoice_pipeline",
        dst,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    generate_invoice_texts(3, dst / "data" / "invoices_text", 1)
    return dst


@pytest.fixture
def fake_api(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    calls = {"n": 0}

    def _fake(
        content: str, model: str, system: str, fields_spec: list, **kwargs: object
    ) -> tuple[dict, dict]:
        calls["n"] += 1
        return {**_FIELDS, "invoice_id": f"INV-{calls['n']}"}, dict(_CALL_USAGE)

    monkeypatch.setattr(llm_backend, "_default_call_api", _fake)
    return calls


def test_run_aggregates_usage_per_model(llm_project: Path, fake_api: dict) -> None:
    results = run_project(llm_project)
    r = next(x for x in results if x.model_name == "raw_invoices_llm")

    assert fake_api["n"] == 3
    assert r.metrics["api_calls"] == 3
    assert r.metrics["cache_hits"] == 0
    assert r.metrics["input_tokens"] == 3000
    assert r.metrics["output_tokens"] == 300
    assert "estimated_cost_usd" not in r.metrics  # no pricing configured


def test_cache_hits_counted_with_zero_tokens(
    llm_project: Path, fake_api: dict
) -> None:
    run_project(llm_project)
    # full_refresh bypasses incremental state, so every document is
    # re-extracted — but through the (persisted) LLM response cache.
    results = run_project(llm_project, full_refresh=True)
    r = next(x for x in results if x.model_name == "raw_invoices_llm")

    assert fake_api["n"] == 3, "second run should be all cache hits"
    assert r.metrics["api_calls"] == 0
    assert r.metrics["cache_hits"] == 3
    assert r.metrics["input_tokens"] == 0


def test_pricing_config_yields_cost_estimate(
    llm_project: Path, fake_api: dict
) -> None:
    profiles = llm_project / "profiles.yml"
    profiles.write_text(
        profiles.read_text().replace(
            "        cache_path: ./target/llm_cache.duckdb",
            "        cache_path: ./target/llm_cache.duckdb\n"
            "        pricing:\n"
            "          input_usd_per_mtok: 1.0\n"
            "          output_usd_per_mtok: 5.0\n",
        )
    )

    results = run_project(llm_project)
    r = next(x for x in results if x.model_name == "raw_invoices_llm")
    # 3000 in × $1/M + 300 out × $5/M
    assert r.metrics["estimated_cost_usd"] == pytest.approx(0.0045)


def test_usage_persisted_in_run_results(llm_project: Path, fake_api: dict) -> None:
    results = run_project(llm_project)
    payload = json.loads(write_run_results(llm_project, results).read_text())
    row = next(
        x for x in payload["results"] if x["model_name"] == "raw_invoices_llm"
    )
    assert row["metrics"]["api_calls"] == 3
    assert row["metrics"]["input_tokens"] == 3000


def test_run_summary_prints_usage_line(llm_project: Path, fake_api: dict) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--project-dir", str(llm_project), "run"])
    assert result.exit_code == 0, result.output
    assert "llm: 3 calls, 0 cache hits" in result.output
    assert "3,000 in / 300 out tokens" in result.output


def test_compile_warning_uses_configured_api_key_env(
    monkeypatch: pytest.MonkeyPatch, llm_project: Path
) -> None:
    profiles = llm_project / "profiles.yml"
    profiles.write_text(
        profiles.read_text().replace(
            "api_key_env: ANTHROPIC_API_KEY",
            "api_key_env: DBT_ML_ANTHROPIC_KEY",
        )
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "wrong-default-secret")
    monkeypatch.delenv("DBT_ML_ANTHROPIC_KEY", raising=False)

    result = CliRunner().invoke(
        cli, ["--project-dir", str(llm_project), "compile"]
    )

    assert result.exit_code == 0, result.output
    assert "warning: DBT_ML_ANTHROPIC_KEY is not set" in result.output
    assert "warning: ANTHROPIC_API_KEY is not set" not in result.output
    assert "wrong-default-secret" not in result.output


def test_api_key_secret_is_not_persisted_or_logged(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    llm_project: Path,
    fake_api: dict,
) -> None:
    profiles = llm_project / "profiles.yml"
    profiles.write_text(
        profiles.read_text().replace(
            "api_key_env: ANTHROPIC_API_KEY",
            "api_key_env: DBT_ML_ANTHROPIC_KEY",
        )
    )
    secret = "custom-secret-that-must-never-be-persisted"
    fallback_secret = "wrong-default-secret-that-must-never-be-used"
    monkeypatch.setenv("DBT_ML_ANTHROPIC_KEY", secret)
    monkeypatch.setenv("ANTHROPIC_API_KEY", fallback_secret)

    results = run_project(llm_project)
    write_manifest(llm_project)
    write_run_results(llm_project, results)

    target_files = [p for p in (llm_project / "target").rglob("*") if p.is_file()]
    assert target_files
    for path in target_files:
        payload = path.read_bytes()
        assert secret.encode() not in payload
        assert fallback_secret.encode() not in payload
    assert secret not in caplog.text
    assert fallback_secret not in caplog.text


def test_non_llm_backend_has_empty_metrics(
    tmp_path: Path, example_project_dir: Path
) -> None:
    from dbt_ml.synth import generate_invoices

    dst = tmp_path / "json_proj"
    shutil.copytree(
        example_project_dir,
        dst,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    generate_invoices(3, dst / "data" / "invoices", seed=1)
    results = run_project(dst)
    raw = next(x for x in results if x.model_name == "raw_invoices")
    assert raw.metrics == {}
