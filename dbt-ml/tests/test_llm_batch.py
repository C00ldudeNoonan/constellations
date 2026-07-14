"""Batch API mode for LLM extraction (issue #75, part 2).

Exercises BaseBackend.extract_batch's sequential default, the LLM backend's
Message Batches integration (with the API faked), and the runner's batch
dispatch end-to-end on examples/llm_invoice_pipeline.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from dbt_ml.backends import get_backend, llm_backend
from dbt_ml.backends.base import ExtractionResult
from dbt_ml.providers import (
    BatchInferenceItem,
    BatchInferenceRequest,
    BatchInferenceResult,
    InferenceRequest,
    InferenceResult,
    ProviderRequestError,
    ProviderResponseError,
    ProviderUsage,
)
from dbt_ml.runner import run_project
from dbt_ml.synth import generate_invoice_texts

_SCHEMA = [{"name": "vendor", "type": "string"}, {"name": "total", "type": "number"}]


@pytest.fixture(autouse=True)
def _default_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")


def _succeeded_result(
    fields: dict[str, Any],
    *,
    input_tokens: int = 100,
    output_tokens: int = 10,
) -> InferenceResult:
    return InferenceResult(
        fields,
        usage=ProviderUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


def _errored_item(request_id: str = "req-1") -> BatchInferenceItem:
    return BatchInferenceItem(
        request_id,
        error=ProviderRequestError("anthropic", "batch item", code="errored"),
    )


class _FakeBatchAPI:
    """Stands in for _run_message_batch: echoes per-request fields derived from
    the document text, building the result dict in reverse order to prove
    custom_id keying (results stream back unordered in the real API)."""

    def __init__(
        self, overrides: dict[str, BatchInferenceItem] | None = None
    ) -> None:
        self.calls = 0
        self.submitted: list[list[BatchInferenceRequest]] = []
        self.overrides = overrides or {}

    def __call__(
        self,
        requests: list[BatchInferenceRequest],
        *,
        provider: str,
        poll_seconds: float,
        api_key_env: str,
        max_retries: int,
    ) -> BatchInferenceResult:
        self.calls += 1
        self.submitted.append(requests)
        out: list[BatchInferenceItem] = []
        for req in reversed(requests):
            cid = req.request_id
            if cid in self.overrides:
                out.append(self.overrides[cid])
                continue
            text = req.request.content
            out.append(
                BatchInferenceItem(
                    cid,
                    result=_succeeded_result(
                        {"vendor": f"v-{text[:8]}", "total": 1.0}
                    ),
                )
            )
        return BatchInferenceResult(tuple(out), batch_submissions=1)


@pytest.fixture
def docs(tmp_path: Path) -> list[Path]:
    paths = []
    for i in range(3):
        p = tmp_path / f"doc{i}.txt"
        p.write_text(f"INVOICE-{i} body text")
        paths.append(p)
    return paths


def test_base_default_extract_batch_loops_and_isolates_errors(tmp_path: Path) -> None:
    backend = get_backend("json")
    good = tmp_path / "good.json"
    good.write_text('{"a": 1}')
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json")

    out = backend.extract_batch([good, bad], {"fields": ["a"]})
    assert isinstance(out[0], ExtractionResult)
    assert out[0].fields == {"a": 1}
    assert isinstance(out[1], Exception)


def test_llm_batch_maps_unordered_results_by_custom_id(
    monkeypatch: pytest.MonkeyPatch, docs: list[Path], tmp_path: Path
) -> None:
    fake = _FakeBatchAPI()
    monkeypatch.setattr(llm_backend, "_run_message_batch", fake)

    backend = get_backend("llm")
    out = backend.extract_batch(
        docs, {"fields": _SCHEMA, "cache_path": str(tmp_path / "c.duckdb")}
    )

    assert fake.calls == 1
    assert len(fake.submitted[0]) == 3
    for path, res in zip(docs, out, strict=True):
        assert isinstance(res, ExtractionResult)
        # Each doc got the fields derived from its own text, not another's.
        assert res.fields["vendor"] == f"v-{path.read_text()[:8]}"
        assert res.metrics["api_calls"] == 1
        assert res.metrics["input_tokens"] == 100


def test_llm_batch_skips_cached_documents(
    monkeypatch: pytest.MonkeyPatch, docs: list[Path], tmp_path: Path
) -> None:
    fake = _FakeBatchAPI()
    monkeypatch.setattr(llm_backend, "_run_message_batch", fake)
    backend = get_backend("llm")
    opts = {"fields": _SCHEMA, "cache_path": str(tmp_path / "c.duckdb")}

    backend.extract_batch(docs, opts)
    out = backend.extract_batch(docs, opts)

    assert fake.calls == 1, "second call should be fully cache-served"
    for res in out:
        assert isinstance(res, ExtractionResult)
        assert res.metrics == {
            "api_calls": 0,
            "cache_hits": 1,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }


def test_llm_batch_errored_item_isolated(
    monkeypatch: pytest.MonkeyPatch, docs: list[Path], tmp_path: Path
) -> None:
    fake = _FakeBatchAPI(overrides={"req-1": _errored_item()})
    monkeypatch.setattr(llm_backend, "_run_message_batch", fake)

    backend = get_backend("llm")
    out = backend.extract_batch(
        docs, {"fields": _SCHEMA, "cache_path": str(tmp_path / "c.duckdb")}
    )

    assert isinstance(out[0], ExtractionResult)
    assert isinstance(out[1], Exception)
    assert "errored" in str(out[1])
    assert isinstance(out[2], ExtractionResult)


def test_llm_batch_max_tokens_item_is_error(
    monkeypatch: pytest.MonkeyPatch, docs: list[Path], tmp_path: Path
) -> None:
    truncated = BatchInferenceItem(
        "req-0",
        error=ProviderResponseError("LLM response truncated at max_tokens=2048"),
    )
    fake = _FakeBatchAPI(overrides={"req-0": truncated})
    monkeypatch.setattr(llm_backend, "_run_message_batch", fake)

    backend = get_backend("llm")
    out = backend.extract_batch(
        docs, {"fields": _SCHEMA, "cache_path": str(tmp_path / "c.duckdb")}
    )
    assert isinstance(out[0], Exception)
    assert "max_tokens" in str(out[0])


def test_message_batch_uses_custom_api_key_env(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    init_kwargs: dict[str, Any] = {}
    submitted: list[list[dict[str, Any]]] = []

    class _FakeBatches:
        def create(self, *, requests: list[dict[str, Any]]) -> SimpleNamespace:
            submitted.append(requests)
            return SimpleNamespace(id="batch-1")

        def retrieve(self, batch_id: str) -> SimpleNamespace:
            assert batch_id == "batch-1"
            return SimpleNamespace(id=batch_id, processing_status="ended")

        def results(self, batch_id: str) -> list[Any]:
            assert batch_id == "batch-1"
            return []

    class _FakeAnthropic:
        def __init__(self, **kwargs: Any) -> None:
            init_kwargs.update(kwargs)
            self.messages = SimpleNamespace(batches=_FakeBatches())

    secret = "batch-secret-that-must-not-leak"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "wrong-default-secret")
    monkeypatch.setenv("DBT_ML_BATCH_KEY", secret)
    monkeypatch.setattr("anthropic.Anthropic", _FakeAnthropic)

    request = BatchInferenceRequest(
        "req-0",
        InferenceRequest(
            model="claude-haiku-4-5",
            content="invoice",
            system_prompt="extract",
            output_schema={"type": "object", "properties": {}},
        ),
    )
    result = llm_backend._run_message_batch(
        [request],
        poll_seconds=0,
        api_key_env="DBT_ML_BATCH_KEY",
    )

    assert len(result.items) == 1
    assert result.items[0].request_id == "req-0"
    assert result.items[0].error is not None
    assert "missing_result" in str(result.items[0].error)
    assert result.batch_submissions == 1
    assert init_kwargs == {"api_key": secret, "max_retries": 4}
    assert len(submitted) == 1
    assert secret not in caplog.text


def test_message_batch_missing_custom_key_fails_before_client_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed = False

    class _FakeAnthropic:
        def __init__(self, **kwargs: Any) -> None:
            nonlocal constructed
            constructed = True

    fallback_secret = "wrong-default-secret"
    monkeypatch.setenv("ANTHROPIC_API_KEY", fallback_secret)
    monkeypatch.delenv("DBT_ML_BATCH_KEY", raising=False)
    monkeypatch.setattr("anthropic.Anthropic", _FakeAnthropic)

    with pytest.raises(RuntimeError, match="DBT_ML_BATCH_KEY") as exc_info:
        llm_backend._run_message_batch(
            [
                BatchInferenceRequest(
                    "req-0",
                    InferenceRequest(
                        model="claude-haiku-4-5",
                        content="invoice",
                        system_prompt="extract",
                        output_schema={"type": "object", "properties": {}},
                    ),
                )
            ],
            poll_seconds=0,
            api_key_env="DBT_ML_BATCH_KEY",
        )

    assert not constructed
    assert fallback_secret not in str(exc_info.value)


# ─── runner end-to-end ─────────────────────────────────────────────────────


@pytest.fixture
def batch_project(tmp_path: Path) -> Path:
    repo = Path(__file__).resolve().parents[1]
    dst = tmp_path / "batch_proj"
    shutil.copytree(
        repo / "examples" / "llm_invoice_pipeline",
        dst,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    model = dst / "models" / "raw_invoices_llm.yml"
    model.write_text(
        model.read_text().replace("      options:", "      options:\n        batch: true", 1)
    )
    generate_invoice_texts(3, dst / "data" / "invoices_text", 1)
    return dst


def _invoice_result(n: int) -> InferenceResult:
    return _succeeded_result(
        {
            "vendor": "Batched Vendor",
            "invoice_id": f"INV-{n}",
            "issue_date": "2026-01-01",
            "currency": "USD",
            "total": 10.0 * n,
        },
        input_tokens=1000,
        output_tokens=100,
    )


def test_runner_batch_mode_end_to_end(
    monkeypatch: pytest.MonkeyPatch, batch_project: Path
) -> None:
    calls = {"n": 0}

    def fake(
        requests: list[BatchInferenceRequest],
        *,
        provider: str,
        poll_seconds: float,
        api_key_env: str,
        max_retries: int,
    ) -> BatchInferenceResult:
        calls["n"] += 1
        return BatchInferenceResult(
            tuple(
                BatchInferenceItem(req.request_id, result=_invoice_result(i))
                for i, req in enumerate(requests)
            ),
            batch_submissions=1,
        )

    monkeypatch.setattr(llm_backend, "_run_message_batch", fake)

    results = run_project(batch_project)
    r = next(x for x in results if x.model_name == "raw_invoices_llm")

    assert calls["n"] == 1, "one batch submission for the whole model"
    assert r.rows_written == 3
    assert not r.errors
    assert r.metrics["batch"] is True
    assert r.metrics["api_calls"] == 3
    assert r.metrics["input_tokens"] == 3000

    # Second run: incremental state skips everything; no new batch.
    results2 = run_project(batch_project)
    r2 = next(x for x in results2 if x.model_name == "raw_invoices_llm")
    assert calls["n"] == 1
    assert r2.documents_skipped == 3


def test_runner_batch_cost_estimate_halved(
    monkeypatch: pytest.MonkeyPatch, batch_project: Path
) -> None:
    profiles = batch_project / "profiles.yml"
    profiles.write_text(
        profiles.read_text().replace(
            "        cache_path: ./target/llm_cache.duckdb",
            "        cache_path: ./target/llm_cache.duckdb\n"
            "        pricing:\n"
            "          input_usd_per_mtok: 1.0\n"
            "          output_usd_per_mtok: 5.0\n",
        )
    )

    def fake(
        requests: list[BatchInferenceRequest],
        *,
        provider: str,
        poll_seconds: float,
        api_key_env: str,
        max_retries: int,
    ) -> BatchInferenceResult:
        return BatchInferenceResult(
            tuple(
                BatchInferenceItem(req.request_id, result=_invoice_result(i))
                for i, req in enumerate(requests)
            ),
            batch_submissions=1,
        )

    monkeypatch.setattr(llm_backend, "_run_message_batch", fake)

    results = run_project(batch_project)
    r = next(x for x in results if x.model_name == "raw_invoices_llm")
    # Sync-equivalent: (3000 × $1/M + 300 × $5/M) = 0.0045 → batch bills half.
    assert r.metrics["estimated_cost_usd"] == pytest.approx(0.00225)


def test_runner_batch_isolates_per_document_errors(
    monkeypatch: pytest.MonkeyPatch, batch_project: Path
) -> None:
    def fake(
        requests: list[BatchInferenceRequest],
        *,
        provider: str,
        poll_seconds: float,
        api_key_env: str,
        max_retries: int,
    ) -> BatchInferenceResult:
        out = [
            BatchInferenceItem(req.request_id, result=_invoice_result(i))
            for i, req in enumerate(requests)
        ]
        out[1] = _errored_item()
        return BatchInferenceResult(tuple(out), batch_submissions=1)

    monkeypatch.setattr(llm_backend, "_run_message_batch", fake)

    results = run_project(batch_project)
    r = next(x for x in results if x.model_name == "raw_invoices_llm")
    assert r.rows_written == 2
    assert len(r.errors) == 1
    assert "errored" in r.errors[0]


def test_runner_records_batch_submission_when_every_item_fails(
    monkeypatch: pytest.MonkeyPatch, batch_project: Path
) -> None:
    def fake(
        requests: list[BatchInferenceRequest],
        *,
        provider: str,
        poll_seconds: float,
        api_key_env: str,
        max_retries: int,
    ) -> BatchInferenceResult:
        del provider, poll_seconds, api_key_env, max_retries
        return BatchInferenceResult(
            tuple(_errored_item(request.request_id) for request in requests),
            batch_submissions=1,
        )

    monkeypatch.setattr(llm_backend, "_run_message_batch", fake)

    results = run_project(batch_project)
    result = next(x for x in results if x.model_name == "raw_invoices_llm")

    assert len(result.errors) == 3
    assert result.metrics["batch_submissions"] == 1
