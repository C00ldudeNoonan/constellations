"""Bounded, resumable native batches and enforceable budgets (issue #149).

Everything runs against a deterministic fake native-batch provider — no
network, no SDK. Covers partitioning at the provider limit and the
batch_size option, crash/resume without resubmission, artifact-safe
persisted job state, timeout + cancellation, the partial-success policy,
and every budget boundary.
"""
from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, cast

import duckdb
import pytest
from pydantic import ValidationError

from stel.backends import get_backend, llm_backend
from stel.backends.llm_backend import BatchCancelledError, ExtractionResult
from stel.backends.options import LLMBackendOptions
from stel.budget import (
    BudgetExceededError,
    BudgetGuard,
    BudgetLedger,
    LLMBudgetConfig,
)
from stel.config.profile import LLMConfig
from stel.providers import (
    BatchInferenceItem,
    BatchInferenceRequest,
    BatchInferenceResult,
    BatchJobStatus,
    InferenceProvider,
    InferenceRequest,
    InferenceResult,
    ProviderBatchError,
    ProviderCredential,
    ProviderRuntimeOptions,
    ProviderUsage,
    register_inference_provider,
)
from stel.runner import run_project
from stel.synth import generate_invoice_texts

_FIELDS_SPEC = [{"name": "value", "type": "string"}]


@register_inference_provider
class _FakeNativeProvider(InferenceProvider):
    provider_name = "fake-native-batch"
    implementation_version = "1"
    requires_credentials = False
    default_model = "fake-model"
    supports_native_batch = True
    max_batch_requests = 2
    batch_cost_multiplier = 0.5

    jobs: ClassVar[dict[str, list[BatchInferenceRequest]]] = {}
    submissions: ClassVar[list[str]] = []
    cancels: ClassVar[list[str]] = []
    polls: ClassVar[int] = 0
    fail_polls_remaining: ClassVar[int] = 0
    never_done: ClassVar[bool] = False

    @classmethod
    def reset(cls) -> None:
        cls.jobs = {}
        cls.submissions = []
        cls.cancels = []
        cls.polls = 0
        cls.fail_polls_remaining = 0
        cls.never_done = False

    def complete(
        self,
        request: InferenceRequest,
        *,
        credential: ProviderCredential | None,
        runtime: ProviderRuntimeOptions,
    ) -> InferenceResult:
        return InferenceResult(
            {"value": request.content},
            usage=ProviderUsage(input_tokens=10, output_tokens=5),
        )

    def submit_batch(
        self,
        requests: Sequence[BatchInferenceRequest],
        *,
        credential: ProviderCredential | None,
        runtime: ProviderRuntimeOptions,
    ) -> str:
        cls = type(self)
        batch_id = f"job-{len(cls.submissions)}"
        cls.submissions.append(batch_id)
        cls.jobs[batch_id] = list(requests)
        return batch_id

    def poll_batch(
        self,
        batch_id: str,
        *,
        credential: ProviderCredential | None,
        runtime: ProviderRuntimeOptions,
    ) -> BatchJobStatus:
        cls = type(self)
        cls.polls += 1
        if cls.fail_polls_remaining > 0:
            cls.fail_polls_remaining -= 1
            raise ProviderBatchError("simulated poll interruption")
        if cls.never_done:
            return BatchJobStatus(done=False, processing=1)
        return BatchJobStatus(done=True, succeeded=len(cls.jobs[batch_id]))

    def fetch_batch_results(
        self,
        batch_id: str,
        requests: Sequence[BatchInferenceRequest],
        *,
        credential: ProviderCredential | None,
        runtime: ProviderRuntimeOptions,
    ) -> BatchInferenceResult:
        items = tuple(
            BatchInferenceItem(
                request.request_id,
                result=InferenceResult(
                    {"value": request.request.content},
                    usage=ProviderUsage(input_tokens=10, output_tokens=5),
                ),
            )
            for request in requests
        )
        return BatchInferenceResult(items)

    def cancel_batch(
        self,
        batch_id: str,
        *,
        credential: ProviderCredential | None,
        runtime: ProviderRuntimeOptions,
    ) -> None:
        type(self).cancels.append(batch_id)


@pytest.fixture(autouse=True)
def _reset_provider() -> None:
    _FakeNativeProvider.reset()


@pytest.fixture
def docs(tmp_path: Path) -> list[Path]:
    paths = []
    for i in range(5):
        p = tmp_path / f"doc{i}.txt"
        p.write_text(f"INVOICE-{i} body text")
        paths.append(p)
    return paths


def _options(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    options: dict[str, Any] = {
        "provider": "fake-native-batch",
        "fields": _FIELDS_SPEC,
        "cache_path": str(tmp_path / "cache.duckdb"),
        "batch_poll_seconds": 0.1,
        **overrides,
    }
    return options


def _job_rows(cache_path: Path) -> list[tuple[Any, ...]]:
    con = duckdb.connect(str(cache_path), read_only=True)
    try:
        return con.execute(
            "SELECT job_key, provider, batch_id FROM llm_batch_jobs"
        ).fetchall()
    except duckdb.CatalogException:
        return []
    finally:
        con.close()


def test_partitions_respect_provider_limit_deterministically(
    docs: list[Path], tmp_path: Path
) -> None:
    backend = get_backend("llm")
    out = backend.extract_batch_with_metrics(docs, _options(tmp_path))

    assert _FakeNativeProvider.submissions == ["job-0", "job-1", "job-2"]
    partitions = [
        [request.request.content for request in _FakeNativeProvider.jobs[job]]
        for job in _FakeNativeProvider.submissions
    ]
    assert partitions == [
        ["INVOICE-0 body text", "INVOICE-1 body text"],
        ["INVOICE-2 body text", "INVOICE-3 body text"],
        ["INVOICE-4 body text"],
    ]
    assert [cast(ExtractionResult, item).fields["value"] for item in out.items] == [
        f"INVOICE-{i} body text" for i in range(5)
    ]
    assert out.metrics["batch_submissions"] == 3
    assert out.metrics["batches_completed"] == 3
    assert out.metrics["batches_resumed"] == 0

    # Completed jobs leave no persisted records, and a re-run is all cache.
    assert _job_rows(tmp_path / "cache.duckdb") == []
    again = backend.extract_batch_with_metrics(docs, _options(tmp_path))
    assert len(_FakeNativeProvider.submissions) == 3
    assert all(
        cast(ExtractionResult, item).metrics["cache_hits"] == 1 for item in again.items
    )


def test_batch_size_option_bounds_partitions(
    docs: list[Path], tmp_path: Path
) -> None:
    backend = get_backend("llm")
    backend.extract_batch_with_metrics(docs, _options(tmp_path, batch_size=1))
    assert len(_FakeNativeProvider.submissions) == 5
    assert all(
        len(requests) == 1 for requests in _FakeNativeProvider.jobs.values()
    )


def test_interrupted_batch_resumes_without_resubmission(
    docs: list[Path], tmp_path: Path
) -> None:
    backend = get_backend("llm")
    _FakeNativeProvider.fail_polls_remaining = 1

    with pytest.raises(Exception, match="batch inference failed"):
        backend.extract_batch_with_metrics(docs[:2], _options(tmp_path))

    assert _FakeNativeProvider.submissions == ["job-0"]
    rows = _job_rows(tmp_path / "cache.duckdb")
    assert len(rows) == 1
    assert rows[0][1] == "fake-native-batch"
    assert rows[0][2] == "job-0"

    out = backend.extract_batch_with_metrics(docs[:2], _options(tmp_path))

    # The submitted job was billed exactly once: no second submission.
    assert _FakeNativeProvider.submissions == ["job-0"]
    assert out.metrics["batch_submissions"] == 0
    assert out.metrics["batches_resumed"] == 1
    assert [cast(ExtractionResult, item).fields["value"] for item in out.items] == [
        "INVOICE-0 body text",
        "INVOICE-1 body text",
    ]
    assert _job_rows(tmp_path / "cache.duckdb") == []


def test_persisted_job_state_is_artifact_safe(
    docs: list[Path], tmp_path: Path
) -> None:
    backend = get_backend("llm")
    _FakeNativeProvider.fail_polls_remaining = 1
    with pytest.raises(Exception, match="batch inference failed"):
        backend.extract_batch_with_metrics(docs[:2], _options(tmp_path))

    cache_path = tmp_path / "cache.duckdb"
    con = duckdb.connect(str(cache_path), read_only=True)
    try:
        columns = [
            row[0]
            for row in con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'llm_batch_jobs'"
            ).fetchall()
        ]
        values = con.execute("SELECT * FROM llm_batch_jobs").fetchall()
    finally:
        con.close()

    assert sorted(columns) == ["batch_id", "created_at", "job_key", "provider"]
    serialized = repr(values)
    assert "INVOICE" not in serialized  # no document content
    assert "extract" not in serialized  # no prompt or schema text


def test_timeout_cancels_job_and_raises(
    docs: list[Path], tmp_path: Path
) -> None:
    backend = get_backend("llm")
    _FakeNativeProvider.never_done = True

    with pytest.raises(BatchCancelledError, match="batch_timeout_seconds"):
        backend.extract_batch_with_metrics(
            docs[:2], _options(tmp_path, batch_timeout_seconds=0.001)
        )

    assert _FakeNativeProvider.cancels == ["job-0"]
    assert _job_rows(tmp_path / "cache.duckdb") == []


def test_budget_stops_before_batch_submission(
    docs: list[Path], tmp_path: Path
) -> None:
    guard = BudgetGuard(
        BudgetLedger(LLMBudgetConfig(max_api_calls=1), scope="model 'm'"),
        None,
    )
    backend = get_backend("llm")

    with pytest.raises(BudgetExceededError, match="max_api_calls"):
        backend.extract_batch_with_metrics(
            docs[:3], _options(tmp_path), budget=guard
        )

    # The first partition needed 2 calls against a cap of 1: exhaustion
    # fired before submission, so nothing was ever billed.
    assert _FakeNativeProvider.submissions == []


def test_file_bytes_budget_blocks_oversized_document(
    docs: list[Path], tmp_path: Path
) -> None:
    guard = BudgetGuard(
        BudgetLedger(LLMBudgetConfig(max_file_bytes=4), scope="model 'm'"),
        None,
    )
    backend = get_backend("llm")
    with pytest.raises(BudgetExceededError, match="max_file_bytes"):
        backend.extract_batch_with_metrics(
            docs[:1], _options(tmp_path), budget=guard
        )
    assert _FakeNativeProvider.submissions == []


def test_total_bytes_budget_accumulates_submitted_documents(
    docs: list[Path], tmp_path: Path
) -> None:
    size = docs[0].stat().st_size
    guard = BudgetGuard(
        BudgetLedger(
            LLMBudgetConfig(max_total_bytes=size + 1), scope="model 'm'"
        ),
        None,
    )
    backend = get_backend("llm")
    with pytest.raises(BudgetExceededError, match="max_total_bytes"):
        backend.extract_batch_with_metrics(
            docs[:3], _options(tmp_path), budget=guard
        )
    assert _FakeNativeProvider.submissions == []


class TestBudgetLedger:
    def test_documents_boundary(self) -> None:
        ledger = BudgetLedger(LLMBudgetConfig(max_documents=2), scope="run")
        ledger.charge_documents(2)
        with pytest.raises(BudgetExceededError, match="max_documents"):
            ledger.charge_documents(1)

    def test_api_calls_boundary_blocks_next_call(self) -> None:
        ledger = BudgetLedger(LLMBudgetConfig(max_api_calls=2), scope="run")
        ledger.ensure_headroom(next_calls=2)
        ledger.charge_usage(api_calls=2)
        with pytest.raises(BudgetExceededError, match="max_api_calls"):
            ledger.ensure_headroom()

    def test_token_boundaries_stop_after_overrun(self) -> None:
        ledger = BudgetLedger(
            LLMBudgetConfig(max_input_tokens=100, max_output_tokens=50),
            scope="run",
        )
        ledger.charge_usage(input_tokens=99, output_tokens=49)
        ledger.ensure_headroom()
        ledger.charge_usage(input_tokens=1)
        with pytest.raises(BudgetExceededError, match="max_input_tokens"):
            ledger.ensure_headroom()

    def test_output_token_boundary(self) -> None:
        ledger = BudgetLedger(LLMBudgetConfig(max_output_tokens=10), scope="run")
        ledger.charge_usage(output_tokens=10)
        with pytest.raises(BudgetExceededError, match="max_output_tokens"):
            ledger.ensure_headroom()

    def test_cost_boundary(self) -> None:
        ledger = BudgetLedger(LLMBudgetConfig(max_cost_usd=0.5), scope="run")
        ledger.charge_usage(cost_usd=0.49)
        ledger.ensure_headroom()
        ledger.charge_usage(cost_usd=0.01)
        with pytest.raises(BudgetExceededError, match="max_cost_usd"):
            ledger.ensure_headroom()

    def test_run_scope_is_shared_across_models(self) -> None:
        run_ledger = BudgetLedger(LLMBudgetConfig(max_api_calls=3), scope="run")
        first = BudgetGuard(None, run_ledger)
        second = BudgetGuard(None, run_ledger)
        first.charge_usage(api_calls=2)
        second.ensure_headroom()  # 2 + 1 <= 3
        second.charge_usage(api_calls=1)
        with pytest.raises(BudgetExceededError, match="max_api_calls"):
            first.ensure_headroom()

    def test_model_and_run_scopes_both_enforced(self) -> None:
        guard = BudgetGuard(
            BudgetLedger(LLMBudgetConfig(max_api_calls=1), scope="model 'm'"),
            BudgetLedger(LLMBudgetConfig(max_api_calls=10), scope="run"),
        )
        guard.charge_usage(api_calls=1)
        with pytest.raises(BudgetExceededError, match="model 'm'"):
            guard.ensure_headroom()

    def test_charge_metrics_prefers_reported_cost(self) -> None:
        charged: list[float] = []

        guard = BudgetGuard(
            BudgetLedger(LLMBudgetConfig(max_cost_usd=1.0), scope="model 'm'"),
            None,
            cost_estimator=lambda metrics: charged.append(-1.0) or 99.0,
        )
        guard.charge_metrics(
            {"api_calls": 1, "input_tokens": 5, "reported_cost_usd": 0.25}
        )
        # Estimator not consulted when the provider reported spend.
        assert charged == []
        with pytest.raises(BudgetExceededError):
            guard.charge_metrics({"api_calls": 1})  # estimator says 99.0
            guard.ensure_headroom()

    def test_error_text_is_artifact_safe(self) -> None:
        ledger = BudgetLedger(LLMBudgetConfig(max_documents=1), scope="run")
        with pytest.raises(BudgetExceededError) as exc_info:
            ledger.charge_documents(2)
        assert str(exc_info.value) == (
            "run budget exceeded: max_documents=2 over cap 1"
        )


def test_budget_config_is_strictly_typed() -> None:
    with pytest.raises(ValidationError):
        LLMBudgetConfig(max_documents=0)
    with pytest.raises(ValidationError):
        LLMBudgetConfig(max_cost_usd=-1.0)
    # Deliberately invalid kwargs (a dict[str, Any] keeps the static checker from
    # rejecting them before the runtime ValidationError we are asserting).
    bad_kwargs: dict[str, Any] = {"unknown_cap": 5}
    with pytest.raises(ValidationError):
        LLMBudgetConfig(**bad_kwargs)
    bad_kwargs = {"max_api_calls": "10"}
    with pytest.raises(ValidationError):
        LLMBudgetConfig(**bad_kwargs)


def test_llm_backend_options_validate_batch_safety_fields() -> None:
    base: dict[str, Any] = {"fields": _FIELDS_SPEC}
    parsed = LLMBackendOptions.model_validate(
        {**base, "budget": {"max_api_calls": 3}, "batch_size": 10}
    )
    assert parsed.budget is not None and parsed.budget.max_api_calls == 3
    assert parsed.on_partial_batch == "fail"

    with pytest.raises(ValidationError):
        LLMBackendOptions.model_validate(
            {**base, "on_partial_batch": "sometimes"}
        )
    with pytest.raises(ValidationError):
        LLMBackendOptions.model_validate(
            {**base, "batch_poll_seconds": 60.0, "batch_poll_max_seconds": 30.0}
        )


def test_profile_llm_budget_parses() -> None:
    config = LLMConfig(budget={"max_cost_usd": 1.5, "max_api_calls": 100})
    assert config.budget is not None
    assert config.budget.max_cost_usd == 1.5
    with pytest.raises(ValidationError):
        LLMConfig(budget={"max_documents": -1})


@pytest.fixture
def invoice_project(tmp_path: Path) -> Path:
    repo = Path(__file__).resolve().parents[1]
    dst = tmp_path / "budget_proj"
    shutil.copytree(
        repo / "examples" / "llm_invoice_pipeline",
        dst,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    generate_invoice_texts(3, dst / "data" / "invoices_text", 1)
    return dst


@pytest.fixture(autouse=True)
def _default_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")


def _add_model_options(project: Path, lines: str) -> None:
    model = project / "models" / "raw_invoices_llm.yml"
    model.write_text(
        model.read_text().replace("      options:", f"      options:\n{lines}", 1)
    )


def test_runner_budget_exceeded_is_distinct_and_publishes_nothing(
    monkeypatch: pytest.MonkeyPatch, invoice_project: Path
) -> None:
    _add_model_options(invoice_project, "        budget:\n          max_documents: 1")

    def _never(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("provider must not be called after budget stop")

    monkeypatch.setattr(llm_backend, "_default_call_api", _never)

    results = run_project(invoice_project)

    (result,) = [r for r in results if r.model_name == "raw_invoices_llm"]
    assert result.status == "budget_exceeded"
    assert result.rows_written == 0
    assert any("max_documents" in error for error in result.errors)


def test_runner_api_call_budget_stops_before_next_call(
    monkeypatch: pytest.MonkeyPatch, invoice_project: Path
) -> None:
    _add_model_options(
        invoice_project, "        budget:\n          max_api_calls: 1"
    )
    calls = {"n": 0}

    def fake_call(
        content: str,
        model: str,
        system: str,
        fields_spec: list[dict[str, Any]],
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        calls["n"] += 1
        return (
            {
                "vendor": "V",
                "invoice_id": "INV-1",
                "issue_date": "2026-01-01",
                "currency": "USD",
                "total": 1.0,
            },
            {"input_tokens": 10, "output_tokens": 5},
        )

    monkeypatch.setattr(llm_backend.LLMBackend, "_call_api", staticmethod(fake_call))

    results = run_project(invoice_project)

    (result,) = [r for r in results if r.model_name == "raw_invoices_llm"]
    assert result.status == "budget_exceeded"
    assert calls["n"] == 1  # the cap allowed exactly one call, never a second
    assert any("max_api_calls" in error for error in result.errors)


def test_runner_cancelled_batch_has_distinct_status(
    monkeypatch: pytest.MonkeyPatch, invoice_project: Path
) -> None:
    _add_model_options(invoice_project, "        batch: true")

    def cancelled(*args: Any, **kwargs: Any) -> None:
        raise BatchCancelledError(
            "provider batch exceeded batch_timeout_seconds=1 and was cancelled"
        )

    monkeypatch.setattr(llm_backend, "_run_message_batch", cancelled)

    results = run_project(invoice_project)

    (result,) = [r for r in results if r.model_name == "raw_invoices_llm"]
    assert result.status == "cancelled"
    assert result.rows_written == 0
    assert any("batch_timeout_seconds" in error for error in result.errors)


def test_runner_partial_batch_failure_publishes_nothing_by_default(
    monkeypatch: pytest.MonkeyPatch, invoice_project: Path
) -> None:
    from stel.providers import ProviderRequestError
    from stel.runner import RunError

    _add_model_options(invoice_project, "        batch: true")

    def fake(
        requests: Sequence[BatchInferenceRequest],
        **kwargs: Any,
    ) -> tuple[BatchInferenceResult, bool]:
        items = [
            BatchInferenceItem(
                request.request_id,
                result=InferenceResult(
                    {
                        "vendor": "V",
                        "invoice_id": "INV-1",
                        "issue_date": "2026-01-01",
                        "currency": "USD",
                        "total": 1.0,
                    },
                    usage=ProviderUsage(input_tokens=10, output_tokens=5),
                ),
            )
            for request in requests
        ]
        items[0] = BatchInferenceItem(
            items[0].request_id,
            error=ProviderRequestError("anthropic", "batch item", code="errored"),
        )
        return BatchInferenceResult(tuple(items), batch_submissions=1), False

    monkeypatch.setattr(llm_backend, "_run_message_batch", fake)

    with pytest.raises(RunError, match="on_partial_batch=fail"):
        run_project(invoice_project)
