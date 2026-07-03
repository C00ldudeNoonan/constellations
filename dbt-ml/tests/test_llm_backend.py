from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dbt_ml.backends import get_backend
from dbt_ml.backends.llm_backend import LLMBackend


class _CallCounter:
    """Stand-in for the LLM API: records calls and returns canned responses.

    Used via monkeypatch on the unbound _call_api function — Python doesn't
    auto-bind `self` for callable instances, so the signature here has no
    leading `self` for the backend instance.
    """

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls = 0
        self.kwargs: dict[str, Any] = {}

    def __call__(
        self,
        content: str,
        model: str,
        system: str,
        fields_spec: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls += 1
        self.kwargs = kwargs
        return dict(self.response)


@pytest.fixture
def doc(tmp_path: Path) -> Path:
    p = tmp_path / "doc.txt"
    p.write_text(
        "INVOICE\nFrom: Acme\nInvoice number: INV-00001\nTotal due: USD 99.99\n"
    )
    return p


@pytest.fixture
def schema() -> list[dict[str, Any]]:
    return [
        {"name": "vendor", "type": "string"},
        {"name": "invoice_id", "type": "string"},
        {"name": "total", "type": "number"},
    ]


def test_llm_backend_registered() -> None:
    backend = get_backend("llm")
    assert backend.name() == "llm"
    assert ".txt" in backend.supported_formats()


def test_llm_backend_calls_api_on_miss(
    monkeypatch: pytest.MonkeyPatch, doc: Path, schema: list[dict[str, Any]], tmp_path: Path
) -> None:
    counter = _CallCounter({"vendor": "Acme", "invoice_id": "INV-00001", "total": 99.99})
    monkeypatch.setattr(LLMBackend, "_call_api", counter)

    backend = get_backend("llm")
    result = backend.extract(
        doc,
        {
            "cache_path": str(tmp_path / "cache.duckdb"),
            "fields": schema,
        },
    )
    assert result.fields == {"vendor": "Acme", "invoice_id": "INV-00001", "total": 99.99}
    assert counter.calls == 1


def test_llm_backend_uses_cache_on_repeat(
    monkeypatch: pytest.MonkeyPatch, doc: Path, schema: list[dict[str, Any]], tmp_path: Path
) -> None:
    counter = _CallCounter({"vendor": "Acme", "invoice_id": "X", "total": 1.0})
    monkeypatch.setattr(LLMBackend, "_call_api", counter)
    cache = tmp_path / "cache.duckdb"

    backend = get_backend("llm")
    opts = {"cache_path": str(cache), "fields": schema}
    backend.extract(doc, opts)
    backend.extract(doc, opts)
    backend.extract(doc, opts)
    assert counter.calls == 1, "subsequent calls should hit the cache"


def test_llm_backend_recalls_when_content_changes(
    monkeypatch: pytest.MonkeyPatch, doc: Path, schema: list[dict[str, Any]], tmp_path: Path
) -> None:
    counter = _CallCounter({"vendor": "v", "invoice_id": "id", "total": 0.0})
    monkeypatch.setattr(LLMBackend, "_call_api", counter)
    cache = tmp_path / "cache.duckdb"

    backend = get_backend("llm")
    opts = {"cache_path": str(cache), "fields": schema}
    backend.extract(doc, opts)
    doc.write_text("DIFFERENT INVOICE BODY")
    backend.extract(doc, opts)
    assert counter.calls == 2


def test_llm_backend_recalls_when_schema_changes(
    monkeypatch: pytest.MonkeyPatch, doc: Path, schema: list[dict[str, Any]], tmp_path: Path
) -> None:
    counter = _CallCounter({"vendor": "v", "invoice_id": "id", "total": 0.0})
    monkeypatch.setattr(LLMBackend, "_call_api", counter)
    cache = tmp_path / "cache.duckdb"

    backend = get_backend("llm")
    backend.extract(doc, {"cache_path": str(cache), "fields": schema})
    new_schema = [*schema, {"name": "currency", "type": "string"}]
    backend.extract(doc, {"cache_path": str(cache), "fields": new_schema})
    assert counter.calls == 2


def test_llm_backend_requires_fields(doc: Path) -> None:
    backend = get_backend("llm")
    with pytest.raises(ValueError, match=r"options\.fields"):
        backend.extract(doc, {})


def test_llm_pipeline_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Run the LLM example project through the full runner, with the API mocked."""
    import shutil

    import duckdb

    from dbt_ml.runner import run_project
    from dbt_ml.synth import generate_invoice_texts

    repo = Path(__file__).resolve().parents[1]
    src_example = repo / "examples" / "llm_invoice_pipeline"
    project = tmp_path / "proj"
    shutil.copytree(
        src_example,
        project,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    generate_invoice_texts(3, project / "data" / "invoices_text", seed=1)

    canned = {
        "vendor": "ACME Corp",
        "invoice_id": "INV-MOCKED",
        "issue_date": "2026-04-01",
        "currency": "USD",
        "total": 123.45,
    }

    def fake(
        self: LLMBackend, content: str, model: str, system: str, fields_spec: list,
        **kwargs: Any,
    ) -> dict:
        return dict(canned)

    monkeypatch.setattr(LLMBackend, "_call_api", fake)

    results = run_project(project)
    assert {r.model_name for r in results} == {"raw_invoices_llm"}
    raw = results[0]
    assert raw.documents_processed == 3
    assert raw.rows_written == 3

    db = project / "target" / "dbt_ml.duckdb"
    con = duckdb.connect(str(db), read_only=True)
    try:
        rows = con.execute(
            'SELECT vendor, total FROM "dbt_ml"."llm_invoices".raw_invoices_llm'
        ).fetchall()
    finally:
        con.close()
    assert len(rows) == 3
    assert rows[0] == ("ACME Corp", 123.45)


def test_llm_backend_no_api_key_raises(
    monkeypatch: pytest.MonkeyPatch, doc: Path, schema: list[dict[str, Any]], tmp_path: Path
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    backend = get_backend("llm")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        backend.extract(
            doc, {"cache_path": str(tmp_path / "c.duckdb"), "fields": schema}
        )


def test_generation_params_default_and_override(
    monkeypatch: pytest.MonkeyPatch, doc: Path, schema: list[dict[str, Any]]
) -> None:
    counter = _CallCounter({"vendor": "v", "invoice_id": "i", "total": 0.0})
    monkeypatch.setattr(LLMBackend, "_call_api", counter)
    backend = get_backend("llm")

    backend.extract(doc, {"fields": schema})
    assert counter.kwargs == {"temperature": 0.0, "max_tokens": 2048, "max_retries": 4}

    backend.extract(
        doc,
        {"fields": schema, "temperature": 0.3, "max_tokens": 8192, "max_retries": 1},
    )
    assert counter.kwargs == {"temperature": 0.3, "max_tokens": 8192, "max_retries": 1}


def test_temperature_is_part_of_cache_key(
    monkeypatch: pytest.MonkeyPatch, doc: Path, schema: list[dict[str, Any]], tmp_path: Path
) -> None:
    counter = _CallCounter({"vendor": "v", "invoice_id": "i", "total": 0.0})
    monkeypatch.setattr(LLMBackend, "_call_api", counter)
    backend = get_backend("llm")

    opts = {"cache_path": str(tmp_path / "cache.duckdb"), "fields": schema}
    backend.extract(doc, opts)
    backend.extract(doc, {**opts, "temperature": 0.7})
    assert counter.calls == 2, "different temperature must not reuse cached output"
    backend.extract(doc, {**opts, "temperature": 0.7})
    assert counter.calls == 2


def test_truncated_response_is_an_error(
    monkeypatch: pytest.MonkeyPatch, doc: Path, schema: list[dict[str, Any]]
) -> None:
    """A max_tokens-truncated response raises instead of yielding partial data;
    the client is constructed with the retry budget and called with temperature 0
    and a plain-string system prompt."""
    init_kwargs: dict[str, Any] = {}
    create_kwargs: dict[str, Any] = {}

    class _FakeResp:
        def __init__(self) -> None:
            self.stop_reason = "max_tokens"
            self.content: list[Any] = []

    class _FakeMessages:
        def create(self, **kwargs: Any) -> _FakeResp:
            create_kwargs.update(kwargs)
            return _FakeResp()

    class _FakeAnthropic:
        def __init__(self, **kwargs: Any) -> None:
            init_kwargs.update(kwargs)
            self.messages = _FakeMessages()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("anthropic.Anthropic", _FakeAnthropic)

    backend = get_backend("llm")
    with pytest.raises(RuntimeError, match="max_tokens"):
        backend.extract(doc, {"fields": schema})

    assert init_kwargs == {"max_retries": 4}
    assert create_kwargs["temperature"] == 0.0
    assert create_kwargs["max_tokens"] == 2048
    assert isinstance(create_kwargs["system"], str)


def test_max_concurrent_gates_api_calls() -> None:
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from dbt_ml.backends.llm_backend import extract_fields_from_text

    active = 0
    peak = 0
    lock = threading.Lock()

    def fake(
        content: str, model: str, system: str, fields_spec: list, **kwargs: Any
    ) -> dict[str, Any]:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return {"x": 1}

    def one(text: str) -> dict[str, Any]:
        return extract_fields_from_text(
            text, fields_spec=[{"name": "x"}], call_api=fake, max_concurrent=1
        )

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(one, [f"doc {i}" for i in range(8)]))

    assert all(r == {"x": 1} for r in results)
    assert peak == 1, "max_concurrent=1 must serialize API calls"
