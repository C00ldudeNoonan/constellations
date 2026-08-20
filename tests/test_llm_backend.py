from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from stel.backends import get_backend, llm_backend
from stel.backends.llm_backend import LLMBackend, extract_fields_from_text
from stel.credentials import CredentialReference
from stel.hashing import HASH_DIGEST_SIZE
from stel.providers import base as provider_base
from stel.providers import get_inference_provider


@pytest.fixture(autouse=True)
def _default_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")


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

    from stel.runner import run_project
    from stel.synth import generate_invoice_texts

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
        self: Any, content: str, model: str, system: str,
        fields_spec: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        return dict(canned)

    monkeypatch.setattr(LLMBackend, "_call_api", fake)

    results = run_project(project)
    assert {r.model_name for r in results} == {"raw_invoices_llm"}
    raw = results[0]
    assert raw.documents_processed == 3
    assert raw.rows_written == 3

    db = project / "target" / "stel.duckdb"
    con = duckdb.connect(str(db), read_only=True)
    try:
        rows = con.execute(
            'SELECT vendor, total FROM "stel"."llm_invoices".raw_invoices_llm'
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
    unread = tmp_path / "not-read.txt"
    with pytest.raises(RuntimeError, match="credential environment variable") as exc_info:
        backend.extract(
            unread, {"cache_path": str(tmp_path / "c.duckdb"), "fields": schema}
        )
    assert "test-api-key" not in str(exc_info.value)
    assert "ANTHROPIC_API_KEY" not in str(exc_info.value)


def test_invalid_numeric_options_fail_before_file_or_api_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, schema: list[dict[str, Any]]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    missing = tmp_path / "must-not-be-read.txt"

    with pytest.raises(ValueError, match="max_tokens"):
        get_backend("llm").extract(
            missing,
            {"fields": schema, "max_tokens": 0},
        )


def test_reusable_llm_helper_validates_limits_before_injected_call(
    schema: list[dict[str, Any]],
) -> None:
    called = False

    def _unexpected(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        return {}

    with pytest.raises(ValueError, match="max_concurrent"):
        extract_fields_from_text(
            "text",
            fields_spec=schema,
            max_concurrent=0,
            call_api=_unexpected,
        )

    assert not called


def test_custom_api_key_env_wins_for_sync_and_reusable_clients(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    doc: Path,
    schema: list[dict[str, Any]],
) -> None:
    init_kwargs: list[dict[str, Any]] = []

    class _FakeMessages:
        def create(self, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(
                stop_reason="tool_use",
                content=[
                    SimpleNamespace(
                        type="tool_use", name="extract", input={"vendor": "Acme"}
                    )
                ],
                usage=SimpleNamespace(input_tokens=0, output_tokens=0),
            )

    class _FakeAnthropic:
        def __init__(self, **kwargs: Any) -> None:
            init_kwargs.append(kwargs)
            self.messages = _FakeMessages()

    secret = "custom-secret-that-must-not-leak"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "wrong-default-secret")
    monkeypatch.setenv("STEL_ANTHROPIC_KEY", secret)
    monkeypatch.setattr("anthropic.Anthropic", _FakeAnthropic)

    result = get_backend("llm").extract(
        doc,
        {"fields": schema, "api_key_env": "STEL_ANTHROPIC_KEY"},
    )

    assert result.fields == {"vendor": "Acme"}
    helper_fields = extract_fields_from_text(
        "invoice text",
        fields_spec=schema,
        api_key_env="STEL_ANTHROPIC_KEY",
    )
    assert helper_fields == {"vendor": "Acme"}
    assert init_kwargs == [
        {"api_key": secret, "max_retries": 4, "timeout": 60.0},
        {"api_key": secret, "max_retries": 4, "timeout": 60.0},
    ]
    assert secret not in caplog.text


def test_missing_custom_api_key_does_not_fall_back_to_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, schema: list[dict[str, Any]]
) -> None:
    fallback_secret = "wrong-default-secret"
    monkeypatch.setenv("ANTHROPIC_API_KEY", fallback_secret)
    monkeypatch.delenv("STEL_ANTHROPIC_KEY", raising=False)

    with pytest.raises(RuntimeError, match="credential environment variable") as exc_info:
        get_backend("llm").extract(
            tmp_path / "not-read.txt",
            {"fields": schema, "api_key_env": "STEL_ANTHROPIC_KEY"},
        )

    assert fallback_secret not in str(exc_info.value)
    assert "STEL_ANTHROPIC_KEY" not in str(exc_info.value)


def test_generation_params_default_and_override(
    monkeypatch: pytest.MonkeyPatch, doc: Path, schema: list[dict[str, Any]]
) -> None:
    counter = _CallCounter({"vendor": "v", "invoice_id": "i", "total": 0.0})
    monkeypatch.setattr(LLMBackend, "_call_api", counter)
    backend = get_backend("llm")

    backend.extract(doc, {"fields": schema})
    first_reference = counter.kwargs.pop("api_key_env")
    assert isinstance(first_reference, CredentialReference)
    assert "ANTHROPIC_API_KEY" not in repr(first_reference)
    assert counter.kwargs == {
        "provider": "anthropic",
        "temperature": 0.0,
        "max_tokens": 2048,
        "max_retries": 4,
    }

    backend.extract(
        doc,
        {"fields": schema, "temperature": 0.3, "max_tokens": 8192, "max_retries": 1},
    )
    second_reference = counter.kwargs.pop("api_key_env")
    assert isinstance(second_reference, CredentialReference)
    assert "ANTHROPIC_API_KEY" not in repr(second_reference)
    assert counter.kwargs == {
        "provider": "anthropic",
        "temperature": 0.3,
        "max_tokens": 8192,
        "max_retries": 1,
    }


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


def test_max_tokens_is_part_of_cache_key(
    monkeypatch: pytest.MonkeyPatch, doc: Path, schema: list[dict[str, Any]], tmp_path: Path
) -> None:
    counter = _CallCounter({"vendor": "v", "invoice_id": "i", "total": 0.0})
    monkeypatch.setattr(LLMBackend, "_call_api", counter)
    backend = get_backend("llm")
    opts = {"cache_path": str(tmp_path / "cache.duckdb"), "fields": schema}

    backend.extract(doc, opts)
    backend.extract(doc, {**opts, "max_tokens": 4096})
    backend.extract(doc, {**opts, "max_tokens": 4096})

    assert counter.calls == 2


def test_vllm_base_url_is_part_of_cache_key(
    tmp_path: Path, schema: list[dict[str, Any]]
) -> None:
    calls = 0

    def fake_call(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        nonlocal calls
        calls += 1
        return {"vendor": "v", "invoice_id": "i", "total": 1.0}

    common: dict[str, Any] = {
        "fields_spec": schema,
        "provider": "vllm",
        "model": "invoice-extractor",
        "cache_path": tmp_path / "cache.duckdb",
        "call_api": fake_call,
    }
    extract_fields_from_text(
        "invoice",
        base_url="https://first.example.test/v1",
        **common,
    )
    extract_fields_from_text(
        "invoice",
        base_url="https://second.example.test/v1/",
        **common,
    )
    extract_fields_from_text(
        "invoice",
        base_url="HTTPS://SECOND.EXAMPLE.TEST:443/v1",
        **common,
    )

    assert calls == 2


def test_no_endpoint_preserves_legacy_cache_key() -> None:
    values: dict[str, Any] = {
        "provider": "anthropic",
        "provider_identity": "anthropic/1",
        "model": "claude-test",
        "content_hash": "content",
        "schema_hash": "schema",
        "max_tokens": 2048,
    }
    canonical = json.dumps(
        {
            "contract_version": provider_base.PROVIDER_CONTRACT_VERSION,
            **values,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.blake2b(
        canonical.encode(), digest_size=HASH_DIGEST_SIZE
    ).hexdigest()

    assert llm_backend._cache_key(**values, base_url=None) == (
        f"provider-v{provider_base.PROVIDER_CONTRACT_VERSION}|anthropic|"
        f"claude-test|{digest}"
    )


def test_cache_key_separates_provider_implementations() -> None:
    common: dict[str, Any] = {
        "provider": "anthropic",
        "model": "shared-model",
        "content_hash": "content",
        "schema_hash": "schema",
        "max_tokens": 2048,
    }
    first = llm_backend._cache_key(
        **common,
        provider_identity="implementation-one-with-sensitive-path",
    )
    second = llm_backend._cache_key(
        **common,
        provider_identity="implementation-two",
    )

    assert first != second
    assert "sensitive-path" not in first


def test_cache_key_survives_stel_release_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The response cache is keyed on the semantic request and the provider
    contract identity — never the stel release — so routine upgrades do
    not force paid re-extraction."""
    common: dict[str, Any] = {
        "provider": "anthropic",
        "model": "shared-model",
        "content_hash": "content",
        "schema_hash": "schema",
        "max_tokens": 2048,
    }
    versions = {"stel": "1.0", "anthropic": "5.0"}
    monkeypatch.setattr(
        "stel.providers.base.package_version",
        lambda package: versions[package],
    )
    try:
        provider_base._implementation_identity.cache_clear()
        identity = get_inference_provider("anthropic").implementation_identity()
        first = llm_backend._cache_key(**common, provider_identity=identity)
        versions["stel"] = "2.0"
        provider_base._implementation_identity.cache_clear()
        bumped_identity = get_inference_provider(
            "anthropic"
        ).implementation_identity()
        second = llm_backend._cache_key(
            **common, provider_identity=bumped_identity
        )
    finally:
        provider_base._implementation_identity.cache_clear()

    assert identity == bumped_identity
    assert first == second


def test_legacy_cache_entry_is_not_reused(tmp_path: Path) -> None:
    text = "legacy cached document"
    model = "claude-haiku-4-5"
    system = "extract"
    fields_spec = [{"name": "x"}]
    content_hash = hashlib.blake2b(
        text.encode(), digest_size=HASH_DIGEST_SIZE
    ).hexdigest()
    schema_hash = llm_backend._hash_schema(system, fields_spec, 0.0)
    legacy_key = f"{model}|{content_hash}|{schema_hash}"
    cache_path = tmp_path / "cache.duckdb"
    llm_backend._cache_put(
        cache_path,
        legacy_key,
        model=model,
        content_hash=content_hash,
        schema_hash=schema_hash,
        fields={"x": "legacy"},
    )
    counter = _CallCounter({"x": "fresh"})

    fields, usage = llm_backend.extract_fields_with_usage(
        text,
        fields_spec=fields_spec,
        model=model,
        system=system,
        cache_path=cache_path,
        call_api=counter,
    )

    assert fields == {"x": "fresh"}
    assert usage["cache_hits"] == 0
    assert counter.calls == 1


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_cache_file_is_private_and_rejects_symlinks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "cache.duckdb"
    wal_path = Path(f"{cache_path}.wal")
    wal_modes: list[int] = []
    real_connect = llm_backend.duckdb.connect

    class InspectingConnection:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._connection = real_connect(*args, **kwargs)

        def __getattr__(self, name: str) -> object:
            return getattr(self._connection, name)

        def execute(self, *args: Any, **kwargs: Any) -> object:
            result = self._connection.execute(*args, **kwargs)
            if args and "INSERT INTO llm_cache" in str(args[0]):
                wal_modes.append(stat.S_IMODE(wal_path.stat().st_mode))
            return result

    monkeypatch.setattr(llm_backend.duckdb, "connect", InspectingConnection)
    original_umask = os.umask(0o022)
    try:
        llm_backend._cache_put(
            cache_path,
            "provider-v1|anthropic|model|private",
            model="model",
            content_hash="content",
            schema_hash="schema",
            fields={"private": "value"},
        )
    finally:
        os.umask(original_umask)

    assert stat.S_IMODE(cache_path.stat().st_mode) == 0o600
    assert wal_modes == [0o600]

    os.chmod(cache_path, 0o644)
    llm_backend._cache_put(
        cache_path,
        "provider-v1|anthropic|model|normalized",
        model="model",
        content_hash="content",
        schema_hash="schema",
        fields={"private": "value"},
    )
    assert stat.S_IMODE(cache_path.stat().st_mode) == 0o600

    symlink_path = tmp_path / "cache-link.duckdb"
    symlink_path.symlink_to(cache_path)
    with pytest.raises(ValueError, match="non-symlink"):
        llm_backend._cache_put(
            symlink_path,
            "provider-v1|anthropic|model|symlink",
            model="model",
            content_hash="content",
            schema_hash="schema",
            fields={"private": "value"},
        )
    with pytest.raises(ValueError, match="non-symlink"):
        llm_backend._cache_get(symlink_path, "private")

    victim = tmp_path / "victim.txt"
    victim.write_text("unchanged")
    wal_path.symlink_to(victim)
    with pytest.raises(ValueError, match="non-symlink"):
        llm_backend._cache_put(
            cache_path,
            "provider-v1|anthropic|model|wal-symlink",
            model="model",
            content_hash="content",
            schema_hash="schema",
            fields={"private": "value"},
        )
    assert victim.read_text() == "unchanged"


@pytest.mark.skipif(os.name == "nt", reason="POSIX special files")
def test_cache_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.fifo"
    os.mkfifo(cache_path)

    with pytest.raises(ValueError, match="regular"):
        llm_backend._cache_get(cache_path, "key")
    with pytest.raises(ValueError, match="regular"):
        llm_backend._cache_put(
            cache_path,
            "provider-v1|anthropic|model|fifo",
            model="model",
            content_hash="content",
            schema_hash="schema",
            fields={"private": "value"},
        )


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

    assert init_kwargs == {
        "api_key": "test-key",
        "max_retries": 4,
        "timeout": 60.0,
    }
    assert create_kwargs["temperature"] == 0.0
    assert create_kwargs["max_tokens"] == 2048
    assert isinstance(create_kwargs["system"], str)


def test_extract_with_usage_carries_tokens(tmp_path: Path) -> None:
    from stel.backends.llm_backend import extract_fields_with_usage

    def fake(
        content: str, model: str, system: str, fields_spec: list[dict[str, Any]],
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return {"x": 1}, {"input_tokens": 500, "output_tokens": 42}

    fields, usage = extract_fields_with_usage(
        "some text",
        fields_spec=[{"name": "x"}],
        call_api=fake,
        cache_path=tmp_path / "cache.duckdb",
    )
    assert fields == {"x": 1}
    assert usage["api_calls"] == 1
    assert usage["cache_hits"] == 0
    assert usage["input_tokens"] == 500
    assert usage["output_tokens"] == 42
    assert usage["cache_read_input_tokens"] == 0


def test_extract_with_usage_cache_hit_is_zero_tokens(tmp_path: Path) -> None:
    from stel.backends.llm_backend import extract_fields_with_usage

    def fake(
        content: str, model: str, system: str, fields_spec: list[dict[str, Any]],
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return {"x": 1}, {"input_tokens": 500, "output_tokens": 42}

    kwargs: dict[str, Any] = {
        "fields_spec": [{"name": "x"}],
        "call_api": fake,
        "cache_path": tmp_path / "cache.duckdb",
    }
    extract_fields_with_usage("same text", **kwargs)
    fields, usage = extract_fields_with_usage("same text", **kwargs)

    assert fields == {"x": 1}
    assert usage == {
        "api_calls": 0,
        "cache_hits": 1,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }


def test_extract_with_usage_accepts_bare_dict_fake() -> None:
    """Injected call_api fns predating #75 return fields only — still valid."""
    from stel.backends.llm_backend import extract_fields_with_usage

    def fake(
        content: str, model: str, system: str, fields_spec: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {"x": 1}

    fields, usage = extract_fields_with_usage(
        "text", fields_spec=[{"name": "x"}], call_api=fake
    )
    assert fields == {"x": 1}
    assert usage["api_calls"] == 1
    assert usage["input_tokens"] == 0


def test_max_concurrent_gates_api_calls() -> None:
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from stel.backends.llm_backend import extract_fields_from_text

    active = 0
    peak = 0
    lock = threading.Lock()

    def fake(
        content: str, model: str, system: str, fields_spec: list[dict[str, Any]],
        **kwargs: Any,
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
