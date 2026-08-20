from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import polars as pl
import pytest
import yaml

from stel.adapters import parse_warehouse_config
from stel.text import PIIEntity, detect_pii, redact_pii
from stel.text import pii as pii_module
from stel.text.transforms import redact_pii as t_redact_pii
from stel.transforms import TransformContext


def _ctx(options: dict[str, Any] | None = None) -> TransformContext:
    return TransformContext(
        project_dir=Path("."),
        profile_name="test",
        target_name="dev",
        warehouse=parse_warehouse_config(
            {"type": "duckdb", "path": "./t.duckdb", "schema": "main"}
        ),
        llm=None,
        options=options or {},
    )


@pytest.fixture(autouse=True)
def _reset_analyzer_cache() -> None:
    """LRU caches in pii.py can hold the patched mock across tests; clear them."""
    pii_module._get_analyzer.cache_clear()
    pii_module._get_anonymizer.cache_clear()


def _mock_analyzer(monkeypatch: pytest.MonkeyPatch, spans: list[dict[str, Any]]) -> None:
    """Patch _get_analyzer to return a stub that reports the given spans."""

    class _StubResult:
        def __init__(self, entity_type: str, start: int, end: int, score: float) -> None:
            self.entity_type = entity_type
            self.start = start
            self.end = end
            self.score = score

    stub_results = [_StubResult(**s) for s in spans]

    fake_analyzer = MagicMock()
    fake_analyzer.analyze = MagicMock(return_value=stub_results)

    def fake_get(model: str = "en_core_web_sm"):
        return fake_analyzer

    monkeypatch.setattr(pii_module, "_get_analyzer", fake_get)


def test_detect_pii_returns_typed_entities(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "Email me at alex@example.com please"
    email = "alex@example.com"
    start = text.index(email)
    _mock_analyzer(monkeypatch, [
        {"entity_type": "EMAIL_ADDRESS", "start": start, "end": start + len(email), "score": 0.95},
    ])

    entities = detect_pii(text)
    assert len(entities) == 1
    e = entities[0]
    assert isinstance(e, PIIEntity)
    assert e.type == "EMAIL_ADDRESS"
    assert e.text == email
    assert e.score == 0.95


def test_detect_pii_empty_input() -> None:
    # No analyzer needed for empty input
    assert detect_pii("") == []


def test_detect_pii_score_threshold_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    """score_threshold should be forwarded to Presidio."""
    captured: dict[str, Any] = {}

    class _Stub:
        def analyze(self, **kwargs) -> list[Any]:
            captured.update(kwargs)
            return []

    def fake_get(model: str = "en_core_web_sm"):
        return _Stub()

    monkeypatch.setattr(pii_module, "_get_analyzer", fake_get)

    detect_pii("hello", score_threshold=0.7)
    assert captured.get("score_threshold") == 0.7


def test_redact_pii_with_type_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "Email me at alex@example.com please"
    _mock_analyzer(monkeypatch, [
        {"entity_type": "EMAIL_ADDRESS", "start": 12, "end": 28, "score": 0.95},
    ])
    redacted, entities = redact_pii(text)
    assert redacted == "Email me at [EMAIL_ADDRESS] please"
    assert entities[0].type == "EMAIL_ADDRESS"


def test_redact_pii_custom_replacement(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "Call (555) 123-4567 now"
    _mock_analyzer(monkeypatch, [
        {"entity_type": "PHONE_NUMBER", "start": 5, "end": 19, "score": 0.9},
    ])
    redacted, _ = redact_pii(text, replacement="<<redacted>>")
    assert redacted == "Call <<redacted>> now"


def test_redact_pii_multiple_spans_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "From alex@example.com to bob@example.com"
    _mock_analyzer(monkeypatch, [
        {"entity_type": "EMAIL_ADDRESS", "start": 5, "end": 21, "score": 0.95},
        {"entity_type": "EMAIL_ADDRESS", "start": 25, "end": 40, "score": 0.95},
    ])
    redacted, entities = redact_pii(text)
    assert redacted == "From [EMAIL_ADDRESS] to [EMAIL_ADDRESS]"
    assert len(entities) == 2


def test_redact_pii_no_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_analyzer(monkeypatch, [])
    redacted, entities = redact_pii("nothing sensitive here")
    assert redacted == "nothing sensitive here"
    assert entities == []


def test_redact_pii_overlapping_spans_keeps_higher_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Presidio sometimes returns overlapping entities (e.g. EMAIL_ADDRESS
    containing a URL). The redactor should keep the highest-score one per
    cluster so character offsets don't get mangled."""
    text = "email alex@example.com here"
    email_start = text.index("alex@example.com")
    url_start = text.index("example.com")
    _mock_analyzer(monkeypatch, [
        {
            "entity_type": "URL",
            "start": url_start,
            "end": url_start + 11,
            "score": 0.5,
        },
        {
            "entity_type": "EMAIL_ADDRESS",
            "start": email_start,
            "end": email_start + 16,
            "score": 1.0,
        },
    ])
    redacted, kept = redact_pii(text)
    assert redacted == "email [EMAIL_ADDRESS] here"
    assert len(kept) == 1
    assert kept[0].type == "EMAIL_ADDRESS"


def test_pii_entity_to_dict() -> None:
    e = PIIEntity(type="PERSON", start=0, end=5, score=0.876543, text="Alex")
    d = e.to_dict()
    assert d == {"type": "PERSON", "start": 0, "end": 5, "score": 0.8765}


def test_pii_entity_to_dict_can_include_raw_text() -> None:
    e = PIIEntity(type="PERSON", start=0, end=5, score=0.876543, text="Alex")
    assert e.to_dict(include_text=True)["text"] == "Alex"


# ─── transform integration ────────────────────────────────────────────────


def test_transform_redact_pii_in_place(monkeypatch: pytest.MonkeyPatch) -> None:
    body = "Ping alex@example.com later"
    start = body.index("alex@example.com")
    _mock_analyzer(monkeypatch, [
        {"entity_type": "EMAIL_ADDRESS", "start": start, "end": start + 16, "score": 0.95},
    ])
    df = pl.DataFrame({"body": [body]})
    out = t_redact_pii.run(
        {"upstream": df}, _ctx({"text_field": "body"})
    )
    assert out["body"][0] == "Ping [EMAIL_ADDRESS] later"


def test_transform_redact_pii_with_entities_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = "Ping alex@example.com later"
    start = body.index("alex@example.com")
    _mock_analyzer(monkeypatch, [
        {"entity_type": "EMAIL_ADDRESS", "start": start, "end": start + 16, "score": 0.95},
    ])
    df = pl.DataFrame({"body": [body]})
    out = t_redact_pii.run(
        {"upstream": df},
        _ctx(
            {
                "text_field": "body",
                "output_field": "body_clean",
                "entities_field": "found_pii",
            }
        ),
    )
    assert "body" not in out.columns
    assert out["body_clean"][0] == "Ping [EMAIL_ADDRESS] later"
    spans = json.loads(out["found_pii"][0])
    assert len(spans) == 1
    assert spans[0]["type"] == "EMAIL_ADDRESS"
    assert "text" not in spans[0]


def test_transform_requires_explicit_opt_in_to_retain_input_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = "Ping alex@example.com later"
    start = body.index("alex@example.com")
    _mock_analyzer(monkeypatch, [
        {
            "entity_type": "EMAIL_ADDRESS",
            "start": start,
            "end": start + 16,
            "score": 0.95,
        },
    ])
    out = t_redact_pii.run(
        {"upstream": pl.DataFrame({"body": [body]})},
        _ctx(
            {
                "text_field": "body",
                "output_field": "body_clean",
                "retain_input_text": True,
            }
        ),
    )
    assert out["body"][0] == body
    assert out["body_clean"][0] == "Ping [EMAIL_ADDRESS] later"


def test_transform_default_output_does_not_serialize_detected_pii(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    email = "distinctive.privacy+115@example.com"
    body = f"Ping {email} later"
    start = body.index(email)
    _mock_analyzer(monkeypatch, [
        {
            "entity_type": "EMAIL_ADDRESS",
            "start": start,
            "end": start + len(email),
            "score": 0.95,
        },
    ])
    out = t_redact_pii.run(
        {"upstream": pl.DataFrame({"body": [body]})},
        _ctx({"text_field": "body", "entities_field": "found_pii"}),
    )
    assert email not in json.dumps(out.to_dicts())


def test_transform_can_opt_in_to_raw_entity_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    email = "alex@example.com"
    body = f"Ping {email} later"
    start = body.index(email)
    _mock_analyzer(monkeypatch, [
        {
            "entity_type": "EMAIL_ADDRESS",
            "start": start,
            "end": start + len(email),
            "score": 0.95,
        },
    ])
    out = t_redact_pii.run(
        {"upstream": pl.DataFrame({"body": [body]})},
        _ctx(
            {
                "text_field": "body",
                "entities_field": "found_pii",
                "include_raw_text": True,
            }
        ),
    )
    assert json.loads(out["found_pii"][0])[0]["text"] == email


def test_transform_keep_fields_is_exact_safe_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    email = "private.customer@example.com"
    body = f"Ping {email} later"
    start = body.index(email)
    _mock_analyzer(monkeypatch, [
        {
            "entity_type": "EMAIL_ADDRESS",
            "start": start,
            "end": start + len(email),
            "score": 0.95,
        },
    ])
    out = t_redact_pii.run(
        {
            "upstream": pl.DataFrame(
                {
                    "ticket_id": ["T-1"],
                    "customer_email": [email],
                    "body": [body],
                }
            )
        },
        _ctx(
            {
                "text_field": "body",
                "output_field": "body_redacted",
                "entities_field": "found_pii",
                "keep_fields": ["ticket_id", "body_redacted", "found_pii"],
            }
        ),
    )
    assert out.columns == ["ticket_id", "body_redacted", "found_pii"]
    assert email not in json.dumps(out.to_dicts())


def test_transform_drop_fields_removes_originals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_analyzer(monkeypatch, [])
    out = t_redact_pii.run(
        {
            "upstream": pl.DataFrame(
                {"ticket_id": ["T-1"], "body": ["safe text"]}
            )
        },
        _ctx(
            {
                "text_field": "body",
                "output_field": "body_redacted",
                "drop_fields": ["body"],
            }
        ),
    )
    assert out.columns == ["ticket_id", "body_redacted"]


def test_transform_rejects_conflicting_projection_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_analyzer(monkeypatch, [])
    with pytest.raises(ValueError, match="mutually exclusive"):
        t_redact_pii.run(
            {"upstream": pl.DataFrame({"body": ["safe text"]})},
            _ctx(
                {
                    "text_field": "body",
                    "keep_fields": ["body"],
                    "drop_fields": ["body"],
                }
            ),
        )


def test_support_tickets_redacted_example_omits_sensitive_source_fields() -> None:
    model_path = (
        Path(__file__).parents[1]
        / "examples"
        / "support_tickets_pipeline"
        / "models"
        / "redacted_tickets.yml"
    )
    config = yaml.safe_load(model_path.read_text())
    options = config["models"][0]["transform"]["options"]

    assert {"customer_name", "customer_email", "summary"}.isdisjoint(
        options["keep_fields"]
    )
    assert "summary_redacted" in options["keep_fields"]
    assert options.get("include_raw_text", False) is False
