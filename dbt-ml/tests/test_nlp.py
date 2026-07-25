from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from dbt_ml.adapters import parse_warehouse_config
from dbt_ml.compiler import validate_project_contract
from dbt_ml.config import load_project
from dbt_ml.text import (
    NLPDocument,
    NLPEntity,
    NLPError,
    NLPIdentity,
    NLPToken,
)
from dbt_ml.text import nlp as nlp_core
from dbt_ml.text.transforms import nlp_entities, nlp_tokens
from dbt_ml.transforms import TransformContext

_IDENTITY = NLPIdentity(
    provider="fake",
    provider_version="1.2.3",
    model="economic-en",
    model_version="2026.07",
    language="en",
)
_TOKENS = (
    NLPToken(
        index=0,
        sentence_index=0,
        start=0,
        end=7,
        text="Federal",
        lemma="federal",
        pos="ADJ",
        tag="JJ",
        is_stop=False,
        is_alpha=True,
        is_space=False,
    ),
    NLPToken(
        index=1,
        sentence_index=0,
        start=7,
        end=8,
        text=" ",
        lemma=" ",
        pos="SPACE",
        tag="_SP",
        is_stop=False,
        is_alpha=False,
        is_space=True,
    ),
    NLPToken(
        index=2,
        sentence_index=0,
        start=8,
        end=15,
        text="Reserve",
        lemma="Reserve",
        pos="PROPN",
        tag="NNP",
        is_stop=False,
        is_alpha=True,
        is_space=False,
    ),
)
_ENTITIES = (
    NLPEntity(
        index=0,
        sentence_index=0,
        start=0,
        end=15,
        label="ORG",
        text="Federal Reserve",
    ),
)
_DOCUMENT = NLPDocument(tokens=_TOKENS, entities=_ENTITIES)


class _FakeProvider:
    def __init__(
        self,
        documents: Iterable[NLPDocument] = (_DOCUMENT,),
    ) -> None:
        self.identity = _IDENTITY
        self.documents = tuple(documents)
        self.calls: list[tuple[list[str], int]] = []

    def pipe(
        self,
        texts: Iterable[str],
        *,
        batch_size: int,
    ) -> Iterable[NLPDocument]:
        self.calls.append((list(texts), batch_size))
        return iter(self.documents)


def _ctx(options: dict[str, object] | None = None) -> TransformContext:
    return TransformContext(
        project_dir=Path("."),
        profile_name="test",
        target_name="dev",
        warehouse=parse_warehouse_config(
            {"type": "duckdb", "path": "./test.duckdb", "schema": "main"}
        ),
        llm=None,
        options=options or {},
    )


def _use_provider(
    monkeypatch: pytest.MonkeyPatch,
    provider: _FakeProvider,
) -> None:
    monkeypatch.setattr(nlp_core, "get_nlp_provider", lambda _options: provider)


def test_token_transform_emits_stable_child_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _FakeProvider()
    _use_provider(monkeypatch, provider)
    upstream = pl.DataFrame(
        {
            "doc_key": ["fed-minutes-1"],
            "body": ["Federal Reserve"],
            "release_year": [2026],
            "sensitive_notes": ["do not copy"],
        }
    )
    options = {
        "document_id_field": "doc_key",
        "text_field": "body",
        "include_fields": ["release_year"],
        "batch_size": 7,
    }

    first = nlp_tokens.run({"raw": upstream}, _ctx(options))
    second = nlp_tokens.run({"raw": upstream}, _ctx(options))

    assert provider.calls == [
        (["Federal Reserve"], 7),
        (["Federal Reserve"], 7),
    ]
    assert first["token_text"].to_list() == ["Federal", "Reserve"]
    assert first["token_index"].to_list() == [0, 2]
    assert first["document_id"].to_list() == ["fed-minutes-1", "fed-minutes-1"]
    assert first["release_year"].to_list() == [2026, 2026]
    assert first["token_id"].to_list() == second["token_id"].to_list()
    assert first["nlp_model"].to_list() == ["economic-en", "economic-en"]
    assert "body" not in first.columns
    assert "sensitive_notes" not in first.columns


def test_token_transform_can_include_space_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _FakeProvider()
    _use_provider(monkeypatch, provider)
    upstream = pl.DataFrame(
        {"document_id": ["fed-minutes-1"], "text": ["Federal Reserve"]}
    )

    output = nlp_tokens.run(
        {"raw": upstream},
        _ctx({"include_space": True}),
    )

    assert output["token_index"].to_list() == [0, 1, 2]


def test_entity_transform_excludes_matched_text_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _FakeProvider()
    _use_provider(monkeypatch, provider)
    upstream = pl.DataFrame(
        {"document_id": ["fed-minutes-1"], "text": ["Federal Reserve"]}
    )

    default = nlp_entities.run({"raw": upstream}, _ctx())
    opted_in = nlp_entities.run(
        {"raw": upstream},
        _ctx({"include_text": True}),
    )

    assert default["label"].to_list() == ["ORG"]
    assert default["confidence"].to_list() == [None]
    assert "entity_text" not in default.columns
    assert opted_in["entity_text"].to_list() == ["Federal Reserve"]
    assert default["entity_id"].to_list() == opted_in["entity_id"].to_list()


def test_entity_id_does_not_fingerprint_excluded_surface_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream = pl.DataFrame(
        {"document_id": ["fed-minutes-1"], "text": ["Federal Reserve"]}
    )
    original_provider = _FakeProvider()
    _use_provider(monkeypatch, original_provider)
    original = nlp_entities.run({"raw": upstream}, _ctx())

    changed_entity = NLPEntity(
        index=0,
        sentence_index=0,
        start=0,
        end=15,
        label="ORG",
        text="sensitive value",
    )
    changed_provider = _FakeProvider(
        documents=(NLPDocument(tokens=_TOKENS, entities=(changed_entity,)),)
    )
    _use_provider(monkeypatch, changed_provider)
    changed = nlp_entities.run({"raw": upstream}, _ctx())

    assert original["entity_id"].to_list() == changed["entity_id"].to_list()
    assert "entity_text" not in changed.columns


def test_empty_upstream_preserves_schema_without_loading_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_loaded(_options: object) -> None:
        raise AssertionError("provider should not load for an empty input")

    monkeypatch.setattr(nlp_core, "get_nlp_provider", fail_if_loaded)
    upstream = pl.DataFrame(
        schema={
            "document_id": pl.String(),
            "text": pl.String(),
            "release_year": pl.Int64(),
        }
    )

    output = nlp_entities.run(
        {"raw": upstream},
        _ctx({"include_fields": ["release_year"]}),
    )

    assert output.is_empty()
    assert output.schema["entity_id"] == pl.String()
    assert output.schema["release_year"] == pl.Int64()


@pytest.mark.parametrize(
    ("upstream", "options", "message"),
    [
        (
            pl.DataFrame({"document_id": ["a", "a"], "text": ["one", "two"]}),
            {},
            "duplicate value 'a'",
        ),
        (
            pl.DataFrame({"document_id": [""], "text": ["one"]}),
            {},
            "null or empty values",
        ),
        (
            pl.DataFrame({"document_id": ["a"], "text": [1]}),
            {},
            "must contain strings or nulls",
        ),
        (
            pl.DataFrame({"document_id": ["a"], "text": ["one"]}),
            {"include_fields": ["missing"]},
            "unknown columns",
        ),
        (
            pl.DataFrame(
                {"document_id": ["a"], "text": ["one"], "token_id": ["unsafe"]}
            ),
            {"include_fields": ["token_id"]},
            "collides with NLP output columns",
        ),
    ],
)
def test_token_transform_rejects_invalid_upstream_contracts(
    monkeypatch: pytest.MonkeyPatch,
    upstream: pl.DataFrame,
    options: dict[str, object],
    message: str,
) -> None:
    provider = _FakeProvider()
    _use_provider(monkeypatch, provider)

    with pytest.raises(ValueError, match=message):
        nlp_tokens.run({"raw": upstream}, _ctx(options))

    assert provider.calls == []


def test_entity_transform_reserves_optional_text_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _FakeProvider()
    _use_provider(monkeypatch, provider)
    upstream = pl.DataFrame(
        {
            "document_id": ["a"],
            "text": ["one"],
            "entity_text": ["ambiguous"],
        }
    )

    with pytest.raises(ValueError, match="collides with NLP output columns"):
        nlp_entities.run(
            {"raw": upstream},
            _ctx({"include_fields": ["entity_text"]}),
        )

    assert provider.calls == []


def test_transform_rejects_provider_output_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _FakeProvider(documents=())
    _use_provider(monkeypatch, provider)
    upstream = pl.DataFrame({"document_id": ["a"], "text": ["one"]})

    with pytest.raises(ValueError, match="expected 1, got 0"):
        nlp_tokens.run({"raw": upstream}, _ctx())


@pytest.mark.parametrize(
    ("validator", "options", "message"),
    [
        (nlp_tokens.validate_options, {"batch_size": 0}, "greater than or equal to 1"),
        (
            nlp_tokens.validate_options,
            {"include_fields": ["text"]},
            "must not repeat",
        ),
        (
            nlp_entities.validate_options,
            {"unexpected": True},
            "Extra inputs are not permitted",
        ),
    ],
)
def test_transform_options_are_strict(
    validator: Callable[[dict[str, object]], None],
    options: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        validator(options)


def test_missing_spacy_model_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeSpacy:
        @staticmethod
        def load(_model: str, *, disable: list[str]) -> None:
            assert disable == []
            raise OSError("not installed")

    nlp_core._spacy_provider.cache_clear()
    monkeypatch.setattr(
        nlp_core,
        "import_optional_dependency",
        lambda *_args, **_kwargs: _FakeSpacy(),
    )

    with pytest.raises(
        NLPError,
        match=r"python -m spacy download en_core_web_sm",
    ):
        nlp_core._spacy_provider("en_core_web_sm", "en", ())


def test_economic_nlp_example_compiles_without_loading_spacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_loaded(_options: object) -> None:
        raise AssertionError("compile should not load an NLP provider")

    monkeypatch.setattr(nlp_core, "get_nlp_provider", fail_if_loaded)
    project_dir = Path(__file__).resolve().parents[1] / "examples" / "economic_nlp"
    project, sources, models = load_project(project_dir)

    dag = validate_project_contract(project, sources, models, project_dir)

    assert dag.execution_order() == [
        "raw_documents",
        "document_entities",
        "document_tokens",
    ]
