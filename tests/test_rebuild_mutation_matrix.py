"""The mutation matrix for incremental identity (constellations#707, ALE-284).

Build Systems a la Carte separates what a build system must persist from what
it may rebuild on. stel's contract is that split: a semantic change moves
`code_version` so published rows reprocess, and an execution-only change must
not, so throughput tuning never bills a corpus again (#572's original report
and #587's `batch_size` recovery are both this line crossed). The matrix
states the split per kind and per axis in one place, so a new identity field
has a row to be added to rather than a scattered convention to be remembered.

Only unpinned rows live here; the repo pins one test per distinct failure
mode, so an axis already pinned elsewhere keeps its existing test and this
file's coverage map points at it:

- prompt, model selection, and inference-implementation identity for
  `extraction: llm` -- `test_versioning.py`
  (`test_profile_llm_semantics_change_model_code_version`,
  `test_provider_and_model_change_model_code_version`,
  `test_provider_implementation_changes_model_code_version`)
- prompt identity for a native `llm:` model -- `test_llm_map.py`
  (`test_code_version_changes_with_prompt_and_provider`)
- backend parser and implementation identity for plain extraction --
  `test_versioning.py` (`test_backend_parser_identity_changes_model_code_version`)
- an embed model's provider/model identity, plus `batch_size` and
  `max_retries` as execution-only -- `test_embedding.py`
  (`test_embed_code_version_excludes_execution_tuning`)
- `commit_every` for transforms, `flush_every`/`publish_every` for
  extraction, and `api_key_env`/`cache_path` for `extraction: llm` --
  `test_incremental_transforms.py`, `test_versioning.py`

Input add/change/delete is deliberately not a row: it moves
`input_fingerprint`, never `code_version`, and its end-to-end behaviour is
pinned by the equivalence harness in `test_incremental_equivalence.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from stel.adapters.duckdb import DuckDBWarehouseConfig
from stel.config.model import (
    EmbedConfig,
    ExtractionConfig,
    FieldConfig,
    LLMTransformConfig,
    ModelConfig,
    TransformConfig,
)
from stel.config.profile import LLMConfig
from stel.config.project import ProjectConfig
from stel.profile import ResolvedProfile
from stel.versioning import compute_model_code_version

_PROJECT = ProjectConfig(name="p")


def _llm_map_model() -> ModelConfig:
    return ModelConfig(
        name="facts",
        depends_on=["ref('raw')"],
        llm=LLMTransformConfig(
            input_field="text",
            prompt="extract facts",
            provider="deterministic",
            model="deterministic-v1",
        ),
        fields=[FieldConfig(name="sentiment", data_type="string")],
    )


def _embed_model() -> ModelConfig:
    return ModelConfig(
        name="embeddings",
        depends_on=["ref('chunks')"],
        embed=EmbedConfig(
            provider="deterministic",
            model="contract-v1",
            text_field="text",
            id_field="chunk_id",
            vector_field="embedding",
            dimensions=4,
        ),
    )


def _llm_extraction_model() -> ModelConfig:
    return ModelConfig(
        name="raw",
        source="ref('docs')",
        extraction=ExtractionConfig(
            backend="llm",
            options={"fields": [{"name": "title", "type": "string"}]},
        ),
    )


def _resolved_llm(**overrides: Any) -> ResolvedProfile:
    return ResolvedProfile(
        profile_name="p",
        target_name="dev",
        warehouse=DuckDBWarehouseConfig(path=Path("/tmp/matrix.duckdb")),
        llm=LLMConfig(provider="deterministic", model="deterministic-v1", **overrides),
        source_paths={},
        profiles_path=Path("/tmp/profiles.yml"),
    )


def _mutate_llm(model: ModelConfig, update: dict[str, Any]) -> ModelConfig:
    assert model.llm is not None
    return model.model_copy(update={"llm": model.llm.model_copy(update=update)})


def _mutate_embed(model: ModelConfig, update: dict[str, Any]) -> ModelConfig:
    assert model.embed is not None
    return model.model_copy(update={"embed": model.embed.model_copy(update=update)})


def _mutate_extraction_options(model: ModelConfig, key: str, value: Any) -> ModelConfig:
    assert model.extraction is not None
    options = {**model.extraction.options, key: value}
    return model.model_copy(
        update={"extraction": model.extraction.model_copy(update={"options": options})}
    )


# A semantic mutation is one a row of #572's report would have paid for again:
# it changes what a published row means, so it must move code_version.
_SEMANTIC_ROWS: tuple[
    tuple[str, Callable[[], ModelConfig], Callable[[ModelConfig], ModelConfig]],
    ...,
] = (
    (
        "llm_map-schema",
        _llm_map_model,
        lambda model: model.model_copy(
            update={
                "fields": [
                    *model.fields,
                    FieldConfig(name="topic", data_type="string"),
                ]
            }
        ),
    ),
    (
        "llm_map-model",
        _llm_map_model,
        lambda model: _mutate_llm(model, {"model": "deterministic-v2"}),
    ),
    (
        "embed-embedding-identity-dimensions",
        _embed_model,
        lambda model: _mutate_embed(model, {"dimensions": 8}),
    ),
)


@pytest.mark.parametrize(
    ("case_id", "build", "mutate"),
    _SEMANTIC_ROWS,
    ids=[row[0] for row in _SEMANTIC_ROWS],
)
def test_semantic_mutations_move_code_version(
    case_id: str,
    build: Callable[[], ModelConfig],
    mutate: Callable[[ModelConfig], ModelConfig],
    tmp_path: Path,
) -> None:
    del case_id
    baseline = compute_model_code_version(build(), _PROJECT, tmp_path)

    assert compute_model_code_version(mutate(build()), _PROJECT, tmp_path) != baseline


# An execution-only mutation must be free: each knob below shapes how a run
# executes, never what a published row says, so moving code_version for one
# re-bills a corpus an operator was only tuning (#587).
_EXECUTION_ROWS: tuple[
    tuple[str, Callable[[], ModelConfig], Callable[[ModelConfig], ModelConfig]],
    ...,
] = (
    *(
        (
            f"llm_map-{knob}",
            _llm_map_model,
            lambda model, knob=knob, value=value: _mutate_llm(model, {knob: value}),
        )
        for knob, value in (
            ("max_concurrent", 16),
            ("max_retries", 9),
            ("flush_every", 2000),
            ("on_code_change", "reprocess"),
            ("reprocess_limit", 100),
        )
    ),
    *(
        (
            f"embed-{knob}",
            _embed_model,
            lambda model, knob=knob, value=value: _mutate_embed(model, {knob: value}),
        )
        for knob, value in (
            ("max_concurrent", 16),
            ("flush_every", 200),
            ("on_code_change", "reprocess"),
            ("reprocess_limit", 100),
        )
    ),
    *(
        (
            f"llm_extraction-{knob}",
            _llm_extraction_model,
            lambda model, knob=knob, value=value: _mutate_extraction_options(model, knob, value),
        )
        for knob, value in (("batch_size", 20), ("max_concurrent", 4))
    ),
)


@pytest.mark.parametrize(
    ("case_id", "build", "mutate"),
    _EXECUTION_ROWS,
    ids=[row[0] for row in _EXECUTION_ROWS],
)
def test_execution_only_mutations_are_free(
    case_id: str,
    build: Callable[[], ModelConfig],
    mutate: Callable[[ModelConfig], ModelConfig],
    tmp_path: Path,
) -> None:
    del case_id
    baseline = compute_model_code_version(build(), _PROJECT, tmp_path)

    assert compute_model_code_version(mutate(build()), _PROJECT, tmp_path) == baseline


def test_llm_helper_implementation_change_moves_code_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `transform: uses_llm` model's identity includes the helper's own.

    The profile's inference-implementation identity is pinned in
    `test_versioning.py`; this row is the other half of that kind -- the
    llm helper backend's own implementation identity, folded in beside it.
    """
    model = ModelConfig(
        name="derived",
        depends_on=["ref('raw')"],
        transform=TransformConfig(type="python", uses_llm=True),
    )
    resolved = _resolved_llm()

    def implementation(identity: str) -> Callable[[pytest.MonkeyPatch], None]:
        def patch(monkeypatch: pytest.MonkeyPatch) -> None:
            monkeypatch.setattr(
                "stel.versioning.get_backend",
                lambda name: SimpleNamespace(
                    implementation_identity=lambda: identity,
                    parser_identity=lambda: "parser/stable",
                ),
            )

        return patch

    implementation("implementation/one")(monkeypatch)
    baseline = compute_model_code_version(model, _PROJECT, tmp_path, resolved=resolved)
    implementation("implementation/two")(monkeypatch)

    assert compute_model_code_version(model, _PROJECT, tmp_path, resolved=resolved) != baseline
