"""Tests for declarative for_each / matrix model expansion (issue #57)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dbt_ml.config import ConfigError, load_project
from dbt_ml.config.loader import (
    _MAX_FOR_EACH_VARIANTS,
    _expand_for_each,
    _value_slug,
    _variant_name,
)
from dbt_ml.config.model import ModelConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ml_model(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": "ticket_tfidf",
        "depends_on": ["ref('raw_tickets')"],
        "ml": {
            "task": "features",
            "mode": "fit_transform",
            "provider": "builtin.tfidf",
            "text_field": "body",
            "artifact": {"path": "target/artifacts/ticket_tfidf"},
        },
    }
    base.update(overrides)
    return base


def _make_project(tmp_path: Path, model_yaml: str) -> Path:
    (tmp_path / "dbt_ml_project.yml").write_text(
        "name: test_project\nversion: '0.1.0'\nmodel-paths: ['models']\n"
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "models.yml").write_text(model_yaml)
    return tmp_path


# ---------------------------------------------------------------------------
# _value_slug
# ---------------------------------------------------------------------------


def test_slug_integer() -> None:
    assert _value_slug(1) == "1"


def test_slug_float() -> None:
    assert _value_slug(0.5) == "0_5"


def test_slug_string_lowercase_and_replace() -> None:
    assert _value_slug("Word2Vec") == "word2vec"
    assert _value_slug("with spaces") == "with_spaces"


def test_slug_bool() -> None:
    assert _value_slug(True) == "true"
    assert _value_slug(False) == "false"


def test_slug_none() -> None:
    assert _value_slug(None) == "none"


def test_slug_list_of_ints() -> None:
    assert _value_slug([1, 2]) == "1_2"


def test_slug_list_of_strings() -> None:
    assert _value_slug(["a", "b"]) == "a_b"


def test_slug_long_value_truncated_with_hash() -> None:
    long_val = "a" * 100
    slug = _value_slug(long_val)
    assert len(slug) <= 32 + 1 + 8  # _MAX_SLUG_LEN + _ + 8-char hash
    assert "_" in slug  # contains the separator


# ---------------------------------------------------------------------------
# _variant_name
# ---------------------------------------------------------------------------


def test_variant_name_single_axis() -> None:
    name = _variant_name("ticket_tfidf", ["min_df"], (1,))
    assert name == "ticket_tfidf__min_df_1"


def test_variant_name_two_axes() -> None:
    name = _variant_name("ticket_tfidf", ["min_df", "ngram_range"], (1, [1, 2]))
    assert name == "ticket_tfidf__min_df_1__ngram_range_1_2"


def test_variant_name_uses_double_underscore_axis_separator() -> None:
    name = _variant_name("base", ["a", "b"], ("x", "y"))
    assert name.startswith("base__")
    parts = name.split("__")
    assert parts[0] == "base"
    assert parts[1] == "a_x"
    assert parts[2] == "b_y"


# ---------------------------------------------------------------------------
# _expand_for_each — unit tests on ModelConfig objects
# ---------------------------------------------------------------------------


def test_passthrough_without_for_each() -> None:
    model = ModelConfig.model_validate(_ml_model())
    result = _expand_for_each([model])
    assert result == [model]


def test_single_axis_produces_correct_count() -> None:
    model = ModelConfig.model_validate(
        _ml_model(for_each={"min_df": [1, 2, 5]})
    )
    variants = _expand_for_each([model])
    assert len(variants) == 3


def test_two_axes_cartesian_product() -> None:
    model = ModelConfig.model_validate(
        _ml_model(for_each={"min_df": [1, 2], "max_feat": [100, 1000]})
    )
    variants = _expand_for_each([model])
    assert len(variants) == 4  # 2 × 2


def test_cartesian_product_ordering() -> None:
    model = ModelConfig.model_validate(
        _ml_model(for_each={"a": [1, 2], "b": [10, 20]})
    )
    variants = _expand_for_each([model])
    names = [v.name for v in variants]
    assert names == [
        "ticket_tfidf__a_1__b_10",
        "ticket_tfidf__a_1__b_20",
        "ticket_tfidf__a_2__b_10",
        "ticket_tfidf__a_2__b_20",
    ]


def test_variant_has_no_for_each() -> None:
    model = ModelConfig.model_validate(
        _ml_model(for_each={"min_df": [1, 2]})
    )
    for v in _expand_for_each([model]):
        assert v.for_each is None


def test_template_model_not_in_result() -> None:
    model = ModelConfig.model_validate(
        _ml_model(for_each={"min_df": [1, 2]})
    )
    result = _expand_for_each([model])
    assert all(v.name != "ticket_tfidf" for v in result)


# ---------------------------------------------------------------------------
# Type-preserving substitution
# ---------------------------------------------------------------------------


def test_exact_placeholder_preserves_int_type() -> None:
    model = ModelConfig.model_validate(
        _ml_model(
            name="feat_model",
            for_each={"min_df": [1, 5]},
            ml={
                "task": "features",
                "mode": "fit_transform",
                "provider": "builtin.tfidf",
                "text_field": "body",
                "artifact": {"path": "target/artifacts/feat_model"},
                "options": {"min_df": "${matrix.min_df}"},
            },
        )
    )
    variants = _expand_for_each([model])
    for v in variants:
        assert v.ml is not None
        val = v.ml.options["min_df"]
        assert isinstance(val, int)
    assert [v.ml.options["min_df"] for v in variants if v.ml] == [1, 5]  # type: ignore[union-attr]


def test_exact_placeholder_preserves_list_type() -> None:
    model = ModelConfig.model_validate(
        _ml_model(
            name="feat_model",
            for_each={"ngram_range": [[1, 1], [1, 2]]},
            ml={
                "task": "features",
                "mode": "fit_transform",
                "provider": "builtin.tfidf",
                "text_field": "body",
                "artifact": {"path": "target/artifacts/feat_model"},
                "options": {"ngram_range": "${matrix.ngram_range}"},
            },
        )
    )
    variants = _expand_for_each([model])
    for v in variants:
        assert v.ml is not None
        val = v.ml.options["ngram_range"]
        assert isinstance(val, list)
    assert [v.ml.options["ngram_range"] for v in variants if v.ml] == [[1, 1], [1, 2]]  # type: ignore[union-attr]


def test_partial_placeholder_produces_string() -> None:
    model = ModelConfig.model_validate(
        _ml_model(
            name="feat_model",
            for_each={"label": ["small", "large"]},
            ml={
                "task": "features",
                "mode": "fit_transform",
                "provider": "builtin.tfidf",
                "text_field": "body",
                "artifact": {"path": "target/artifacts/feat_model_${matrix.label}"},
            },
        )
    )
    variants = _expand_for_each([model])
    for v in variants:
        assert v.ml is not None
        assert v.ml.artifact is not None
    # Compare via as_posix() — a Path's str() is OS-dependent (backslashes on
    # Windows), while POSIX is the config's canonical serialized form.
    paths = [
        v.ml.artifact.path.as_posix()
        for v in variants
        if v.ml and v.ml.artifact and v.ml.artifact.path is not None
    ]
    assert paths == [
        "target/artifacts/feat_model_small",
        "target/artifacts/feat_model_large",
    ]


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


def test_base_name_added_as_tag() -> None:
    model = ModelConfig.model_validate(
        _ml_model(name="base_model", for_each={"k": [1, 2]})
    )
    for v in _expand_for_each([model]):
        assert "base_model" in v.tags


def test_existing_tags_preserved() -> None:
    model = ModelConfig.model_validate(
        _ml_model(name="base_model", tags=["experiment", "v2"], for_each={"k": [1]})
    )
    variant = _expand_for_each([model])[0]
    assert "experiment" in variant.tags
    assert "v2" in variant.tags
    assert "base_model" in variant.tags


def test_base_name_tag_not_duplicated_when_already_present() -> None:
    model = ModelConfig.model_validate(
        _ml_model(name="base_model", tags=["base_model"], for_each={"k": [1]})
    )
    variant = _expand_for_each([model])[0]
    assert variant.tags.count("base_model") == 1


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_provenance_copied_from_template(tmp_path: Path) -> None:
    project_path = _make_project(
        tmp_path,
        "\n".join(
            [
                "version: 2",
                "models:",
                "  - name: tfidf",
                "    for_each:",
                "      min_df: [1, 2]",
                "    depends_on: [ref('src')]",
                "    ml:",
                "      task: features",
                "      mode: fit_transform",
                "      provider: builtin.tfidf",
                "      text_field: body",
                "      artifact:",
                "        path: target/artifacts/tfidf",
            ]
        ),
    )
    _, _, models = load_project(project_path)
    for m in models:
        assert m._yaml_provenance is not None


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_empty_for_each_dict_raises() -> None:
    with pytest.raises(ValueError, match="at least one axis"):
        ModelConfig.model_validate(_ml_model(for_each={}))


def test_empty_axis_values_raises() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        ModelConfig.model_validate(_ml_model(for_each={"min_df": []}))


def test_invalid_axis_name_raises() -> None:
    with pytest.raises(ValueError, match="valid identifier"):
        ModelConfig.model_validate(_ml_model(for_each={"bad-name": [1]}))


def test_too_many_variants_raises() -> None:
    model = ModelConfig.model_validate(
        _ml_model(for_each={"a": list(range(_MAX_FOR_EACH_VARIANTS + 1))})
    )
    with pytest.raises(ConfigError, match="expands to"):
        _expand_for_each([model])


def test_slug_collision_raises() -> None:
    # "1.0" and "1_0" both produce slug "1_0"
    model = ModelConfig.model_validate(
        _ml_model(for_each={"val": [1.0, "1_0"]})
    )
    with pytest.raises(ConfigError, match="collision"):
        _expand_for_each([model])


# ---------------------------------------------------------------------------
# Full integration through load_project
# ---------------------------------------------------------------------------


def test_load_project_expands_for_each(tmp_path: Path) -> None:
    project_path = _make_project(
        tmp_path,
        "\n".join(
            [
                "version: 2",
                "models:",
                "  - name: ticket_tfidf",
                "    depends_on: [ref('raw_tickets')]",
                "    for_each:",
                "      min_df: [1, 2]",
                "      ngram_range: [[1, 1], [1, 2]]",
                "    ml:",
                "      task: features",
                "      mode: fit_transform",
                "      provider: builtin.tfidf",
                "      text_field: body",
                "      artifact:",
                "        path: target/artifacts/ticket_tfidf",
                "      options:",
                "        min_df: ${matrix.min_df}",
                "        ngram_range: ${matrix.ngram_range}",
            ]
        ),
    )
    _, _, models = load_project(project_path)
    assert len(models) == 4  # 2 × 2
    names = {m.name for m in models}
    assert "ticket_tfidf__min_df_1__ngram_range_1_1" in names
    assert "ticket_tfidf__min_df_1__ngram_range_1_2" in names
    assert "ticket_tfidf__min_df_2__ngram_range_1_1" in names
    assert "ticket_tfidf__min_df_2__ngram_range_1_2" in names
    assert "ticket_tfidf" not in names

    # Check the substituted values are correct types
    v11 = next(m for m in models if m.name == "ticket_tfidf__min_df_1__ngram_range_1_1")
    assert v11.ml is not None
    assert v11.ml.options["min_df"] == 1
    assert isinstance(v11.ml.options["min_df"], int)
    assert v11.ml.options["ngram_range"] == [1, 1]
    assert isinstance(v11.ml.options["ngram_range"], list)

    v22 = next(m for m in models if m.name == "ticket_tfidf__min_df_2__ngram_range_1_2")
    assert v22.ml is not None
    assert v22.ml.options["min_df"] == 2
    assert v22.ml.options["ngram_range"] == [1, 2]


def test_load_project_template_not_in_models(tmp_path: Path) -> None:
    project_path = _make_project(
        tmp_path,
        "\n".join(
            [
                "version: 2",
                "models:",
                "  - name: tfidf",
                "    depends_on: [ref('src')]",
                "    for_each:",
                "      min_df: [1]",
                "    ml:",
                "      task: features",
                "      mode: fit_transform",
                "      provider: builtin.tfidf",
                "      text_field: body",
                "      artifact:",
                "        path: target/artifacts/tfidf",
            ]
        ),
    )
    _, _, models = load_project(project_path)
    assert all(m.name != "tfidf" for m in models)


def test_non_for_each_models_alongside_for_each(tmp_path: Path) -> None:
    project_path = _make_project(
        tmp_path,
        "\n".join(
            [
                "version: 2",
                "models:",
                "  - name: plain_model",
                "    depends_on: [ref('src')]",
                "    ml:",
                "      task: features",
                "      mode: fit_transform",
                "      provider: builtin.tfidf",
                "      text_field: body",
                "      artifact:",
                "        path: target/artifacts/plain_model",
                "  - name: matrix_model",
                "    depends_on: [ref('src')]",
                "    for_each:",
                "      k: [3, 5]",
                "    ml:",
                "      task: features",
                "      mode: fit_transform",
                "      provider: builtin.tfidf",
                "      text_field: body",
                "      artifact:",
                "        path: target/artifacts/matrix_model",
                "      options:",
                "        k: ${matrix.k}",
            ]
        ),
    )
    _, _, models = load_project(project_path)
    names = {m.name for m in models}
    assert "plain_model" in names
    assert "matrix_model__k_3" in names
    assert "matrix_model__k_5" in names
    assert "matrix_model" not in names
    assert len(models) == 3  # 1 plain + 2 expanded


def test_typed_field_placeholder_substituted_before_validation(tmp_path: Path) -> None:
    """${matrix.KEY} in a strictly-typed field (chunk_size: int) must work via
    YAML loading — expansion on raw dicts happens before ModelConfig validation."""
    project_path = _make_project(
        tmp_path,
        "\n".join(
            [
                "version: 2",
                "models:",
                "  - name: chunker",
                "    for_each:",
                "      size: [256, 512, 1024]",
                "    chunk:",
                "      text_field: body",
                "      chunk_size: ${matrix.size}",
                "      chunk_overlap: 0",
            ]
        ),
    )
    _, _, models = load_project(project_path)
    assert len(models) == 3
    sizes = sorted(m.chunk.chunk_size for m in models if m.chunk)  # type: ignore[union-attr]
    assert sizes == [256, 512, 1024]
    assert all(isinstance(m.chunk.chunk_size, int) for m in models if m.chunk)  # type: ignore[union-attr]


def test_yaml_level_validation_error_has_no_cause(tmp_path: Path) -> None:
    """Validation errors at load time must not chain the ValidationError as
    __cause__ (prevents credential leakage through exception chains)."""
    project_path = _make_project(
        tmp_path,
        "\n".join(
            [
                "version: 2",
                "models:",
                "  - name: bad",
                "    chunk:",
                "      text_field: body",
                "      unknown_field: oops",
            ]
        ),
    )
    with pytest.raises(ConfigError) as exc_info:
        load_project(project_path)
    assert exc_info.value.__cause__ is None
