"""End-to-end tests for the built-in text transforms.

Each transform is invoked via its public `run(deps, ctx)` signature with a
TransformContext that carries `options`. No DuckDB / runner involvement —
that's tested separately. This file just locks in the per-transform contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from stel.adapters import parse_warehouse_config
from stel.text.transforms import (
    clean_encoding as t_clean_encoding,
)
from stel.text.transforms import (
    count_tokens as t_count_tokens,
)
from stel.text.transforms import (
    detect_language as t_detect_language,
)
from stel.text.transforms import (
    find_duplicates as t_find_duplicates,
)
from stel.text.transforms import (
    text_stats as t_text_stats,
)
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


def _deps(df: pl.DataFrame, dep_name: str = "upstream") -> dict[str, pl.DataFrame]:
    return {dep_name: df}


# ─── text_stats ──────────────────────────────────────────────────────────


def test_text_stats_default_fields() -> None:
    df = pl.DataFrame({"text": ["hello world.", "two sentences. another one."]})
    out = t_text_stats.run(_deps(df), _ctx({"text_field": "text"}))
    assert {"word_count", "char_count", "sentence_count", "paragraph_count"} <= set(out.columns)
    assert out["word_count"].to_list() == [2, 4]


def test_text_stats_emit_filter() -> None:
    df = pl.DataFrame({"body": ["hi there"]})
    out = t_text_stats.run(
        _deps(df),
        _ctx({"text_field": "body", "emit": ["word_count"]}),
    )
    assert "word_count" in out.columns
    assert "char_count" not in out.columns


def test_text_stats_with_prefix() -> None:
    df = pl.DataFrame({"text": ["hi"]})
    out = t_text_stats.run(
        _deps(df),
        _ctx({"text_field": "text", "prefix": "doc_", "emit": ["word_count"]}),
    )
    assert "doc_word_count" in out.columns


# ─── clean_encoding ──────────────────────────────────────────────────────


def test_clean_encoding_in_place() -> None:
    df = pl.DataFrame({"text": ["I donâ€™t know", "normal text"]})
    out = t_clean_encoding.run(_deps(df), _ctx({"text_field": "text"}))
    assert "â€™" not in out["text"][0]
    assert out["text"][1] == "normal text"


def test_clean_encoding_separate_output() -> None:
    df = pl.DataFrame({"text": ["donâ€™t"]})
    out = t_clean_encoding.run(
        _deps(df),
        _ctx({"text_field": "text", "output_field": "text_clean"}),
    )
    assert "text_clean" in out.columns
    assert out["text"][0] == "donâ€™t"  # original untouched
    assert "â€™" not in out["text_clean"][0]


# ─── detect_language ─────────────────────────────────────────────────────


def test_detect_language_default_column() -> None:
    df = pl.DataFrame(
        {"text": ["This is a sentence in English.", "Esto es español aquí."]}
    )
    out = t_detect_language.run(_deps(df), _ctx({"text_field": "text"}))
    assert "language" in out.columns
    assert out["language"].to_list() == ["en", "es"]


def test_detect_language_custom_output_with_fallback() -> None:
    df = pl.DataFrame({"text": ["hi"]})  # too short for detection
    out = t_detect_language.run(
        _deps(df),
        _ctx({"text_field": "text", "output_field": "lang", "default": "en"}),
    )
    assert out["lang"][0] == "en"


# ─── count_tokens ────────────────────────────────────────────────────────


def test_count_tokens_default() -> None:
    df = pl.DataFrame({"text": ["hello world", "", "one"]})
    out = t_count_tokens.run(_deps(df), _ctx({"text_field": "text"}))
    assert out["token_count"].to_list() == [2, 0, 1]


def test_count_tokens_model_alias() -> None:
    df = pl.DataFrame({"text": ["hello world"]})
    out = t_count_tokens.run(
        _deps(df),
        _ctx({"text_field": "text", "model": "gpt-4o"}),
    )
    assert out["token_count"][0] > 0


# ─── find_duplicates ─────────────────────────────────────────────────────


def test_find_duplicates_flags_clusters() -> None:
    df = pl.DataFrame(
        {
            "text": [
                "the quick brown fox jumps over the lazy dog",
                "the quick brown fox jumps over the lazy dog",  # dup
                "completely different content about cars and trucks",
            ]
        }
    )
    out = t_find_duplicates.run(
        _deps(df), _ctx({"text_field": "text", "threshold": 0.5, "shingle_size": 3})
    )
    groups = out["duplicate_group"].to_list()
    assert groups[0] is not None
    assert groups[0] == groups[1]  # both in same cluster
    assert groups[2] is None
