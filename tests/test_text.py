from __future__ import annotations

from stel.text import (
    clean_encoding,
    count_tokens,
    detect_language,
    near_duplicates,
    text_stats,
)


def test_count_tokens_simple() -> None:
    n = count_tokens("hello world")
    assert n == 2  # tiktoken/cl100k tokenizes these as two tokens


def test_count_tokens_empty() -> None:
    assert count_tokens("") == 0


def test_count_tokens_via_family_alias() -> None:
    # gpt-4 / openai aliases route to cl100k_base
    assert count_tokens("hello world", model="gpt-4") == 2
    assert count_tokens("hello world", model="openai") == 2


def test_count_tokens_via_o200k() -> None:
    n = count_tokens("hello world", model="o200k_base")
    assert n > 0


def test_clean_encoding_fixes_mojibake() -> None:
    bad = "I donâ€™t know"  # classic UTF-8-as-Latin-1 confusion
    out = clean_encoding(bad)
    assert "’" in out or "'" in out  # right single quote or ASCII apostrophe
    assert "â€™" not in out


def test_clean_encoding_pass_through() -> None:
    assert clean_encoding("hello") == "hello"
    assert clean_encoding("") == ""


def test_detect_language_english() -> None:
    assert detect_language("This is a sentence in English with enough words.") == "en"


def test_detect_language_spanish() -> None:
    assert detect_language(
        "Esto es una oración en español con suficientes palabras."
    ) == "es"


def test_detect_language_short_returns_default() -> None:
    assert detect_language("hi", default="en") == "en"
    assert detect_language("", default="en") == "en"
    assert detect_language("a", default=None) is None


def test_text_stats_basic() -> None:
    s = text_stats("Hello world. This is a test.\n\nNew paragraph here.")
    assert s.char_count > 0
    assert s.word_count == 9
    assert s.sentence_count == 3
    assert s.paragraph_count == 2


def test_text_stats_empty() -> None:
    s = text_stats("")
    assert s.char_count == 0
    assert s.word_count == 0
    assert s.sentence_count == 0
    assert s.paragraph_count == 0


def test_text_stats_single_paragraph() -> None:
    s = text_stats("Just one sentence with no breaks")
    assert s.paragraph_count == 1
    assert s.word_count == 6


def test_near_duplicates_finds_clusters() -> None:
    texts = [
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox jumps over the lazy dog",  # exact duplicate
        "the quick brown fox jumps over the very lazy dog",  # near duplicate
        "completely different content about cars and trucks driving fast",
    ]
    clusters = near_duplicates(texts, threshold=0.5, k=3)
    assert len(clusters) == 1
    cluster = clusters[0]
    assert 0 in cluster
    assert 1 in cluster
    assert 3 not in cluster


def test_near_duplicates_no_matches() -> None:
    texts = ["completely unique one", "totally different two", "third independent body"]
    clusters = near_duplicates(texts, threshold=0.9, k=3)
    assert clusters == []
