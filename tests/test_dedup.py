"""First-party MinHash + LSH near-duplicate detection (issue #797)."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from stel.text import dedup, minhash_signature, near_duplicates

_BASE = (
    "the committee reviewed the quarterly results and agreed that revenue growth "
    "was driven mainly by strong demand in the cloud segment while margins "
    "improved because of lower hosting costs across the region"
)


def _doc(topic: str, n: int = 40) -> str:
    return " ".join(f"{topic}{i}" for i in range(n))


def _estimate(a: str, b: str, *, num_perm: int, k: int) -> float:
    sa = minhash_signature(a, num_perm=num_perm, k=k)
    sb = minhash_signature(b, num_perm=num_perm, k=k)
    return sum(x == y for x, y in zip(sa, sb, strict=True)) / num_perm


def test_signature_is_a_tuple_of_num_perm_ints() -> None:
    sig = minhash_signature(_BASE, num_perm=64, k=3)
    assert isinstance(sig, tuple)
    assert len(sig) == 64
    assert all(isinstance(v, int) for v in sig)


def test_signature_is_pinned_across_processes() -> None:
    """The hash family is a contract: a changed seed, hash or prime would move
    every near-threshold cluster, so it must fail here rather than silently."""
    sig = minhash_signature("the quick brown fox jumps over the lazy dog", num_perm=4, k=3)
    assert sig == (
        160400492161758006,
        704786019256672597,
        1445730175855926,
        309060058869619526,
    )


def test_signature_ignores_case_punctuation_and_shingle_repeats() -> None:
    a = minhash_signature("Alpha beta, gamma delta epsilon!", k=2)
    b = minhash_signature("alpha  BETA gamma delta   epsilon", k=2)
    assert a == b
    assert minhash_signature("a b a b a b", k=2) == minhash_signature("a b a b", k=2)


def test_empty_text_signatures_are_equal_and_distinct_from_content() -> None:
    empty = minhash_signature("", num_perm=16)
    assert empty == minhash_signature("   ...  ", num_perm=16)
    assert empty != minhash_signature("something", num_perm=16)


@pytest.mark.parametrize("num_perm", [0, -3])
def test_signature_rejects_non_positive_num_perm(num_perm: int) -> None:
    with pytest.raises(ValueError, match="num_perm"):
        minhash_signature("text", num_perm=num_perm)


def test_signature_estimates_known_jaccard() -> None:
    # 10 words, k=1 -> 10 shingles each; 5 shared, 15 in the union: J = 1/3.
    a = "a0 a1 a2 a3 a4 s0 s1 s2 s3 s4"
    b = "b0 b1 b2 b3 b4 s0 s1 s2 s3 s4"
    estimate = _estimate(a, b, num_perm=512, k=1)
    assert estimate == pytest.approx(1 / 3, abs=0.08)
    assert _estimate(a, a, num_perm=64, k=1) == 1.0
    assert _estimate(a, _doc("z", 10), num_perm=512, k=1) < 0.05


def test_near_duplicates_clusters_exact_and_near_copies() -> None:
    texts = [
        _BASE,
        _BASE,
        _BASE + " today",
        _doc("cars"),
        _doc("boats"),
    ]
    assert near_duplicates(texts, threshold=0.8, k=3) == [{0, 1, 2}]


def test_near_duplicates_omits_singletons() -> None:
    texts = [_doc("a"), _doc("b"), _doc("c")]
    assert near_duplicates(texts, threshold=0.5, k=3) == []


def test_near_duplicates_keeps_separate_clusters_separate() -> None:
    first = _doc("alpha", 60)
    second = _doc("beta", 60)
    texts = [first, second, first + " x", second + " y", _doc("gamma", 60)]
    clusters = near_duplicates(texts, threshold=0.8, k=3)
    assert sorted(map(sorted, clusters)) == [[0, 2], [1, 3]]


def test_near_duplicates_threshold_separates_a_half_similar_pair() -> None:
    # 20 shared + 10 + 10 unique words, k=1: J = 20/40 = 0.5.
    shared = " ".join(f"s{i}" for i in range(20))
    a = shared + " " + " ".join(f"a{i}" for i in range(10))
    b = shared + " " + " ".join(f"b{i}" for i in range(10))
    assert near_duplicates([a, b], threshold=0.2, k=1) == [{0, 1}]
    assert near_duplicates([a, b], threshold=0.95, k=1) == []


def test_near_duplicates_empty_texts_are_duplicates_of_each_other() -> None:
    assert near_duplicates(["", "  ", "real words here"], k=3) == [{0, 1}]


def test_near_duplicates_no_input() -> None:
    assert near_duplicates([]) == []
    assert near_duplicates(["only one"]) == []


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.5])
def test_near_duplicates_rejects_threshold_outside_unit_interval(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        near_duplicates(["a", "b"], threshold=threshold)


def test_banding_uses_at_most_num_perm_rows_and_tracks_threshold() -> None:
    bands_lo, rows_lo = dedup._banding(0.3, 128)
    bands_hi, rows_hi = dedup._banding(0.9, 128)
    assert bands_lo * rows_lo <= 128
    assert bands_hi * rows_hi <= 128
    # A stricter threshold needs more rows per band (fewer chance collisions).
    assert rows_hi > rows_lo


def test_text_extra_no_longer_declares_datasketch() -> None:
    pyproject = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )
    extras = pyproject["project"]["optional-dependencies"]
    assert not any("datasketch" in dep for deps in extras.values() for dep in deps)
