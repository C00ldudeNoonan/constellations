"""Near-duplicate detection: word shingles, MinHash signatures, banded LSH.

First-party and stdlib-only (issue #797): the signature hashes shingles with
``hashlib`` and derives ``num_perm`` permutations from one 64-bit hash with
fixed affine maps modulo a Mersenne prime. The maps come from a fixed seed,
so signatures are deterministic across processes and platforms.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from collections.abc import Sequence
from functools import lru_cache

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

_PRIME = (1 << 61) - 1
# Signature value for a text with no shingles; larger than any hash value.
_EMPTY = _PRIME
_SEED = 1
# Resolution of the numerical integration that picks the LSH banding.
_INTEGRATION_STEPS = 100

MinHashSignature = tuple[int, ...]


def _shingles(text: str, k: int = 5) -> list[str]:
    """k-word shingles. Word-level (not char-level) is more robust to
    cosmetic differences and matches how production dedupe pipelines
    typically work."""
    words = _TOKEN_RE.findall(text.lower())
    if len(words) < k:
        return [" ".join(words)] if words else []
    return [" ".join(words[i : i + k]) for i in range(len(words) - k + 1)]


@lru_cache(maxsize=8)
def _permutations(num_perm: int) -> tuple[tuple[int, int], ...]:
    rng = random.Random(_SEED)
    return tuple(
        (rng.randrange(1, _PRIME), rng.randrange(0, _PRIME)) for _ in range(num_perm)
    )


def minhash_signature(text: str, *, num_perm: int = 128, k: int = 5) -> MinHashSignature:
    """Build a MinHash signature for `text`: `num_perm` minimum hash values.

    `num_perm` trades accuracy for memory (128 is the default; 256 for higher
    precision). `k` is the shingle size in words. Two signatures of the same
    length estimate Jaccard similarity as the fraction of equal positions.
    """
    if num_perm < 1:
        raise ValueError(f"num_perm must be at least 1, got {num_perm}")
    perms = _permutations(num_perm)
    xs = [
        int.from_bytes(hashlib.blake2b(sh.encode("utf-8"), digest_size=8).digest(), "little")
        % _PRIME
        for sh in set(_shingles(text, k=k))
    ]
    if not xs:
        return (_EMPTY,) * num_perm
    return tuple(min([(a * x + b) % _PRIME for x in xs]) for a, b in perms)


def _band_error(threshold: float, bands: int, rows: int) -> float:
    """Equally weighted false-positive plus false-negative probability mass of
    a `bands` x `rows` banding around `threshold`."""

    def collide(s: float) -> float:
        return 1.0 - (1.0 - s**rows) ** bands

    step = 1.0 / _INTEGRATION_STEPS
    error = 0.0
    for n in range(_INTEGRATION_STEPS):
        s = (n + 0.5) * step
        error += (collide(s) if s < threshold else 1.0 - collide(s)) * step
    return error


@lru_cache(maxsize=32)
def _banding(threshold: float, num_perm: int) -> tuple[int, int]:
    """The (bands, rows) split of `num_perm` that best separates pairs above
    `threshold` from pairs below it."""
    best = (1, num_perm)
    best_error = float("inf")
    for bands in range(1, num_perm + 1):
        for rows in range(1, num_perm // bands + 1):
            error = _band_error(threshold, bands, rows)
            if error < best_error:
                best, best_error = (bands, rows), error
    return best


def near_duplicates(
    texts: Sequence[str],
    *,
    threshold: float = 0.8,
    num_perm: int = 128,
    k: int = 5,
) -> list[set[int]]:
    """Return clusters of `texts` indices with Jaccard similarity >= `threshold`.

    Uses MinHash + banded LSH for O(n) expected time. Returns one set per
    cluster; singletons are omitted. Near the threshold, membership is
    probabilistic.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    sigs = [minhash_signature(t, num_perm=num_perm, k=k) for t in texts]
    bands, rows = _banding(threshold, num_perm)

    buckets: list[defaultdict[MinHashSignature, list[int]]] = [
        defaultdict(list) for _ in range(bands)
    ]
    for i, sig in enumerate(sigs):
        for band in range(bands):
            buckets[band][sig[band * rows : (band + 1) * rows]].append(i)

    seen: set[int] = set()
    clusters: list[set[int]] = []
    for i, sig in enumerate(sigs):
        if i in seen:
            continue
        neighbors: set[int] = set()
        for band in range(bands):
            neighbors.update(buckets[band][sig[band * rows : (band + 1) * rows]])
        cluster = neighbors - seen
        if len(neighbors) > 1 and len(cluster) >= 2:
            clusters.append(cluster)
            seen |= cluster
    return clusters
