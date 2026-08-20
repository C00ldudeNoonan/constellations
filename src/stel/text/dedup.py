from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from ..optional_dependencies import import_optional_dependency

if TYPE_CHECKING:
    from datasketch import MinHash

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _shingles(text: str, k: int = 5) -> list[str]:
    """k-word shingles. Word-level (not char-level) is more robust to
    cosmetic differences and matches how production dedupe pipelines
    typically work."""
    words = _TOKEN_RE.findall(text.lower())
    if len(words) < k:
        return [" ".join(words)] if words else []
    return [" ".join(words[i : i + k]) for i in range(len(words) - k + 1)]


def minhash_signature(text: str, *, num_perm: int = 128, k: int = 5) -> MinHash:
    """Build a MinHash signature for `text`. `num_perm` trades accuracy for
    memory (128 is the datasketch default; 256 for higher precision).
    `k` is the shingle size in words."""
    datasketch = import_optional_dependency(
        "datasketch", extra="text", feature="Near-duplicate detection"
    )

    mh = datasketch.MinHash(num_perm=num_perm)
    for sh in _shingles(text, k=k):
        mh.update(sh.encode("utf-8"))
    return cast("MinHash", mh)


def near_duplicates(
    texts: Sequence[str],
    *,
    threshold: float = 0.8,
    num_perm: int = 128,
    k: int = 5,
) -> list[set[int]]:
    """Return clusters of `texts` indices with Jaccard similarity >= `threshold`.

    Uses MinHash + LSH for O(n) expected time. Returns one set per cluster;
    singletons are omitted.
    """
    datasketch = import_optional_dependency(
        "datasketch", extra="text", feature="Near-duplicate detection"
    )

    sigs: list[MinHash] = [
        minhash_signature(t, num_perm=num_perm, k=k) for t in texts
    ]
    lsh = datasketch.MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, sig in enumerate(sigs):
        lsh.insert(str(i), sig)

    seen: set[int] = set()
    clusters: list[set[int]] = []
    for i, sig in enumerate(sigs):
        if i in seen:
            continue
        neighbors = {int(j) for j in lsh.query(sig)}
        if len(neighbors) <= 1:
            continue
        cluster = neighbors - seen
        if len(cluster) >= 2:
            clusters.append(cluster)
            seen |= cluster
    return clusters
