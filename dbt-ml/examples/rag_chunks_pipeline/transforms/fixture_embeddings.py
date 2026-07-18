from __future__ import annotations

import hashlib
import math

import polars as pl

_DIMENSIONS = 8


def _embed(text: str) -> list[float]:
    values = [0.0] * _DIMENSIONS
    for token in text.casefold().split():
        digest = hashlib.sha256(token.encode()).digest()
        values[digest[0] % _DIMENSIONS] += 1.0 if digest[1] & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        values[0] = 1.0
        return values
    return [value / norm for value in values]


def run(deps: dict[str, pl.DataFrame]) -> pl.DataFrame:
    chunks = deps["document_chunks"]
    return chunks.with_columns(
        pl.col("text").map_elements(_embed, return_dtype=pl.List(pl.Float64)).alias("embedding")
    )
