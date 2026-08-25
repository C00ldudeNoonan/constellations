"""Promoting candidate judgments into golden sets (issue #380, #329 phase 3).

See `contract` for the reviewed file format and `golden_set` for the transform
that materializes it into the relation `retrieval_tests:` reads.
"""
from .contract import (
    GOLDEN_SET_VERSION,
    GoldenSetFile,
    PromotedQuery,
    PromotionError,
    PromotionEvidence,
    load_golden_set,
)

__all__ = [
    "GOLDEN_SET_VERSION",
    "GoldenSetFile",
    "PromotedQuery",
    "PromotionError",
    "PromotionEvidence",
    "load_golden_set",
]
