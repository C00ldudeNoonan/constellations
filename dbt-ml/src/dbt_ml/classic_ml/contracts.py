"""Shared classic-ML literal contracts (issue #190, Workstream B).

Provider and analyzer identifiers used across the artifact layer and every
algorithm family. Imports nothing from the package, so it sits at the bottom
of the dependency graph.
"""

from __future__ import annotations

from typing import Literal

FeatureProvider = Literal["builtin.count", "builtin.tfidf", "builtin.hashing"]
ClassifierProvider = Literal["builtin.naive_bayes"]
Analyzer = Literal["word", "char", "char_wb"]
