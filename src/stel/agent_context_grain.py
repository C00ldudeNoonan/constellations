"""The grains of the agent-context contract, kept free of any dataframe library.

The configuration models name a grain, and validating a project is configuration
work that must not import polars (issue #823, ALE-791). The contract's fields and
validators, which do need polars, stay in `agent_context.py`.
"""

from __future__ import annotations

from enum import StrEnum


class AgentContextGrain(StrEnum):
    DOCUMENT_REGISTRY = "document_registry"
    DOCUMENT_CHUNKS = "document_chunks"
    CONTEXT_ENTITY_LINKS = "context_entity_links"
