from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..hashing import canonical_fingerprint
from .base import RetrievalConfigError, RetrievalStore, RetrievalStoreConfig

_REGISTRY: dict[str, type[RetrievalStore]] = {}


def register(cls: type[RetrievalStore]) -> type[RetrievalStore]:
    _REGISTRY[cls.store_type()] = cls
    return cls


def parse_store_config(raw: dict[str, Any]) -> RetrievalStoreConfig:
    store_type = str(raw.get("type", ""))
    cls = _REGISTRY.get(store_type)
    if cls is None:
        raise RetrievalConfigError(
            f"No retrieval store registered for type='{store_type}'. Known: {sorted(_REGISTRY)}"
        )
    try:
        return cls.config_model().model_validate(raw)
    except ValidationError as error:
        details = "; ".join(
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in error.errors(
                include_url=False,
                include_context=False,
                include_input=False,
            )
        )
        raise RetrievalConfigError(
            f"Invalid config for retrieval store type='{store_type}': {details}"
        ) from None


def create_store(
    config: RetrievalStoreConfig,
    *,
    project_name: str,
    target_name: str,
    alias: str,
) -> RetrievalStore:
    cls = _REGISTRY.get(config.type)
    if cls is None:
        raise RetrievalConfigError(
            f"No retrieval store registered for type='{config.type}'. Known: {sorted(_REGISTRY)}"
        )
    return cls(
        config,
        project_name=project_name,
        target_name=target_name,
        alias=alias,
    )


def absolutize_store_config(
    config: RetrievalStoreConfig, project_dir: Path
) -> RetrievalStoreConfig:
    return config.absolutize(project_dir)


def store_class(store_type: str) -> type[RetrievalStore]:
    cls = _REGISTRY.get(store_type)
    if cls is None:
        raise RetrievalConfigError(
            f"No retrieval store registered for type='{store_type}'. Known: {sorted(_REGISTRY)}"
        )
    return cls


def list_store_types() -> list[str]:
    return sorted(_REGISTRY)


def collection_config_fingerprint(search_config: dict[str, Any], *, store_type: str) -> str:
    return canonical_fingerprint(
        {
            "contract_version": 1,
            "store_type": store_type,
            "store_implementation": store_class(store_type).implementation_identity(),
            "search": search_config,
        },
        domain="dbt-ml-search-collection-config",
    )
