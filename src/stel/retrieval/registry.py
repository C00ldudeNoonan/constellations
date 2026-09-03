from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..hashing import canonical_fingerprint
from .base import (
    RetrievalConfigError,
    RetrievalStore,
    RetrievalStoreConfig,
    StoreRole,
)
from .evolution import semantic_search_config

_REGISTRY: dict[str, type[RetrievalStore]] = {}


def register[StoreT: RetrievalStore](cls: type[StoreT]) -> type[StoreT]:
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
    role: StoreRole = StoreRole.INSPECT,
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
        role=role,
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


def collection_descriptor(
    search_config: dict[str, Any], *, store_type: str
) -> dict[str, Any]:
    """The stored description of a published collection's shape (issue #344).

    Only the semantic projection of the search config takes part: cadence and
    routing fields are excluded so tuning `batch_size` cannot invalidate an
    index. The store contract travels with it because a change of store
    implementation invalidates the collection just as a config change does.
    """
    return {
        "contract_version": 2,
        "store_type": store_type,
        "store_implementation": store_class(store_type).implementation_identity(),
        "search": semantic_search_config(search_config),
    }


def collection_config_fingerprint(search_config: dict[str, Any], *, store_type: str) -> str:
    """Digest of the descriptor, for cheap equality in the serving ledger.

    Change *detection* uses this; change *classification* needs the descriptor
    itself, since a digest cannot say which field moved.
    """
    return canonical_fingerprint(
        collection_descriptor(search_config, store_type=store_type),
        domain="dbt-ml-search-collection-config",
    )


def legacy_collection_config_fingerprint(
    search_config: dict[str, Any], *, store_type: str
) -> str:
    """The pre-#344 fingerprint: contract 1, over the whole search config.

    Kept only to recognize a collection stamped before the descriptor existed
    and prove its configuration is unchanged, so it can be re-stamped in place
    instead of rebuilt. Removable once no collection carries a v1 stamp
    (the #321 category-1 rule).
    """
    return canonical_fingerprint(
        {
            "contract_version": 1,
            "store_type": store_type,
            "store_implementation": store_class(store_type).implementation_identity(),
            "search": search_config,
        },
        domain="dbt-ml-search-collection-config",
    )
