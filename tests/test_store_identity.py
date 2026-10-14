"""A store's identity is declared, not derived from where the store sits (#666).

astrolabe's prod LanceDB store lives at `gs://.../lancedb` while every process
that reads it runs outside GCP, so each build and each query pays internet
egress: 1.43 TB and roughly $150 over two months, about $70 of that per index
rebuild. Reading a local copy is the fix -- but a store's state scope, ledger
row and publication state were all keyed on its *location*, so moving the bytes
produced a store that had never published anything, and the next run re-embedded
the corpus.

These pin the three properties that make the move safe: a store that declares no
identity keeps the fingerprint it already shipped, a declared identity survives
a change of location, and a declared label can never be mistaken for a path.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from stel.adapters import StateScope
from stel.cli_services.serving import _ServingScopes, _sources_to_rekey
from stel.hashing import canonical_fingerprint
from stel.retrieval import (
    DuckDBStore,
    LanceDBStore,
    StoreRole,
    parse_store_config,
)
from stel.retrieval.base import RetrievalConfigError
from stel.retrieval.duckdb import DuckDBConfig
from stel.retrieval.lancedb import LanceDBConfig

# The digests v0.21.x shipped, read off an unmodified tree. Every published
# collection's state scope, serving-ledger row and publisher lock is keyed on
# this value, so a change here is not a refactor: it is every existing index
# reading as unpublished, and a corpus re-embedded to recover from it.
_LOCAL_IDENTITY = "90500be31b4af309f72535d7435b7ab7"
_CLOUD_ROUTED_IDENTITY = "1f3aaf7e4170eed0990d02f101969d36"
_CLOUD_PLAIN_IDENTITY = "c179fcc74d30f736f989e8ee0846a5b5"


def _lance(path: str, **extra: object) -> LanceDBConfig:
    parsed = parse_store_config({"type": "lancedb", "path": path, **extra})
    assert isinstance(parsed, LanceDBConfig)
    return parsed


def _duck(path: str, **extra: object) -> DuckDBConfig:
    parsed = parse_store_config({"type": "duckdb", "path": path, **extra})
    assert isinstance(parsed, DuckDBConfig)
    return parsed


def _lance_store(config: LanceDBConfig) -> LanceDBStore:
    return LanceDBStore(
        config,
        project_name="econ",
        target_name="prod",
        alias="primary",
        role=StoreRole.INSPECT,
    )


def _duck_store(config: DuckDBConfig) -> DuckDBStore:
    return DuckDBStore(
        config,
        project_name="econ",
        target_name="prod",
        alias="primary",
        role=StoreRole.INSPECT,
    )


def _scope_at(config: LanceDBConfig, logical: str = "sec_chunks") -> StateScope:
    descriptor = _lance_store(config).state_descriptor(logical).descriptor()
    return StateScope.for_target_descriptor(
        "sec_search", stage="retrieval_publish", descriptor=descriptor
    )


def test_a_store_that_declares_no_identity_keeps_the_fingerprint_it_shipped() -> None:
    """The compatibility promise, as literals rather than as a recomputation.

    Deriving the expected value the way the code derives it would pass against
    any payload change made in both places at once, which is exactly the
    mistake that would strand a live index.
    """
    assert (
        _lance_store(_lance("/srv/lancedb")).safe_descriptor().safe_target_identity
        == _LOCAL_IDENTITY
    )
    routed = _lance("gs://bucket/prefix", storage_options={"region": "us-central1"})
    assert (
        _lance_store(routed).safe_descriptor().safe_target_identity
        == _CLOUD_ROUTED_IDENTITY
    )
    assert (
        _lance_store(_lance("gs://bucket/prefix"))
        .safe_descriptor()
        .safe_target_identity
        == _CLOUD_PLAIN_IDENTITY
    )


def test_a_duckdb_store_that_declares_no_identity_keeps_the_pre_666_payload() -> None:
    """The same promise for the other store, whose derived location is an
    absolute resolved path and so cannot be pinned as a literal across
    platforms. The payload shape is what is pinned instead."""
    config = _duck("./target/retrieval.duckdb")
    expected = canonical_fingerprint(
        {"store_type": "duckdb", "path": config.identity_key()},
        domain="dbt-ml-safe-retrieval-target",
    )
    assert _duck_store(config).safe_descriptor().safe_target_identity == expected


def test_a_declared_identity_is_one_store_in_two_places() -> None:
    """The point of the field: the store astrolabe publishes to GCS and the
    local copy it reads are the same store, so the local copy is already
    published and its ledger row is already there."""
    cloud = _lance(
        "gs://econ-bucket/lancedb",
        identity="econ-prod",
        storage_options={"region": "us-central1"},
    )
    local = _lance("/srv/lancedb", identity="econ-prod")

    assert (
        _lance_store(cloud).safe_descriptor()
        == _lance_store(local).safe_descriptor()
    )
    # And through to the scope the publication state and ledger row live under.
    assert _scope_at(cloud).target_identity == _scope_at(local).target_identity


def test_a_declared_identity_is_not_the_path_that_spells_it() -> None:
    """A declared label is keyed apart from a derived location, so no label can
    collide with the fingerprint some other store derives from its path."""
    labelled = _lance("/srv/lancedb", identity="econ-prod")
    path_shaped = _lance("econ-prod")
    assert (
        _lance_store(labelled).safe_descriptor().safe_target_identity
        != _lance_store(path_shaped).safe_descriptor().safe_target_identity
    )


def test_a_declared_identity_ignores_routing() -> None:
    """Routing is part of a *location*, so it cannot survive into a declared
    identity: the local copy has no region, and it has to be the same store as
    the bucket it was copied from."""
    one = _lance(
        "gs://econ-bucket/lancedb",
        identity="econ-prod",
        storage_options={"region": "us-central1"},
    )
    two = _lance(
        "gs://econ-bucket/lancedb",
        identity="econ-prod",
        storage_options={"region": "europe-west4"},
    )
    assert (
        _lance_store(one).safe_descriptor().safe_target_identity
        == _lance_store(two).safe_descriptor().safe_target_identity
    )


@pytest.mark.parametrize(
    "value",
    [
        "gs://bucket/lancedb",
        "/srv/lancedb",
        "C:\\data\\lancedb",
        "./target/lancedb",
        "has space",
        "-leading-dash",
        ".leading-dot",
        "",
        "x" * 129,
    ],
)
def test_an_identity_must_be_a_label_and_a_location_is_refused(value: str) -> None:
    """A location is the one thing an identity cannot be -- it exists to stop
    the location being the identity -- and an operator who pastes the old URI
    in has to find out at the profile boundary, not by stranding a scope."""
    with pytest.raises(RetrievalConfigError) as caught:
        _lance("/srv/lancedb", identity=value)
    assert "identity must be a stable label" in str(caught.value)


@pytest.mark.parametrize("value", ["econ-prod", "econ.prod:v2", "a", "0", "x" * 128])
def test_a_label_is_accepted(value: str) -> None:
    assert _lance("/srv/lancedb", identity=value).identity == value


def test_absolutizing_a_relative_path_keeps_the_declared_identity() -> None:
    """A profile's relative `path` is absolutized when the profile resolves,
    and that copy is the one code path every project with a relative store path
    takes. If it dropped the identity the declared label would vanish silently,
    and every scope would re-key to the absolute location."""
    declared = _lance("./target/lancedb", identity="econ-prod")

    absolute = declared.absolutize(Path("/srv/project"))

    assert absolute.identity == "econ-prod"
    assert absolute.path != declared.path
    assert (
        _lance_store(absolute).safe_descriptor().safe_target_identity
        == _lance_store(declared).safe_descriptor().safe_target_identity
    )


def _serving_scopes(store_config: LanceDBConfig) -> _ServingScopes:
    legacy = StateScope.for_target_descriptor(
        "sec_search",
        stage="retrieval_publish",
        descriptor={"store_type": "lancedb", "physical_collection": "econ_prod_sec"},
    )
    return _ServingScopes(
        scope=_scope_at(store_config),
        legacy_scope=legacy,
        resolved=cast(Any, SimpleNamespace(target_name="prod")),
        context=("primary", "lancedb", store_config.path),
        store_config=store_config,
        project_name="econ",
        logical_collection="sec_chunks",
        model_name="sec_search",
    )


def test_the_old_location_resolves_to_the_scope_that_published_there() -> None:
    """`migrate-scope --from-path` has to name the scope the rows are actually
    under: the one the store had before it declared an identity and moved. Get
    this wrong and the command reports success having moved nothing."""
    scopes = _serving_scopes(_lance("/srv/lancedb", identity="econ-prod"))

    moved_from = scopes.scope_at("gs://econ-bucket/lancedb")

    expected = _scope_at(_lance("gs://econ-bucket/lancedb"))
    assert moved_from.target_identity == expected.target_identity
    assert moved_from.target_identity != scopes.scope.target_identity


def test_the_old_location_keeps_the_routing_the_fingerprint_folded_in() -> None:
    """The old scope's fingerprint included the store's non-secret routing, so
    re-deriving it without the routing would look for rows under a key nothing
    ever wrote."""
    config = _lance(
        "/srv/lancedb",
        identity="econ-prod",
        storage_options={"region": "us-central1"},
    )
    scopes = _serving_scopes(config)

    moved_from = scopes.scope_at("gs://econ-bucket/lancedb")

    expected = _scope_at(
        _lance("gs://econ-bucket/lancedb", storage_options={"region": "us-central1"})
    )
    assert moved_from.target_identity == expected.target_identity


def test_a_source_equal_to_the_destination_is_not_rekeyed() -> None:
    """`--from-path` naming the store's own location moves nothing, and a
    re-key onto the same scope matches every row -- so without this the
    command would report the whole corpus as migrated."""
    destination = _scope_at(_lance("/srv/lancedb", identity="econ-prod"))
    assert _sources_to_rekey(destination, [destination]) == []


def test_a_repeated_source_is_rekeyed_once() -> None:
    """The second pass would match nothing and report zero, which reads as a
    partial migration."""
    destination = _scope_at(_lance("/srv/lancedb", identity="econ-prod"))
    source = _scope_at(_lance("gs://econ-bucket/lancedb"))
    assert _sources_to_rekey(destination, [source, source]) == [source]
