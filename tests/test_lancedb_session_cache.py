"""LanceDB session cache budgets, per store role (issue #479).

`lancedb.connect()` with no `Session` takes the library's own defaults —
documented as "equivalent to a 6GB index cache and 1GB metadata cache" — which
a container ceiling is invisible to. That is #412's shape one store over: the
publisher had ~7 GB of budget it never asked for and could not bound.

Two claims are under test. First, that a role gets a defensible default and an
explicit profile setting beats it. Second — the one that would be expensive to
get wrong — that none of this reaches the store's *identity*, so tuning a
cache cannot reclassify a published collection.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from stel.retrieval import LanceDBConfig, LanceDBStore, StoreRole
from stel.retrieval import lancedb as lancedb_store
from stel.retrieval.lancedb import session_cache_budget

_MB = 1024 * 1024
_GB = 1024 * _MB


def _with_ceiling(monkeypatch: pytest.MonkeyPatch, ceiling: int | None) -> None:
    monkeypatch.setattr(
        lancedb_store, "container_memory_limit_bytes", lambda: ceiling
    )


@pytest.fixture(autouse=True)
def _off_a_container(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default every case to "no container", so a test that is about the role
    or the profile is not quietly also about the box it runs on. The ceiling
    cases opt back in with `_with_ceiling`."""
    _with_ceiling(monkeypatch, None)


def _config(**overrides: Any) -> LanceDBConfig:
    payload: dict[str, Any] = {"type": "lancedb", "path": "/tmp/lance"}
    payload.update(overrides)
    return LanceDBConfig.model_validate(payload)


def _store(tmp_path: Path, role: StoreRole, **overrides: Any) -> LanceDBStore:
    return LanceDBStore(
        _config(path=str(tmp_path / "lance"), **overrides),
        project_name="proj",
        target_name="dev",
        alias="default",
        role=role,
    )


# ─── the role defaults ──────────────────────────────────────────────────────


def test_publishing_is_bounded_rather_than_taking_lancedbs_7gb() -> None:
    """The publisher competes with the merge and the index build for one
    ceiling, and gains nothing from caching an index it is replacing."""
    budget = session_cache_budget(_config(), StoreRole.PUBLISH)

    assert budget == (256 * _MB, 64 * _MB)


def test_inspecting_takes_the_smallest_budget() -> None:
    """Compile, manifest and ledger admin read descriptors and row counts.
    They never touch an index, so they never need to cache one."""
    assert session_cache_budget(_config(), StoreRole.INSPECT) == (32 * _MB, 16 * _MB)


def test_serving_keeps_lancedbs_own_defaults() -> None:
    """Absent by design, not an oversight: ANN latency depends on the index
    staying resident, and bounding it here would trade away the thing the
    index exists to provide. An operator who needs serving bounded says so."""
    assert session_cache_budget(_config(), StoreRole.SERVE) is None


def test_the_default_role_is_the_cheapest_one() -> None:
    """A caller that has not thought about this is doing metadata work. Being
    wrong that way costs a cache miss; the other way costs a container."""
    store = LanceDBStore(
        _config(), project_name="proj", target_name="dev", alias="default"
    )

    assert store.role is StoreRole.INSPECT


# ─── the profile wins ───────────────────────────────────────────────────────


def test_an_explicit_setting_beats_the_role_default() -> None:
    budget = session_cache_budget(
        _config(index_cache_size_mb=512, metadata_cache_size_mb=128),
        StoreRole.PUBLISH,
    )

    assert budget == (512 * _MB, 128 * _MB)


def test_an_explicit_setting_bounds_serving_which_has_no_default() -> None:
    """The 20 GiB container hosting a query process is the case for this."""
    budget = session_cache_budget(
        _config(index_cache_size_mb=2048, metadata_cache_size_mb=256),
        StoreRole.SERVE,
    )

    assert budget == (2048 * _MB, 256 * _MB)


def test_bounding_one_cache_does_not_silently_disable_the_other() -> None:
    """Under a role with no default, naming only the index budget must leave
    the metadata cache at LanceDB's documented default rather than at zero —
    nobody asked for it to be turned off."""
    budget = session_cache_budget(_config(index_cache_size_mb=2048), StoreRole.SERVE)

    assert budget == (2048 * _MB, 1024 * _MB)


def test_a_partial_setting_under_a_role_fills_from_that_role() -> None:
    budget = session_cache_budget(_config(index_cache_size_mb=512), StoreRole.PUBLISH)

    assert budget == (512 * _MB, 64 * _MB)


@pytest.mark.parametrize("field", ["index_cache_size_mb", "metadata_cache_size_mb"])
def test_a_zero_or_negative_budget_is_refused(field: str) -> None:
    """Zero would disable the cache entirely, which is never what an operator
    tuning a size means."""
    for value in (0, -1):
        with pytest.raises(ValueError):
            _config(**{field: value})


# ─── the property that would be expensive to get wrong ─────────────────────


def test_cache_settings_stay_out_of_the_store_identity(tmp_path: Path) -> None:
    """A cache size is execution, not identity.

    The safe descriptor keys the state scope and feeds collection
    classification. If a cache budget reached it, tuning one would present as
    a different physical store — stranding state and reclassifying a published
    collection as changed. Byte-identical is the assertion, not "close".
    """
    plain = _store(tmp_path, StoreRole.PUBLISH)
    tuned = _store(
        tmp_path,
        StoreRole.PUBLISH,
        index_cache_size_mb=4096,
        metadata_cache_size_mb=512,
    )

    assert (
        plain.safe_descriptor().safe_target_identity
        == tuned.safe_descriptor().safe_target_identity
    )
    assert plain.state_descriptor("ctx") == tuned.state_descriptor("ctx")


def test_the_role_itself_does_not_change_the_store_identity(tmp_path: Path) -> None:
    """The same collection published then served is one store, not two."""
    publishing = _store(tmp_path, StoreRole.PUBLISH)
    serving = _store(tmp_path, StoreRole.SERVE)

    assert (
        publishing.safe_descriptor().safe_target_identity
        == serving.safe_descriptor().safe_target_identity
    )


def test_cache_settings_are_absent_from_routing_options() -> None:
    """`routing_options` is what feeds the descriptor; it takes storage
    routing only."""
    config = _config(
        storage_options={"region": "us-east-1"},
        index_cache_size_mb=4096,
        metadata_cache_size_mb=512,
    )

    assert config.routing_options() == {"region": "us-east-1"}


# ─── it actually reaches LanceDB ────────────────────────────────────────────


def test_the_budget_is_handed_to_lancedb_as_a_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The kwarg names are LanceDB's and are validated by it — a wrong one
    raises rather than being ignored — so this pins that we pass a real
    Session built from the resolved budget, not that we merely computed one."""
    import lancedb

    captured: dict[str, Any] = {}
    real_connect = lancedb.connect

    def spy(uri: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return real_connect(uri, **kwargs)

    monkeypatch.setattr(lancedb, "connect", spy)
    with _store(tmp_path, StoreRole.PUBLISH, index_cache_size_mb=64):
        pass

    assert isinstance(captured["session"], lancedb.Session)


def test_serving_connects_without_a_session_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unchanged behavior on the query path: no Session, LanceDB's defaults."""
    import lancedb

    captured: dict[str, Any] = {}
    real_connect = lancedb.connect

    def spy(uri: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return real_connect(uri, **kwargs)

    monkeypatch.setattr(lancedb, "connect", spy)
    with _store(tmp_path, StoreRole.SERVE):
        pass

    assert "session" not in captured


# ─── a detected container ceiling clamps defaults (issue #479, ask 3) ───────


def test_serving_on_a_small_container_does_not_take_lancedbs_7gb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The case this exists for. Serving keeps LanceDB's defaults by choice,
    but ~7 GB of cache is most of a 2 GiB container before the query process
    holds anything — and the cgroup is the only thing that knows."""
    _with_ceiling(monkeypatch, 2 * _GB)

    budget = session_cache_budget(_config(), StoreRole.SERVE)

    assert budget is not None
    index, metadata = budget
    # Half the ceiling between them, LanceDB's 6:1 shape preserved.
    assert index + metadata <= _GB
    assert index > metadata


def test_a_roomy_container_leaves_the_serving_default_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 20 GiB box allows 10 GiB and the default asks for 7, so the clamp
    must not bind — serving keeps LanceDB's own behavior exactly."""
    _with_ceiling(monkeypatch, 20 * _GB)

    assert session_cache_budget(_config(), StoreRole.SERVE) is None


def test_no_container_leaves_the_serving_default_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_ceiling(monkeypatch, None)

    assert session_cache_budget(_config(), StoreRole.SERVE) is None


def test_an_explicit_setting_is_never_clamped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Advisory, like #412's DuckDB detection: the operator may know the box
    is larger than the cgroup says, and an explicit number is a decision."""
    _with_ceiling(monkeypatch, 2 * _GB)

    budget = session_cache_budget(
        _config(index_cache_size_mb=4096, metadata_cache_size_mb=512),
        StoreRole.SERVE,
    )

    assert budget == (4096 * _MB, 512 * _MB)


def test_a_tiny_container_still_leaves_each_cache_usable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scaling must not drive a cache to zero, which would disable it."""
    _with_ceiling(monkeypatch, 8 * _MB)

    budget = session_cache_budget(_config(), StoreRole.SERVE)

    assert budget is not None
    assert all(size >= _MB for size in budget)


def test_the_publisher_default_already_fits_a_normal_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """320 MB total against a 20 GiB ceiling: nothing to clamp, so the
    publisher gets exactly its role default."""
    _with_ceiling(monkeypatch, 20 * _GB)

    assert session_cache_budget(_config(), StoreRole.PUBLISH) == (
        256 * _MB,
        64 * _MB,
    )
