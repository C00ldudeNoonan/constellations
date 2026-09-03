"""Cloud object-store (s3://, gs://, az://) support for the lancedb store (#271).

Real cloud round-trips are credential-gated and live elsewhere; these unit tests
pin the wrapper's routing: cloud URIs bypass local-filesystem resolution and the
local mkdir, non-secret routing rides in the store identity while credential
references resolve only at connect, and the single-host publisher lock plus the
identity both key off a canonical physical target. Local-path behavior stays
byte-for-byte the same.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import lancedb
import pytest

from stel.credentials import CredentialReference
from stel.hashing import canonical_fingerprint
from stel.retrieval import LanceDBStore, RetrievalError, StoreRole, parse_store_config
from stel.retrieval.lancedb import LanceDBConfig


def _config(path: str, **extra: object) -> LanceDBConfig:
    parsed = parse_store_config({"type": "lancedb", "path": path, **extra})
    assert isinstance(parsed, LanceDBConfig)
    return parsed


def _store(config: LanceDBConfig) -> LanceDBStore:
    return LanceDBStore(
        config, project_name="demo", target_name="dev", alias="primary",
        role=StoreRole.PUBLISH,
    )


@pytest.mark.parametrize(
    "uri",
    ["gs://bucket/prefix", "s3://bucket/prefix", "az://container/prefix",
     "S3://Bucket/Prefix", "abfss://fs@acct/prefix"],
)
def test_cloud_uris_are_detected(uri: str) -> None:
    assert _config(uri).is_cloud_uri is True


@pytest.mark.parametrize(
    "path",
    ["./target/lancedb", "/abs/local/lancedb", "relative/dir", "C:/data/lance"],
)
def test_local_paths_are_not_cloud(path: str) -> None:
    assert _config(path).is_cloud_uri is False


def test_storage_options_non_secret_routing_parsed() -> None:
    config = _config(
        "gs://bucket/prefix",
        storage_options={"region": "us-central1", "endpoint": "https://x"},
    )
    assert config.storage_options == {"region": "us-central1", "endpoint": "https://x"}


def test_storage_options_default_empty() -> None:
    config = _config("gs://bucket/prefix")
    assert config.storage_options == {}
    assert config.storage_options_env == {}


@pytest.mark.parametrize(
    "secret_key",
    ["aws_secret_access_key", "aws_session_token", "account_key", "sas_token",
     "password", "api_key", "gcs_credential"],
)
def test_secret_looking_key_rejected_from_routing(secret_key: str) -> None:
    with pytest.raises(Exception, match="non-secret routing only"):
        _config("gs://bucket/prefix", storage_options={secret_key: "x"})


def test_storage_options_env_reference_is_redacted() -> None:
    config = _config(
        "s3://bucket/prefix",
        storage_options_env={"aws_secret_access_key": "MY_AWS_SECRET"},
    )
    reference = config.storage_options_env["aws_secret_access_key"]
    assert isinstance(reference, CredentialReference)
    # The env-var NAME is a reference, redacted from every dump surface.
    assert "MY_AWS_SECRET" not in repr(config)
    assert "MY_AWS_SECRET" not in config.model_dump_json()


def test_empty_path_rejected() -> None:
    with pytest.raises(Exception, match="non-empty local path or cloud URI"):
        _config("   ")


@pytest.mark.parametrize("bad", ["s3://", "gs://", "gs:///prefix", "az:///c/p"])
def test_malformed_cloud_uri_rejected_at_parse_time(bad: str) -> None:
    with pytest.raises(Exception, match="must include a bucket/container"):
        _config(bad)


def test_absolutize_leaves_cloud_uri_untouched(tmp_path: Path) -> None:
    config = _config("gs://bucket/prefix")
    assert config.absolutize(tmp_path) is config
    assert config.absolutize(tmp_path).path == "gs://bucket/prefix"


def test_absolutize_resolves_local_relative_path(tmp_path: Path) -> None:
    config = _config("sub/lance")
    resolved = config.absolutize(tmp_path)
    assert Path(resolved.path) == (tmp_path / "sub" / "lance").resolve()


def test_local_data_path_raises_for_cloud() -> None:
    with pytest.raises(RetrievalError, match="cloud_no_local_path"):
        _config("gs://bucket/prefix").local_data_path()


def test_connect_passes_cloud_uri_and_storage_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_connect(target: str, **kwargs: object) -> object:
        captured["target"] = target
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(lancedb, "connect", fake_connect)
    config = _config(
        "gs://bucket/prefix", storage_options={"region": "us-central1"}
    )
    # A passing __enter__ also proves the local mkdir path was not taken:
    # local_data_path() raises for a cloud store, so any mkdir attempt would
    # surface as a RetrievalError here.
    with _store(config):
        pass
    assert captured["target"] == "gs://bucket/prefix"
    # Asserted by key: a session cache budget also rides here now (#479),
    # and this test is about the storage options.
    assert captured["kwargs"]["storage_options"] == {"region": "us-central1"}


def test_connect_local_path_omits_storage_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def fake_connect(target: str, **kwargs: object) -> object:
        captured["target"] = target
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(lancedb, "connect", fake_connect)
    local = tmp_path / "lance"
    config = _config(str(local))
    with _store(config):
        pass
    assert captured["target"] == str(local)
    assert "storage_options" not in captured["kwargs"]
    # Local stores still create their data directory eagerly.
    assert local.is_dir()


def test_storage_options_absent_when_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_connect(target: str, **kwargs: object) -> object:
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(lancedb, "connect", fake_connect)
    with _store(_config("gs://bucket/prefix")):
        pass
    assert "storage_options" not in captured["kwargs"]


def test_connect_resolves_credential_reference_at_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_connect(target: str, **kwargs: object) -> object:
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(lancedb, "connect", fake_connect)
    monkeypatch.setenv("MY_AWS_SECRET", "resolved-secret-value")
    config = _config(
        "s3://bucket/prefix",
        storage_options={"region": "us-east-1"},
        storage_options_env={"aws_secret_access_key": "MY_AWS_SECRET"},
    )
    with _store(config):
        pass
    # Routing rides verbatim; the reference is resolved from the env only here.
    assert captured["kwargs"]["storage_options"] == {
        "region": "us-east-1",
        "aws_secret_access_key": "resolved-secret-value",
    }


def test_connect_missing_credential_reference_raises_before_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fake_connect(target: str, **kwargs: object) -> object:
        nonlocal called
        called = True
        return object()

    monkeypatch.setattr(lancedb, "connect", fake_connect)
    monkeypatch.delenv("MISSING_SECRET", raising=False)
    config = _config(
        "s3://bucket/prefix",
        storage_options_env={"aws_secret_access_key": "MISSING_SECRET"},
    )
    with pytest.raises(RetrievalError, match="storage_credential_missing"):
        with _store(config):
            pass
    assert called is False


def test_cloud_publisher_lock_excludes_second_publisher(tmp_path: Path) -> None:
    lock_dir = str(tmp_path / "shared-locks")
    config = _config("gs://bucket/prefix", publisher_lock_dir=lock_dir)
    store = _store(config)
    other = _store(config)
    # A cloud store's fence must not try to mkdir a gs:// path; it serializes
    # publishers on this host via the shared lock dir, same guarantee as local.
    with store.publisher_fence("demo__dev__context"):
        with pytest.raises(RetrievalError, match="publisher_lock_held"):
            with other.publisher_fence("demo__dev__context"):
                pass
    with other.publisher_fence("demo__dev__context"):
        pass


def test_cloud_publisher_lock_dir_is_keyed_by_target() -> None:
    a = _store(_config("gs://bucket/prefix-a"))._lock_dir()
    b = _store(_config("gs://bucket/prefix-b"))._lock_dir()
    same = _store(_config("gs://bucket/prefix-a"))._lock_dir()
    assert a != b
    assert a == same
    # Never a local mkdir on the object-store namespace.
    assert "bucket" not in a.as_posix()


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [("s3a://bucket/p", "s3://bucket/p"), ("gcs://bucket/p", "gs://bucket/p"),
     ("GS://bucket/p", "gs://bucket/p"), ("s3://bucket/p/", "s3://bucket/p")],
)
def test_scheme_aliases_share_lock_and_identity(alias: str, canonical: str) -> None:
    # Equivalent spellings must resolve to one physical target for both the
    # lock (so concurrent publishers actually contend) and the state scope.
    assert _store(_config(alias))._lock_dir() == _store(_config(canonical))._lock_dir()
    assert (
        _store(_config(alias)).safe_descriptor().safe_target_identity
        == _store(_config(canonical)).safe_descriptor().safe_target_identity
    )


def test_endpoint_routing_yields_distinct_target(tmp_path: Path) -> None:
    # Same URI, different endpoint = a different physical store: distinct state
    # scope (so B's rows aren't reconciled against A's state) AND distinct lock.
    a = _config("s3://bucket/p", storage_options={"endpoint": "https://a.example"})
    b = _config("s3://bucket/p", storage_options={"endpoint": "https://b.example"})
    assert (
        _store(a).safe_descriptor().safe_target_identity
        != _store(b).safe_descriptor().safe_target_identity
    )
    override = str(tmp_path / "locks")
    a2 = _config(
        "s3://bucket/p",
        storage_options={"endpoint": "https://a.example"},
        publisher_lock_dir=override,
    )
    b2 = _config(
        "s3://bucket/p",
        storage_options={"endpoint": "https://b.example"},
        publisher_lock_dir=override,
    )
    assert _store(a2)._lock_dir() != _store(b2)._lock_dir()


def test_cloud_lock_dir_is_independent_of_tmpdir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The single-host lock is worthless if two publishers on one host resolve
    to different dirs. A per-process TMPDIR must NOT move the default lock dir."""
    store = _store(_config("gs://bucket/prefix"))
    monkeypatch.setenv("TMPDIR", "/tmp/publisher-one")
    monkeypatch.setenv("TMP", "/tmp/publisher-one")
    monkeypatch.setenv("TEMP", "/tmp/publisher-one")
    first = store._lock_dir()
    monkeypatch.setenv("TMPDIR", "/tmp/publisher-two")
    monkeypatch.setenv("TMP", "/tmp/publisher-two")
    monkeypatch.setenv("TEMP", "/tmp/publisher-two")
    second = store._lock_dir()
    assert first == second


def test_publisher_lock_dir_override_relocates_lock(tmp_path: Path) -> None:
    override = tmp_path / "vol" / "locks"
    store = _store(_config("gs://bucket/prefix", publisher_lock_dir=str(override)))
    assert store._lock_dir().is_relative_to(override)


def test_local_store_lock_stays_co_located_by_default(tmp_path: Path) -> None:
    data = tmp_path / "lance"
    store = _store(_config(str(data)))
    # Backward-compatible: local publisher lock still lives in the data dir.
    assert store._lock_dir() == data


def test_credential_references_do_not_affect_identity() -> None:
    # Secret references are not part of the physical target: switching which env
    # var holds the key must not change the state scope, and no secret material
    # (nor the env-var name) can appear in the identity fingerprint.
    base = _config("s3://bucket/p")
    with_ref = _config(
        "s3://bucket/p", storage_options_env={"aws_secret_access_key": "MY_ENV"}
    )
    id_base = _store(base).safe_descriptor().safe_target_identity
    id_ref = _store(with_ref).safe_descriptor().safe_target_identity
    assert id_base == id_ref
    assert "MY_ENV" not in id_ref


def test_safe_descriptor_local_identity_is_posix_normalized(tmp_path: Path) -> None:
    local = (tmp_path / "lance").resolve()
    id_one = _store(_config(str(local))).safe_descriptor().safe_target_identity
    id_two = _store(_config(local.as_posix())).safe_descriptor().safe_target_identity
    assert id_one == id_two


def test_local_identity_unchanged_by_new_fields(tmp_path: Path) -> None:
    # Backward-compat guard: a local store's identity must still be the exact
    # pre-routing fingerprint {store_type, path}, so existing state scopes and
    # published collections keep resolving after this change.
    local = (tmp_path / "lance").resolve()
    identity = _store(_config(str(local))).safe_descriptor().safe_target_identity
    expected = canonical_fingerprint(
        {"store_type": "lancedb", "path": local.as_posix()},
        domain="dbt-ml-safe-retrieval-target",
    )
    assert identity == expected
