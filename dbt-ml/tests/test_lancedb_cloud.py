"""Cloud object-store (s3://, gs://, az://) support for the lancedb store (#271).

Real cloud round-trips are credential-gated and live elsewhere; these unit tests
pin the wrapper's routing: cloud URIs bypass local-filesystem resolution and the
local mkdir, storage_options flows into `lancedb.connect()` without leaking into
identity, and the single-host publisher lock relocates to host-local scratch for
cloud stores while local-path behavior stays byte-for-byte the same.
"""
from __future__ import annotations

import json
from pathlib import Path

import lancedb
import pytest
from pydantic import SecretStr

from dbt_ml.retrieval import LanceDBStore, RetrievalError, parse_store_config
from dbt_ml.retrieval.lancedb import LanceDBConfig


def _config(path: str, **extra: object) -> LanceDBConfig:
    parsed = parse_store_config({"type": "lancedb", "path": path, **extra})
    assert isinstance(parsed, LanceDBConfig)
    return parsed


def _store(config: LanceDBConfig) -> LanceDBStore:
    return LanceDBStore(
        config, project_name="demo", target_name="dev", alias="primary"
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


def test_storage_options_parsed() -> None:
    config = _config(
        "gs://bucket/prefix",
        storage_options={"google_service_account": "/run/secrets/sa.json"},
    )
    assert {k: v.get_secret_value() for k, v in config.storage_options.items()} == {
        "google_service_account": "/run/secrets/sa.json"
    }


def test_storage_options_default_empty() -> None:
    assert _config("gs://bucket/prefix").storage_options == {}


def test_storage_options_are_redacted_on_every_serialization_surface() -> None:
    config = _config(
        "gs://bucket/prefix",
        storage_options={"aws_secret_access_key": "SUPERSECRET"},
    )
    # Values are SecretStr, so no diagnostic/serialization surface prints them.
    assert "SUPERSECRET" not in repr(config)
    assert "SUPERSECRET" not in str(config)
    assert "SUPERSECRET" not in config.model_dump_json()
    assert "SUPERSECRET" not in json.dumps(config.model_dump(mode="json"))
    assert isinstance(config.storage_options["aws_secret_access_key"], SecretStr)


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
    captured: dict[str, object] = {}

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
    assert captured["kwargs"] == {"storage_options": {"region": "us-central1"}}


def test_connect_local_path_omits_storage_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

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
    assert captured["kwargs"] == {}
    # Local stores still create their data directory eagerly.
    assert local.is_dir()


def test_storage_options_absent_when_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_connect(target: str, **kwargs: object) -> object:
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(lancedb, "connect", fake_connect)
    with _store(_config("gs://bucket/prefix")):
        pass
    assert captured["kwargs"] == {}


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


def test_cloud_publisher_lock_dir_is_keyed_by_uri() -> None:
    a = _store(_config("gs://bucket/prefix-a"))._lock_dir()
    b = _store(_config("gs://bucket/prefix-b"))._lock_dir()
    same = _store(_config("gs://bucket/prefix-a"))._lock_dir()
    assert a != b
    assert a == same
    # Never a local mkdir on the object-store namespace.
    assert "bucket" not in a.as_posix()


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


def test_safe_descriptor_stable_and_excludes_storage_options(tmp_path: Path) -> None:
    base = _config("gs://bucket/prefix")
    with_secret = _config(
        "gs://bucket/prefix", storage_options={"aws_secret_access_key": "SECRET"}
    )
    id_base = _store(base).safe_descriptor().safe_target_identity
    id_secret = _store(with_secret).safe_descriptor().safe_target_identity
    # storage_options must not perturb identity — and the secret must not be in it.
    assert id_base == id_secret
    assert "SECRET" not in id_base


def test_safe_descriptor_local_identity_is_posix_normalized(tmp_path: Path) -> None:
    local = (tmp_path / "lance").resolve()
    id_one = _store(_config(str(local))).safe_descriptor().safe_target_identity
    id_two = _store(_config(local.as_posix())).safe_descriptor().safe_target_identity
    assert id_one == id_two
