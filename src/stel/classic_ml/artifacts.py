"""Classic-ML artifact core (issue #190, Workstream B).

Algorithm-agnostic artifact layer: the versioned schema and error envelope,
metadata/payload validation, atomic staged publication with a recovery
journal, the artifact registry, and the file primitives those depend on.

Algorithm-family readers and writers (features, classifier, matrix) live in
the package root and call into this layer; nothing here imports them, which
keeps the dependency one-way.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, cast
from uuid import uuid4

from .._distribution import distribution_version
from ..config.model import MLConfig, ModelConfig
from ..config.project import ProjectConfig
from ..hashing import HASH_DIGEST_SIZE
from ..ml_contracts import MLContractError, validate_persisted_ml_options
from .contracts import ClassifierProvider, FeatureProvider

ARTIFACT_SCHEMA_VERSION = 2

ARTIFACT_REGISTRY_FILENAME = "registry.json"

# Required key in every fitted artifact's `metadata.json` runtime block, and
# validated on load. It is written into artifacts already on disk, so renaming
# it fails every one of them with an incompatible-runtime error.
ARTIFACT_RUNTIME_VERSION_KEY = "dbt_ml"


@dataclass
class ClassicMLArtifactPublication:
    final_path: Path
    staged_path: Path
    registry_path: Path
    model_name: str
    registry_entry: dict[str, Any]
    _finished: bool = field(default=False, init=False, repr=False)

    def publish(self) -> None:
        if self._finished:
            return
        _publish_staged_artifact(self)
        self._finished = True

    def discard(self) -> None:
        if self._finished:
            return
        _remove_path(self.staged_path)
        self._finished = True


class ClassicMLArtifactError(ValueError):
    pass


class MissingClassicMLArtifactError(ClassicMLArtifactError, FileNotFoundError):
    pass


class StaleClassicMLArtifactError(ClassicMLArtifactError):
    pass


class IncompatibleClassicMLArtifactError(ClassicMLArtifactError):
    pass


def _validate_metadata(
    metadata: dict[str, Any],
    path: Path,
    provider: str,
    ml: MLConfig,
    *,
    expected_files: tuple[str, ...],
) -> None:
    schema_version = metadata.get("artifact_schema_version")
    if schema_version != ARTIFACT_SCHEMA_VERSION:
        raise IncompatibleClassicMLArtifactError(
            f"incompatible artifact schema at {path}: expected "
            f"{ARTIFACT_SCHEMA_VERSION}, found {schema_version!r}; "
            "feature semantics changed - run fit or fit_transform to rebuild"
        )
    if metadata.get("artifact_type") != "classic_ml":
        raise IncompatibleClassicMLArtifactError(
            f"incompatible artifact type at {path}: {metadata.get('artifact_type')!r}"
        )
    if metadata.get("provider") != provider:
        raise IncompatibleClassicMLArtifactError(
            f"incompatible artifact provider at {path}: expected {provider}, "
            f"found {metadata.get('provider')!r}"
        )
    if metadata.get("task") != ml.task:
        raise IncompatibleClassicMLArtifactError(
            f"incompatible artifact task at {path}: expected {ml.task}, "
            f"found {metadata.get('task')!r}"
        )
    if metadata.get("mode") not in {"fit", "fit_transform", "load_pretrained"}:
        raise IncompatibleClassicMLArtifactError(
            f"incompatible artifact mode at {path}: expected a fitted or pretrained "
            "artifact, "
            f"found {metadata.get('mode')!r}"
        )
    files = metadata.get("files")
    if files != list(expected_files):
        raise IncompatibleClassicMLArtifactError(
            f"incompatible artifact file contract at {path}: expected "
            f"{list(expected_files)!r}, found {files!r}"
        )
    runtime = metadata.get("runtime")
    required_runtime_fields = ("python", ARTIFACT_RUNTIME_VERSION_KEY, "polars", "provider")
    if not isinstance(runtime, dict) or any(
        not isinstance(runtime.get(field), str) or not runtime[field]
        for field in required_runtime_fields
    ):
        raise IncompatibleClassicMLArtifactError(
            f"incompatible artifact runtime contract at {path}: expected non-empty "
            f"fields {list(required_runtime_fields)!r}"
        )
    if runtime["provider"] != provider:
        raise IncompatibleClassicMLArtifactError(
            f"incompatible artifact runtime provider at {path}: expected {provider}, "
            f"found {runtime['provider']!r}"
        )
    if not isinstance(metadata.get("options"), dict):
        raise IncompatibleClassicMLArtifactError(
            f"incompatible artifact options at {path}: expected an object"
        )
    if not isinstance(metadata.get("artifact_files_hash"), str):
        raise IncompatibleClassicMLArtifactError(
            f"incompatible artifact file hash at {path}: expected a string"
        )
    expected_version = _artifact_version(metadata)
    if metadata.get("artifact_version") != expected_version:
        raise StaleClassicMLArtifactError(
            f"stale artifact metadata at {path}: artifact_version does not match metadata"
        )


def _validate_artifact_payload(
    metadata: dict[str, Any],
    path: Path,
    vectorizer: dict[str, Any],
) -> None:
    payload_files = [f for f in metadata.get("files", []) if f != "metadata.json"]
    try:
        actual_hash = _artifact_files_hash(path, payload_files, vectorizer)
    except ClassicMLArtifactError:
        raise
    except OSError as e:
        raise IncompatibleClassicMLArtifactError(
            f"could not validate artifact payload at {path}: {e}"
        ) from e
    expected_hash = metadata.get("artifact_files_hash")
    if actual_hash != expected_hash:
        raise StaleClassicMLArtifactError(
            f"stale artifact payload at {path}: artifact_files_hash does not match files"
        )


def _artifact_version(metadata: dict[str, Any]) -> str:
    payload = {
        key: value for key, value in metadata.items()
        if key != "artifact_version"
    }
    return _hash_json(payload)


def _artifact_files_hash(
    path: Path,
    payload_files: list[str],
    vectorizer: dict[str, Any],
) -> str:
    if not payload_files:
        return _hash_json(
            {
                "provider": vectorizer["provider"],
                "options": vectorizer["options"],
                "n_features": vectorizer["n_features"],
            }
        )
    h = hashlib.blake2b(digest_size=HASH_DIGEST_SIZE)
    for filename in sorted(payload_files):
        file_path = path / filename
        if not file_path.exists():
            raise MissingClassicMLArtifactError(
                f"missing artifact payload '{filename}' at {path}; "
                "run fit or fit_transform again"
            )
        h.update(filename.encode())
        h.update(file_path.read_bytes())
    return h.hexdigest()


def _write_artifact_payload(path: Path, vectorizer: dict[str, Any]) -> list[str]:
    if vectorizer["provider"] == "builtin.hashing":
        return []
    payload = {
        "provider": vectorizer["provider"],
        "terms": vectorizer["vocabulary"],
        "idf": vectorizer["idf"],
        "options": vectorizer["options"],
    }
    (path / "vocabulary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    return ["vocabulary.json"]


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))


def _read_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        raise MissingClassicMLArtifactError(
            f"missing artifact metadata at {metadata_path}; run fit/fit_transform or "
            "supply a stel-native artifact first"
        )
    return _read_artifact_json(metadata_path, path, "metadata")


def _read_artifact_json(
    file_path: Path,
    artifact_path: Path,
    label: str,
) -> dict[str, Any]:
    if not file_path.exists():
        raise MissingClassicMLArtifactError(
            f"missing artifact payload '{file_path.name}' at {artifact_path}; run "
            "fit/fit_transform or supply a stel-native artifact first"
        )
    if file_path.is_symlink() or not file_path.is_file():
        raise IncompatibleClassicMLArtifactError(
            f"incompatible {label} at {artifact_path}: expected a regular, "
            "non-symlink file"
        )
    try:
        payload = json.loads(file_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as e:
        raise IncompatibleClassicMLArtifactError(
            f"malformed {label} JSON at {artifact_path}: {e}"
        ) from e
    if not isinstance(payload, dict):
        raise IncompatibleClassicMLArtifactError(
            f"malformed {label} JSON at {artifact_path}: expected an object"
        )
    return cast(dict[str, Any], payload)


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as e:
        raise ClassicMLArtifactError(f"malformed {label} at {path}: {e}") from e
    if not isinstance(payload, dict):
        raise ClassicMLArtifactError(f"malformed {label} at {path}: expected an object")
    return cast(dict[str, Any], payload)


def _read_artifact_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"artifact_schema_version": ARTIFACT_SCHEMA_VERSION, "artifacts": {}}
    registry = _read_json_object(path, "artifact registry")
    registry.setdefault("artifact_schema_version", ARTIFACT_SCHEMA_VERSION)
    artifacts = registry.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ClassicMLArtifactError(
            f"malformed artifact registry at {path}: 'artifacts' must be an object"
        )
    return registry


def _new_artifact_staging_path(artifact_path: Path) -> Path:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        mkdtemp(
            prefix=f".{artifact_path.name}.staging-",
            dir=artifact_path.parent,
        )
    )


def _recover_artifact_publications(
    project: ProjectConfig,
    project_dir: Path,
) -> None:
    registry_dir = project_dir / project.target_path / "artifacts"
    if not registry_dir.exists():
        return
    registry_path = registry_dir / ARTIFACT_REGISTRY_FILENAME
    lock_path = registry_path.with_name(f".{registry_path.name}.lock")
    with _exclusive_file_lock(lock_path):
        _recover_pending_publications(registry_path)


def _artifact_publication(
    *,
    project: ProjectConfig,
    project_dir: Path,
    model: ModelConfig,
    artifact_path: Path,
    staged_path: Path,
    metadata: dict[str, Any],
) -> ClassicMLArtifactPublication:
    registry_dir = project_dir / project.target_path / "artifacts"
    registry_dir.mkdir(parents=True, exist_ok=True)
    registry_path = registry_dir / ARTIFACT_REGISTRY_FILENAME
    entry = {
        "model_name": model.name,
        "artifact_path": _display_path(artifact_path, project_dir),
        "artifact_version": metadata["artifact_version"],
        "provider": metadata["provider"],
        "task": metadata["task"],
        "code_version": metadata["code_version"],
        "config_hash": metadata["config_hash"],
        "artifact_files_hash": metadata["artifact_files_hash"],
        "training_input": metadata["training_input"],
    }
    if "metrics" in metadata:
        entry["metrics"] = metadata["metrics"]
    return ClassicMLArtifactPublication(
        final_path=artifact_path,
        staged_path=staged_path,
        registry_path=registry_path,
        model_name=model.name,
        registry_entry=entry,
    )


def _publish_staged_artifact(publication: ClassicMLArtifactPublication) -> None:
    final_path = publication.final_path
    staged_path = publication.staged_path
    registry_path = publication.registry_path
    if not staged_path.is_dir() or staged_path.is_symlink():
        raise ClassicMLArtifactError(
            f"staged artifact is missing or invalid at {staged_path}"
        )
    if final_path.is_symlink() or (final_path.exists() and not final_path.is_dir()):
        raise ClassicMLArtifactError(
            f"artifact path is not a regular directory: {final_path}"
        )

    lock_path = registry_path.with_name(f".{registry_path.name}.lock")
    with _exclusive_file_lock(lock_path):
        _recover_pending_publications(registry_path)
        registry_before = _read_artifact_registry(registry_path)
        publication_id = uuid4().hex
        backup_path = final_path.with_name(
            f".{final_path.name}.backup-{publication_id}"
        )
        journal_path = registry_path.with_name(
            f".artifact-publication-{publication_id}.json"
        )
        journal: dict[str, Any] = {
            "publication_id": publication_id,
            "model_name": publication.model_name,
            "artifact_version": publication.registry_entry["artifact_version"],
            "final_path": str(final_path),
            "staged_path": str(staged_path),
            "backup_path": str(backup_path),
            "prior_artifact_exists": final_path.exists(),
            "registry_before": registry_before,
        }
        _atomic_write_json(journal_path, journal)

        try:
            if final_path.exists():
                os.replace(final_path, backup_path)
            os.replace(staged_path, final_path)

            registry = deepcopy(registry_before)
            artifacts = registry.setdefault("artifacts", {})
            if not isinstance(artifacts, dict):
                raise ClassicMLArtifactError(
                    f"malformed artifact registry at {registry_path}: 'artifacts' must be an object"
                )
            artifacts[publication.model_name] = publication.registry_entry
            _publish_registry(registry_path, registry)
        except BaseException as error:
            try:
                _rollback_publication(journal_path, journal, registry_path)
            except BaseException as rollback_error:
                error.add_note(
                    f"Failed to roll back artifact publication: {rollback_error}"
                )
            raise
        else:
            _cleanup_committed_publication(journal_path, journal)


def _publish_registry(path: Path, registry: dict[str, Any]) -> None:
    _atomic_write_json(path, registry)


def _recover_pending_publications(registry_path: Path) -> None:
    for journal_path in sorted(
        registry_path.parent.glob(".artifact-publication-*.json")
    ):
        journal = _read_json_object(journal_path, "artifact publication journal")
        model_name = journal.get("model_name")
        artifact_version = journal.get("artifact_version")
        final_path = Path(str(journal.get("final_path", "")))
        registry = _read_artifact_registry(registry_path)
        artifacts = registry.get("artifacts")
        entry = artifacts.get(model_name) if isinstance(artifacts, dict) else None
        committed = (
            isinstance(entry, dict)
            and entry.get("artifact_version") == artifact_version
            and _artifact_version_at(final_path) == artifact_version
        )
        if committed:
            _cleanup_committed_publication(journal_path, journal)
        else:
            _rollback_publication(journal_path, journal, registry_path)


def _rollback_publication(
    journal_path: Path,
    journal: dict[str, Any],
    registry_path: Path,
) -> None:
    final_path, staged_path, backup_path = _validated_journal_paths(
        journal_path, journal
    )
    artifact_version = journal["artifact_version"]
    prior_exists = bool(journal["prior_artifact_exists"])
    registry_before = journal.get("registry_before")
    if not isinstance(registry_before, dict):
        raise ClassicMLArtifactError(
            f"malformed artifact publication journal at {journal_path}"
        )

    if backup_path.exists():
        _remove_path(final_path)
        os.replace(backup_path, final_path)
    elif not prior_exists and _artifact_version_at(final_path) == artifact_version:
        _remove_path(final_path)

    _atomic_write_json(registry_path, registry_before)
    _remove_path(staged_path)
    journal_path.unlink(missing_ok=True)


def _cleanup_committed_publication(
    journal_path: Path,
    journal: dict[str, Any],
) -> None:
    try:
        _, staged_path, backup_path = _validated_journal_paths(journal_path, journal)
        _remove_path(backup_path)
        _remove_path(staged_path)
        journal_path.unlink(missing_ok=True)
    except OSError as error:
        raise ClassicMLArtifactError(
            "Committed artifact publication cleanup remains pending at "
            f"{journal_path}; retry before publishing another artifact"
        ) from error


def _validated_journal_paths(
    journal_path: Path,
    journal: dict[str, Any],
) -> tuple[Path, Path, Path]:
    publication_id = journal.get("publication_id")
    final_raw = journal.get("final_path")
    staged_raw = journal.get("staged_path")
    backup_raw = journal.get("backup_path")
    if not all(
        isinstance(value, str) and value
        for value in (publication_id, final_raw, staged_raw, backup_raw)
    ):
        raise ClassicMLArtifactError(
            f"malformed artifact publication journal at {journal_path}"
        )
    assert isinstance(publication_id, str)
    assert isinstance(final_raw, str)
    assert isinstance(staged_raw, str)
    assert isinstance(backup_raw, str)
    final_path = Path(final_raw)
    staged_path = Path(staged_raw)
    backup_path = Path(backup_raw)
    valid = (
        journal_path.name == f".artifact-publication-{publication_id}.json"
        and final_path.is_absolute()
        and staged_path.is_absolute()
        and backup_path.is_absolute()
        and staged_path.parent == final_path.parent
        and backup_path.parent == final_path.parent
        and staged_path.name.startswith(f".{final_path.name}.staging-")
        and backup_path.name == f".{final_path.name}.backup-{publication_id}"
    )
    if not valid:
        raise ClassicMLArtifactError(
            f"malformed artifact publication journal paths at {journal_path}"
        )
    return final_path, staged_path, backup_path


def _artifact_version_at(path: Path) -> str | None:
    try:
        metadata = _read_metadata(path)
    except (ClassicMLArtifactError, OSError, UnicodeError, json.JSONDecodeError):
        return None
    version = metadata.get("artifact_version")
    return version if isinstance(version, str) else None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        module = importlib.import_module("msvcrt" if os.name == "nt" else "fcntl")
        if os.name == "nt":
            module.locking(handle.fileno(), module.LK_LOCK, 1)
        else:
            module.flock(handle.fileno(), module.LOCK_EX)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                module.locking(handle.fileno(), module.LK_UNLCK, 1)
            else:
                module.flock(handle.fileno(), module.LOCK_UN)


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def _display_path(path: Path, project_dir: Path) -> str:
    try:
        return path.relative_to(project_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _runtime_versions(provider: str) -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        ARTIFACT_RUNTIME_VERSION_KEY: distribution_version(),
        "polars": _package_version("polars"),
        "provider": provider,
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _hash_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(raw.encode(), digest_size=HASH_DIGEST_SIZE).hexdigest()


def _validated_persisted_options(
    provider: FeatureProvider | ClassifierProvider,
    options: object,
    path: Path,
    *,
    surface: str,
) -> dict[str, Any]:
    try:
        return validate_persisted_ml_options(provider, options)
    except MLContractError as e:
        raise IncompatibleClassicMLArtifactError(
            f"incompatible {surface} at {path}: {e}"
        ) from e
