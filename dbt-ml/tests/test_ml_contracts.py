from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from hashlib import blake2b
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import polars as pl
import pytest

import dbt_ml.classic_ml as classic_ml
from dbt_ml.adapters import WarehouseAdapter
from dbt_ml.classic_ml import (
    ARTIFACT_SCHEMA_VERSION,
    IncompatibleClassicMLArtifactError,
    MissingClassicMLArtifactError,
    _analyze,
    _artifact_version,
    _text_options,
    run_classic_ml_model,
)
from dbt_ml.config import load_project
from dbt_ml.config.model import ModelConfig
from dbt_ml.config.project import ProjectConfig
from dbt_ml.ml_contracts import (
    MLContractError,
    _same_filesystem,
    validate_ml_contract,
    validate_ml_project_contracts,
)
from dbt_ml.runner import RunError, _run_ml_model


def _model(ml: dict[str, object]) -> ModelConfig:
    return ModelConfig(
        name="derived",
        depends_on=["ref('raw')"],
        ml=ml,
    )


def _named_model(
    name: str,
    ml: dict[str, object],
    *dependencies: str,
) -> ModelConfig:
    return ModelConfig(
        name=name,
        depends_on=[f"ref('{dependency}')" for dependency in dependencies],
        ml=ml,
    )


def _features(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "task": "features",
        "provider": "builtin.tfidf",
        "text_field": "text",
    }
    config.update(overrides)
    return config


def _classifier(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "task": "classifier",
        "provider": "builtin.naive_bayes",
        "text_field": "text",
        "label_field": "label",
    }
    config.update(overrides)
    return config


def _require_case_insensitive_filesystem(tmp_path: Path) -> None:
    probe = tmp_path / "case_probe"
    probe.mkdir(exist_ok=True)
    alternate = tmp_path / "CASE_PROBE"
    if not alternate.exists() or not alternate.samefile(probe):
        pytest.skip("requires a case-insensitive filesystem")


def test_shipped_classic_ml_models_have_executable_contracts() -> None:
    project_dir = Path(__file__).resolve().parents[1] / "examples" / "classic_text_ml"
    project, _, models = load_project(project_dir)

    contracts = [
        validate_ml_contract(model, project, project_dir)
        for model in models
        if model.ml is not None
    ]

    assert {contract.provider for contract in contracts} == {
        "builtin.count",
        "builtin.tfidf",
        "builtin.hashing",
        "builtin.naive_bayes",
    }


def test_default_providers_are_resolved(tmp_path: Path) -> None:
    feature_contract = validate_ml_contract(
        _model({"task": "features", "text_field": "text"}),
        ProjectConfig(name="p"),
        tmp_path,
    )
    classifier_contract = validate_ml_contract(
        _model(
            {
                "task": "classifier",
                "text_field": "text",
                "label_field": "label",
            }
        ),
        ProjectConfig(name="p"),
        tmp_path,
    )

    assert feature_contract.provider == "builtin.tfidf"
    assert classifier_contract.provider == "builtin.naive_bayes"


@pytest.mark.parametrize("task", ["regressor", "cluster", "topic_model", "nlp"])
def test_roadmap_tasks_fail_executable_preflight(tmp_path: Path, task: str) -> None:
    with pytest.raises(MLContractError, match=rf"task '{task}' is not executable"):
        validate_ml_contract(
            _model({"task": task, "text_field": "text"}),  # type: ignore[arg-type]
            ProjectConfig(name="p"),
            tmp_path,
        )


@pytest.mark.parametrize(
    ("ml", "message"),
    [
        (_features(provider="custom.model"), "provider 'custom.model' is not executable"),
        (
            _features(provider="builtin.naive_bayes"),
            "implements task 'classifier', not 'features'",
        ),
        (
            _classifier(provider="builtin.tfidf"),
            "implements task 'features', not 'classifier'",
        ),
    ],
)
def test_provider_must_implement_selected_task(
    tmp_path: Path, ml: dict[str, object], message: str
) -> None:
    with pytest.raises(MLContractError, match=message):
        validate_ml_contract(_model(ml), ProjectConfig(name="p"), tmp_path)


@pytest.mark.parametrize(
    ("ml", "message"),
    [
        (_features(text_field=None), "requires `ml.text_field`"),
        (_features(text_field="  "), "requires `ml.text_field`"),
        (_features(label_field="label"), "does not use `ml.label_field`"),
        (_classifier(label_field=None), "requires `ml.label_field` for fitting"),
        (_classifier(label_field=" "), "requires `ml.label_field` for fitting"),
    ],
)
def test_task_fields_are_validated_before_execution(
    tmp_path: Path, ml: dict[str, object], message: str
) -> None:
    with pytest.raises(MLContractError, match=message):
        validate_ml_contract(_model(ml), ProjectConfig(name="p"), tmp_path)


@pytest.mark.parametrize(
    ("ml", "message"),
    [
        (_features(options={"lowercase": "false"}), "lowercase"),
        (_features(options={"ngram_range": [0, 1]}), "ngram_range"),
        (_features(options={"ngram_range": [2, 1]}), "positive and ordered"),
        (_features(options={"ngram_range": [1, 65]}), "less than or equal to 64"),
        (_features(options={"token_pattern": "["}), "valid regular expression"),
        (
            _features(options={"token_pattern": r"(\w+)-(\w+)"}),
            "at most one capturing group",
        ),
        (_features(options={"token_pattern": r"(?=a)"}), "empty string"),
        (_features(options={"token_pattern": r"\w*"}), "empty string"),
        (_features(options={"min_df": True}), "min_df"),
        (_features(options={"max_df": 1.1}), "max_df"),
        (_features(options={"max_features": 0}), "max_features"),
        (_features(options={"alpha": 1.0}), "Extra inputs are not permitted"),
        (
            _features(provider="builtin.hashing", options={"n_features": 0}),
            "n_features",
        ),
        (
            _features(provider="builtin.hashing", options={"min_df": 1}),
            "Extra inputs are not permitted",
        ),
        (_classifier(options={"alpha": 0}), "alpha"),
        (_classifier(options={"alpha": float("inf")}), "finite number"),
        (_classifier(options={"binary": True}), "Extra inputs are not permitted"),
    ],
)
def test_provider_options_are_strict_and_bounded(
    tmp_path: Path, ml: dict[str, object], message: str
) -> None:
    with pytest.raises(MLContractError, match=message):
        validate_ml_contract(_model(ml), ProjectConfig(name="p"), tmp_path)


def test_prediction_cannot_silently_ignore_configured_options(tmp_path: Path) -> None:
    model = _model(_features(mode="predict", options={"min_df": 2}))

    with pytest.raises(
        MLContractError, match=r"loads provider options.*persisted artifact"
    ):
        validate_ml_contract(model, ProjectConfig(name="p"), tmp_path)


@pytest.mark.parametrize(
    ("ml", "message"),
    [
        (
            _features(metrics=["accuracy"]),
            "unsupported metrics for provider 'builtin.tfidf'",
        ),
        (_features(metrics=["row_count", "row_count"]), "duplicate metrics"),
        (
            _features(artifact={"external": True}),
            "external: true.*without an explicit artifact",
        ),
        (_features(artifact={"path": "."}), "dedicated directory"),
    ],
)
def test_metrics_and_artifact_configuration_match_runtime_contract(
    tmp_path: Path, ml: dict[str, object], message: str
) -> None:
    with pytest.raises(MLContractError, match=message):
        validate_ml_contract(_model(ml), ProjectConfig(name="p"), tmp_path)


def test_artifact_path_must_be_a_directory(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"not a directory")

    with pytest.raises(MLContractError, match="artifact path is not a directory"):
        validate_ml_contract(
            _model(_features(artifact={"path": artifact.name})),
            ProjectConfig(name="p"),
            tmp_path,
        )


def test_static_prediction_contract_does_not_require_artifact_to_exist(
    tmp_path: Path,
) -> None:
    contract = validate_ml_contract(
        _model(_features(mode="predict")),
        ProjectConfig(name="p"),
        tmp_path,
    )

    assert contract.artifact_path == tmp_path / "target" / "artifacts" / "derived"


def test_missing_prediction_artifact_fails_before_warehouse_query(tmp_path: Path) -> None:
    adapter = Mock(spec=WarehouseAdapter)
    model = _model(_features(mode="predict"))

    with pytest.raises(MissingClassicMLArtifactError, match="missing artifact metadata"):
        run_classic_ml_model(
            model=model,
            project=ProjectConfig(name="p"),
            project_dir=tmp_path,
            adapter=adapter,
        )

    adapter.query_df.assert_not_called()


def test_ml_contract_error_carries_model_and_yaml_path(tmp_path: Path) -> None:
    with pytest.raises(MLContractError) as exc_info:
        validate_ml_contract(
            _model(_features(options={"token_pattern": r"(?=a)"})),
            ProjectConfig(name="p"),
            tmp_path,
        )

    assert exc_info.value.model_name == "derived"
    assert exc_info.value.path == ("ml", "options", "token_pattern")


def test_runtime_rejects_contextual_zero_width_token_match() -> None:
    options = _text_options({"token_pattern": r"(?=§)"})

    with pytest.raises(ValueError, match="empty match"):
        _analyze("contains § here", options)


def test_requested_metrics_and_include_metrics_are_honored(tmp_path: Path) -> None:
    adapter = Mock(spec=WarehouseAdapter)
    adapter.table_ref.return_value = '"p"."raw"'
    adapter.query_df.return_value = pl.DataFrame(
        {"document_id": ["1", "2"], "text": ["red blue", "blue green"]}
    )
    model = _model(
        _features(
            metrics=["vocabulary_size"],
            artifact={"include_metrics": False},
        )
    )

    output = run_classic_ml_model(
        model=model,
        project=ProjectConfig(name="p"),
        project_dir=tmp_path,
        adapter=adapter,
    )
    output.publish_artifact()

    assert output.metrics == {"vocabulary_size": 3}
    assert "metrics" not in output.artifact_metadata
    assert output.artifact_metadata["integrity"] == {"feature_count": 3}
    registry = json.loads(
        (tmp_path / "target" / "artifacts" / "registry.json").read_text()
    )
    assert "metrics" not in registry["artifacts"]["derived"]


def test_requested_metrics_are_projected_into_artifact(tmp_path: Path) -> None:
    adapter = Mock(spec=WarehouseAdapter)
    adapter.table_ref.return_value = '"p"."raw"'
    adapter.query_df.return_value = pl.DataFrame({"text": ["red blue"]})
    model = _model(_features(metrics=["feature_rows"]))

    output = run_classic_ml_model(
        model=model,
        project=ProjectConfig(name="p"),
        project_dir=tmp_path,
        adapter=adapter,
    )

    assert output.metrics == {"feature_rows": output.df.height}
    assert output.artifact_metadata["metrics"] == output.metrics
    output.discard_staged_artifact()


def _publication_adapter() -> Mock:
    adapter = Mock(spec=WarehouseAdapter)
    adapter.table_ref.return_value = '"p"."raw"'
    adapter.query_df.return_value = pl.DataFrame(
        {"document_id": ["1", "2"], "text": ["red blue", "blue green"]}
    )
    adapter.parse_warehouse_options.return_value = None
    adapter.materialize_full.return_value = 2
    return adapter


def _write_prior_publication(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    artifact_path = tmp_path / "target" / "artifacts" / "derived"
    artifact_path.mkdir(parents=True)
    (artifact_path / "sentinel.txt").write_text("prior artifact")
    registry_path = artifact_path.parent / "registry.json"
    registry: dict[str, object] = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifacts": {
            "derived": {
                "model_name": "derived",
                "artifact_path": "target/artifacts/derived",
                "artifact_version": "prior-version",
            }
        },
    }
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True))
    return artifact_path, registry_path, registry


def _assert_no_publication_debris(artifact_path: Path) -> None:
    names = {path.name for path in artifact_path.parent.iterdir()}
    assert not any(".staging-" in name for name in names)
    assert not any(".backup-" in name for name in names)
    assert not any(name.startswith(".artifact-publication-") for name in names)


def test_warehouse_failure_preserves_prior_artifact_and_registry(
    tmp_path: Path,
) -> None:
    artifact_path, registry_path, registry_before = _write_prior_publication(tmp_path)
    adapter = _publication_adapter()
    adapter.materialize_full.side_effect = RuntimeError("warehouse failed")

    with pytest.raises(RunError, match="warehouse failed"):
        _run_ml_model(
            model=_model(_features()),
            project=ProjectConfig(name="p"),
            project_dir=tmp_path,
            adapter=adapter,
        )

    assert (artifact_path / "sentinel.txt").read_text() == "prior artifact"
    assert json.loads(registry_path.read_text()) == registry_before
    _assert_no_publication_debris(artifact_path)


def test_registry_failure_rolls_back_replacement_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_path, registry_path, registry_before = _write_prior_publication(tmp_path)
    adapter = _publication_adapter()

    def fail_registry(_path: Path, _registry: dict[str, object]) -> None:
        raise OSError("registry publish failed")

    monkeypatch.setattr(classic_ml, "_publish_registry", fail_registry)

    with pytest.raises(RunError, match="registry publish failed"):
        _run_ml_model(
            model=_model(_features()),
            project=ProjectConfig(name="p"),
            project_dir=tmp_path,
            adapter=adapter,
        )

    adapter.materialize_full.assert_called_once()
    assert (artifact_path / "sentinel.txt").read_text() == "prior artifact"
    assert json.loads(registry_path.read_text()) == registry_before
    _assert_no_publication_debris(artifact_path)


def test_registry_failure_does_not_advertise_first_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _publication_adapter()

    def fail_registry(_path: Path, _registry: dict[str, object]) -> None:
        raise OSError("registry publish failed")

    monkeypatch.setattr(classic_ml, "_publish_registry", fail_registry)

    with pytest.raises(RunError, match="registry publish failed"):
        _run_ml_model(
            model=_model(_features()),
            project=ProjectConfig(name="p"),
            project_dir=tmp_path,
            adapter=adapter,
        )

    artifact_path = tmp_path / "target" / "artifacts" / "derived"
    registry_path = artifact_path.parent / "registry.json"
    assert not artifact_path.exists()
    assert json.loads(registry_path.read_text())["artifacts"] == {}
    _assert_no_publication_debris(artifact_path)


def test_committed_journal_blocks_later_publication_until_cleanup_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_path, registry_path, _ = _write_prior_publication(tmp_path)
    first_adapter = _publication_adapter()
    original_remove_path = classic_ml._remove_path

    def fail_backup_cleanup(path: Path) -> None:
        if ".backup-" in path.name:
            raise OSError("backup is busy")
        original_remove_path(path)

    monkeypatch.setattr(classic_ml, "_remove_path", fail_backup_cleanup)

    with pytest.raises(RunError, match="cleanup remains pending"):
        _run_ml_model(
            model=_model(_features()),
            project=ProjectConfig(name="p"),
            project_dir=tmp_path,
            adapter=first_adapter,
        )

    committed_registry = json.loads(registry_path.read_text())
    committed_version = committed_registry["artifacts"]["derived"][
        "artifact_version"
    ]
    assert json.loads((artifact_path / "metadata.json").read_text())[
        "artifact_version"
    ] == committed_version
    assert list(registry_path.parent.glob(".artifact-publication-*.json"))

    second_adapter = _publication_adapter()
    with pytest.raises(
        classic_ml.ClassicMLArtifactError, match="cleanup remains pending"
    ):
        run_classic_ml_model(
            model=_model(_features()),
            project=ProjectConfig(name="p"),
            project_dir=tmp_path,
            adapter=second_adapter,
        )

    second_adapter.query_df.assert_not_called()
    assert json.loads(registry_path.read_text()) == committed_registry
    assert json.loads((artifact_path / "metadata.json").read_text())[
        "artifact_version"
    ] == committed_version


def test_parallel_artifact_publications_preserve_all_registry_entries(
    tmp_path: Path,
) -> None:
    outputs = []
    for model_name in ("first", "second"):
        model = ModelConfig(
            name=model_name,
            depends_on=["ref('raw')"],
            ml=_features(artifact={"path": f"target/artifacts/{model_name}"}),
        )
        outputs.append(
            run_classic_ml_model(
                model=model,
                project=ProjectConfig(name="p"),
                project_dir=tmp_path,
                adapter=_publication_adapter(),
            )
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda output: output.publish_artifact(), outputs))

    registry_path = tmp_path / "target" / "artifacts" / "registry.json"
    registry = json.loads(registry_path.read_text())
    assert set(registry["artifacts"]) == {"first", "second"}
    for output in outputs:
        entry = registry["artifacts"][output.artifact_path.name]
        assert entry["artifact_version"] == output.artifact_version


def test_pending_replacement_is_rolled_back_during_recovery(tmp_path: Path) -> None:
    artifact_path, registry_path, registry_before = _write_prior_publication(tmp_path)
    output = run_classic_ml_model(
        model=_model(_features()),
        project=ProjectConfig(name="p"),
        project_dir=tmp_path,
        adapter=_publication_adapter(),
    )
    publication = output._publication
    assert publication is not None

    backup_path = artifact_path.with_name(".derived.backup-interrupted")
    journal_path = registry_path.with_name(".artifact-publication-interrupted.json")
    journal = {
        "publication_id": "interrupted",
        "model_name": "derived",
        "artifact_version": output.artifact_version,
        "final_path": str(artifact_path),
        "staged_path": str(publication.staged_path),
        "backup_path": str(backup_path),
        "prior_artifact_exists": True,
        "registry_before": registry_before,
    }
    classic_ml._atomic_write_json(journal_path, journal)
    os.replace(artifact_path, backup_path)
    os.replace(publication.staged_path, artifact_path)

    classic_ml._recover_artifact_publications(ProjectConfig(name="p"), tmp_path)

    assert (artifact_path / "sentinel.txt").read_text() == "prior artifact"
    assert json.loads(registry_path.read_text()) == registry_before
    _assert_no_publication_debris(artifact_path)


def test_malformed_recovery_journal_cannot_remove_unrelated_paths(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "target" / "artifacts" / "registry.json"
    registry_path.parent.mkdir(parents=True)
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "sentinel.txt").write_text("keep")
    journal_path = registry_path.with_name(".artifact-publication-unsafe.json")
    classic_ml._atomic_write_json(
        journal_path,
        {
            "publication_id": "unsafe",
            "model_name": "derived",
            "artifact_version": "new",
            "final_path": str(victim),
            "staged_path": str(tmp_path / "not-a-generated-staging-path"),
            "backup_path": str(tmp_path / "not-a-generated-backup-path"),
            "prior_artifact_exists": True,
            "registry_before": {
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "artifacts": {},
            },
        },
    )

    with pytest.raises(classic_ml.ClassicMLArtifactError, match="journal paths"):
        classic_ml._recover_pending_publications(registry_path)

    assert (victim / "sentinel.txt").read_text() == "keep"


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("sources/artifact", "source-paths root"),
        ("models/artifact", "model-paths root"),
        ("transforms/artifact", "transform-paths root"),
        ("target", "target root"),
        ("target/artifacts", "shared artifact root"),
        ("target/docs/model", "reserved target artifact"),
        ("target/manifest.json/model", "reserved target artifact"),
        ("target/run_results.json", "reserved target artifact"),
        ("target/sources.yml/model", "reserved target artifact"),
    ],
)
def test_artifact_path_isolated_from_project_and_reserved_roots(
    tmp_path: Path, path: str, message: str
) -> None:
    with pytest.raises(MLContractError, match=message):
        validate_ml_contract(
            _model(_features(artifact={"path": path})),
            ProjectConfig(name="p"),
            tmp_path,
        )


def test_external_artifact_cannot_own_project_layout(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    with pytest.raises(MLContractError, match="source-paths root"):
        validate_ml_contract(
            _model(
                _features(
                    artifact={"path": str(tmp_path), "external": True},
                )
            ),
            ProjectConfig(name="p"),
            project_dir,
        )


def test_case_variant_artifact_cannot_overlap_model_root(tmp_path: Path) -> None:
    _require_case_insensitive_filesystem(tmp_path)
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "models").mkdir()

    with pytest.raises(MLContractError, match="model-paths root"):
        validate_ml_contract(
            _model(_features(artifact={"path": "Models/artifact"})),
            ProjectConfig(name="p"),
            project_dir,
        )


def test_case_variant_artifact_paths_share_one_writer_identity(tmp_path: Path) -> None:
    _require_case_insensitive_filesystem(tmp_path)
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    models = [
        _named_model(
            "fit_one",
            _features(mode="fit", artifact={"path": "target/artifacts/Shared"}),
            "raw",
        ),
        _named_model(
            "fit_two",
            _features(mode="fit", artifact={"path": "target/artifacts/shared"}),
            "raw",
        ),
    ]

    with pytest.raises(MLContractError, match="multiple fit writers"):
        validate_ml_project_contracts(models, ProjectConfig(name="p"), project_dir)


def test_case_variant_reserved_target_path_is_rejected(tmp_path: Path) -> None:
    _require_case_insensitive_filesystem(tmp_path)
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "target").mkdir()

    with pytest.raises(MLContractError, match="reserved target artifact"):
        validate_ml_contract(
            _model(_features(artifact={"path": "target/Manifest.JSON/model"})),
            ProjectConfig(name="p"),
            project_dir,
        )


def test_broken_symlink_does_not_disable_case_insensitive_detection(
    tmp_path: Path,
) -> None:
    _require_case_insensitive_filesystem(tmp_path)
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    target = project_dir / "target"
    target.mkdir()
    (target / "a_broken_probe").symlink_to(target / "missing")

    with pytest.raises(MLContractError, match="reserved target artifact"):
        validate_ml_contract(
            _model(_features(artifact={"path": "target/Manifest.JSON/model"})),
            ProjectConfig(name="p"),
            project_dir,
        )


def test_case_probe_does_not_inherit_behavior_across_mount_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mounted = tmp_path / "mounted"
    mounted.mkdir()
    original_stat = Path.stat

    def fake_stat(path: Path, *args: object, **kwargs: object) -> object:
        if path == mounted:
            return SimpleNamespace(st_dev=101)
        if path == tmp_path:
            return SimpleNamespace(st_dev=202)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fake_stat)

    assert not _same_filesystem(mounted, tmp_path)


def test_shared_artifact_reader_orders_after_writer_without_changing_data_ref(
    tmp_path: Path,
) -> None:
    artifact = {"path": "target/artifacts/shared"}
    writer = _named_model(
        "fit_features",
        _features(mode="fit", artifact=artifact),
        "raw",
    )
    reader = _named_model(
        "predict_features",
        _features(mode="predict", artifact=artifact),
        "raw",
        "fit_features",
    )

    contracts = validate_ml_project_contracts(
        [writer, reader], ProjectConfig(name="p"), tmp_path
    )

    assert contracts["fit_features"].artifact_path == contracts["predict_features"].artifact_path


def test_shared_artifact_has_only_one_writer(tmp_path: Path) -> None:
    artifact = {"path": "target/artifacts/shared"}
    writers = [
        _named_model("fit_one", _features(mode="fit", artifact=artifact), "raw"),
        _named_model("fit_two", _features(mode="fit", artifact=artifact), "raw"),
    ]

    with pytest.raises(MLContractError, match="multiple fit writers") as exc_info:
        validate_ml_project_contracts(writers, ProjectConfig(name="p"), tmp_path)

    assert exc_info.value.model_name == "fit_two"
    assert exc_info.value.path == ("ml", "artifact", "path")


def test_shared_artifact_requires_one_task_provider_contract(tmp_path: Path) -> None:
    artifact = {"path": "target/artifacts/shared"}
    models = [
        _named_model("count", _features(provider="builtin.count", artifact=artifact), "raw"),
        _named_model(
            "tfidf",
            _features(mode="predict", provider="builtin.tfidf", artifact=artifact),
            "raw",
            "count",
        ),
    ]

    with pytest.raises(MLContractError, match="one task/provider contract"):
        validate_ml_project_contracts(models, ProjectConfig(name="p"), tmp_path)


@pytest.mark.parametrize(
    ("writer_dependencies", "reader_dependencies", "message", "path"),
    [
        (("raw", "unrelated"), ("raw", "fit_features"), "writer.*only", ("depends_on", 1)),
        (("raw",), ("raw",), "must declare writer", ("depends_on", 1)),
        (
            ("raw",),
            ("fit_features", "raw"),
            r"keep its data model as depends_on\[0\]",
            ("depends_on", 0),
        ),
        (
            ("raw",),
            ("raw", "fit_features", "unrelated"),
            "only dependency after",
            ("depends_on", 1),
        ),
    ],
)
def test_artifact_dependency_contract_rejects_inert_dependencies(
    tmp_path: Path,
    writer_dependencies: tuple[str, ...],
    reader_dependencies: tuple[str, ...],
    message: str,
    path: tuple[str | int, ...],
) -> None:
    artifact = {"path": "target/artifacts/shared"}
    writer = _named_model(
        "fit_features",
        _features(mode="fit", artifact=artifact),
        *writer_dependencies,
    )
    reader = _named_model(
        "predict_features",
        _features(mode="predict", artifact=artifact),
        *reader_dependencies,
    )

    with pytest.raises(MLContractError, match=message) as exc_info:
        validate_ml_project_contracts(
            [writer, reader], ProjectConfig(name="p"), tmp_path
        )

    assert exc_info.value.path == path


def test_reader_without_project_writer_rejects_extra_dependencies(tmp_path: Path) -> None:
    reader = _named_model(
        "predict_features",
        _features(mode="predict", artifact={"path": "target/artifacts/native"}),
        "raw",
        "unrelated",
    )

    with pytest.raises(MLContractError, match="no in-project writer"):
        validate_ml_project_contracts([reader], ProjectConfig(name="p"), tmp_path)


def test_distinct_artifact_paths_must_not_be_nested(tmp_path: Path) -> None:
    models = [
        _named_model(
            "outer",
            _features(artifact={"path": "target/artifacts/shared"}),
            "raw",
        ),
        _named_model(
            "inner",
            _features(artifact={"path": "target/artifacts/shared/nested"}),
            "raw",
        ),
    ]

    with pytest.raises(MLContractError, match="dedicated directories"):
        validate_ml_project_contracts(models, ProjectConfig(name="p"), tmp_path)


def _fit_artifact(tmp_path: Path) -> tuple[ModelConfig, Mock]:
    adapter = Mock(spec=WarehouseAdapter)
    adapter.table_ref.return_value = '"p"."raw"'
    adapter.query_df.return_value = pl.DataFrame({"text": ["red blue"]})
    model = _model(_features())
    output = run_classic_ml_model(
        model=model,
        project=ProjectConfig(name="p"),
        project_dir=tmp_path,
        adapter=adapter,
    )
    output.publish_artifact()
    return model, adapter


def test_malformed_artifact_metadata_is_domain_error_before_query(tmp_path: Path) -> None:
    _, adapter = _fit_artifact(tmp_path)
    artifact = tmp_path / "target" / "artifacts" / "derived"
    (artifact / "metadata.json").write_text("{")
    adapter.reset_mock()

    with pytest.raises(IncompatibleClassicMLArtifactError, match="malformed metadata JSON"):
        run_classic_ml_model(
            model=_model(_features(mode="predict")),
            project=ProjectConfig(name="p"),
            project_dir=tmp_path,
            adapter=adapter,
        )

    adapter.query_df.assert_not_called()


def test_malformed_artifact_payload_is_domain_error_before_query(tmp_path: Path) -> None:
    _, adapter = _fit_artifact(tmp_path)
    artifact = tmp_path / "target" / "artifacts" / "derived"
    payload_path = artifact / "vocabulary.json"
    payload_path.write_text("{")
    metadata_path = artifact / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    digest = blake2b(digest_size=8)
    digest.update(b"vocabulary.json")
    digest.update(payload_path.read_bytes())
    metadata["artifact_files_hash"] = digest.hexdigest()
    metadata["artifact_version"] = _artifact_version(metadata)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    adapter.reset_mock()

    with pytest.raises(IncompatibleClassicMLArtifactError, match="malformed vocabulary JSON"):
        run_classic_ml_model(
            model=_model(_features(mode="predict")),
            project=ProjectConfig(name="p"),
            project_dir=tmp_path,
            adapter=adapter,
        )

    adapter.query_df.assert_not_called()


def test_persisted_options_use_typed_provider_contract_before_query(tmp_path: Path) -> None:
    _, adapter = _fit_artifact(tmp_path)
    metadata_path = tmp_path / "target" / "artifacts" / "derived" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["options"]["lowercase"] = "false"
    metadata["artifact_version"] = _artifact_version(metadata)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    adapter.reset_mock()

    with pytest.raises(IncompatibleClassicMLArtifactError, match="lowercase"):
        run_classic_ml_model(
            model=_model(_features(mode="load_pretrained")),
            project=ProjectConfig(name="p"),
            project_dir=tmp_path,
            adapter=adapter,
        )

    adapter.query_df.assert_not_called()


def test_invalid_artifact_runtime_contract_fails_before_warehouse_query(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "target" / "artifacts" / "derived"
    artifact.mkdir(parents=True)
    (artifact / "metadata.json").write_text(
        json.dumps(
            {
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "artifact_type": "classic_ml",
                "task": "features",
                "provider": "builtin.tfidf",
                "mode": "fit",
                "files": ["metadata.json", "vocabulary.json"],
                "runtime": {"provider": "builtin.tfidf"},
                "options": {},
                "artifact_files_hash": "invalid",
                "artifact_version": "invalid",
            }
        )
    )
    adapter = Mock(spec=WarehouseAdapter)

    with pytest.raises(
        IncompatibleClassicMLArtifactError, match="runtime contract"
    ):
        run_classic_ml_model(
            model=_model(_features(mode="predict")),
            project=ProjectConfig(name="p"),
            project_dir=tmp_path,
            adapter=adapter,
        )

    adapter.query_df.assert_not_called()
