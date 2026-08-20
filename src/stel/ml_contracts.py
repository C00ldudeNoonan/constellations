from __future__ import annotations

import re
import stat
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from .config.loader import ConfigError
from .config.model import ModelConfig
from .config.project import ProjectConfig
from .dag import parse_ref
from .paths import resolve_within_project

type ExecutableMLTask = Literal["features", "classifier", "cluster", "topic_model"]
type ExecutableMLProvider = Literal[
    "builtin.count",
    "builtin.tfidf",
    "builtin.hashing",
    "builtin.naive_bayes",
    "builtin.kmeans",
    "builtin.dbscan",
    "builtin.hdbscan",
    "builtin.nmf",
    "builtin.lda",
]

# Tasks that consume a document-feature matrix (from a features model or a dense
# embedding column) rather than raw text, and that emit companion tables.
_MATRIX_TASKS: frozenset[ExecutableMLTask] = frozenset({"cluster", "topic_model"})

_MAX_NGRAM_SIZE = 64
_MAX_FEATURES = 10_000_000
_MAX_HASH_BUCKETS = 2**31
_MAX_ALPHA = 1_000_000.0
_MAX_COMPONENTS = 10_000
_MAX_ITER = 1_000_000
_MAX_REPRESENTATIVES = 1_000
_TOKEN_PATTERN = r"\w+"
_TOKEN_PATTERN_PROBES = (
    "",
    "abcdefghijklmnopqrstuvwxyz",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "0123456789",
    "_ - punctuation!",
    "é中🙂💩",
)

type _DocumentFrequency = (
    Annotated[StrictInt, Field(ge=1)]
    | Annotated[StrictFloat, Field(gt=0.0, le=1.0)]
)


class MLContractError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        model_name: str | None = None,
        path: tuple[str | int, ...] = (),
    ) -> None:
        super().__init__(message)
        self.model_name = model_name
        self.path = path


class _AnalyzerOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    analyzer: Literal["word", "char", "char_wb"] = "word"
    lowercase: StrictBool = True
    token_pattern: StrictStr = _TOKEN_PATTERN
    ngram_range: tuple[
        Annotated[StrictInt, Field(ge=1, le=_MAX_NGRAM_SIZE)],
        Annotated[StrictInt, Field(ge=1, le=_MAX_NGRAM_SIZE)],
    ] = (1, 1)
    stop_words: Literal["english"] | list[StrictStr] | None = None

    @field_validator("token_pattern")
    @classmethod
    def _validate_token_pattern(cls, value: str) -> str:
        try:
            compiled = re.compile(value)
        except re.error as e:
            raise ValueError(f"must be a valid regular expression: {e}") from e
        if compiled.groups > 1:
            raise ValueError("must contain at most one capturing group")
        if any(
            match.start() == match.end()
            for probe in _TOKEN_PATTERN_PROBES
            for match in compiled.finditer(probe)
        ):
            raise ValueError("must not match an empty string")
        return value

    @field_validator("stop_words")
    @classmethod
    def _validate_stop_words(
        cls, value: Literal["english"] | list[str] | None
    ) -> Literal["english"] | list[str] | None:
        if isinstance(value, list):
            empty = [index for index, term in enumerate(value) if not term.strip()]
            if empty:
                raise ValueError(
                    f"stop-word entries must not be blank; invalid indexes: {empty}"
                )
        return value

    @model_validator(mode="after")
    def _validate_ngram_range(self) -> Self:
        minimum, maximum = self.ngram_range
        if maximum < minimum:
            raise ValueError("ngram_range must be positive and ordered")
        return self


class _VocabularyOptions(_AnalyzerOptions):
    min_df: _DocumentFrequency = 1
    max_df: _DocumentFrequency | None = None
    max_features: Annotated[StrictInt, Field(gt=0, le=_MAX_FEATURES)] | None = None

    @model_validator(mode="after")
    def _validate_document_frequencies(self) -> Self:
        if self.max_df is None:
            return self
        if isinstance(self.min_df, int) and isinstance(self.max_df, int):
            if self.max_df < self.min_df:
                raise ValueError("max_df must be greater than or equal to min_df")
        elif isinstance(self.min_df, float) and isinstance(self.max_df, float):
            if self.max_df < self.min_df:
                raise ValueError("max_df must be greater than or equal to min_df")
        return self


class _VocabularyFeatureOptions(_VocabularyOptions):
    binary: StrictBool = False


class _HashingFeatureOptions(_AnalyzerOptions):
    n_features: Annotated[StrictInt, Field(gt=0, le=_MAX_HASH_BUCKETS)] = 2**20
    alternate_sign: StrictBool = True


class _NaiveBayesOptions(_VocabularyOptions):
    alpha: Annotated[
        StrictInt | StrictFloat,
        Field(gt=0.0, le=_MAX_ALPHA),
    ] = 1.0


class _MatrixInputOptions(BaseModel):
    """Shared options for tasks that consume a document-feature matrix.

    The matrix is assembled either from an upstream `features` model
    (long-format `term`/`value` rows, pivoted to documents x terms) or from a
    dense `embedding` column (e.g. 768-dim vectors)."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    input: Literal["features", "embedding"] = "features"
    value_field: StrictStr = "value"
    term_field: StrictStr = "term"
    embedding_field: StrictStr | None = None
    # L2 row normalization makes Euclidean distance track cosine similarity —
    # strongly recommended for TF-IDF and embedding clustering.
    normalize: Literal["none", "l2"] = "none"
    representative_docs: Annotated[StrictInt, Field(ge=0, le=_MAX_REPRESENTATIVES)] = 3
    # Top terms per group emitted to the `<model>__topics` companion (0 disables).
    # For clusters these come from c-TF-IDF; for topic models from components.
    top_terms: Annotated[StrictInt, Field(ge=0, le=1000)] = 10
    # Per-document nearest neighbors emitted to `<model>__neighbors` (0 disables).
    nearest_neighbors: Annotated[StrictInt, Field(ge=0, le=1000)] = 0
    random_state: Annotated[StrictInt, Field(ge=0)] = 0

    @model_validator(mode="after")
    def _validate_input(self) -> Self:
        if self.input == "embedding" and not self.embedding_field:
            raise ValueError("embedding_field is required when input is 'embedding'")
        if self.input == "features" and self.embedding_field is not None:
            raise ValueError("embedding_field applies only when input is 'embedding'")
        return self


class _KMeansOptions(_MatrixInputOptions):
    n_clusters: Annotated[StrictInt, Field(ge=2, le=_MAX_COMPONENTS)] = 8
    max_iter: Annotated[StrictInt, Field(ge=1, le=_MAX_ITER)] = 300
    n_init: Annotated[StrictInt, Field(ge=1, le=1000)] = 10


class _DBSCANOptions(_MatrixInputOptions):
    eps: Annotated[StrictFloat, Field(gt=0.0)] = 0.5
    min_samples: Annotated[StrictInt, Field(ge=1)] = 5
    metric: Literal["euclidean", "cosine", "manhattan"] = "euclidean"


class _HDBSCANOptions(_MatrixInputOptions):
    min_cluster_size: Annotated[StrictInt, Field(ge=2)] = 5
    min_samples: Annotated[StrictInt, Field(ge=1)] | None = None
    metric: Literal["euclidean", "manhattan"] = "euclidean"


class _TopicOptions(_MatrixInputOptions):
    n_topics: Annotated[StrictInt, Field(ge=2, le=_MAX_COMPONENTS)] = 10

    @model_validator(mode="after")
    def _forbid_embedding_input(self) -> Self:
        if self.input == "embedding":
            raise ValueError(
                "topic modeling requires input='features' (a non-negative term "
                "matrix); embedding vectors are not supported"
            )
        return self


class _NMFOptions(_TopicOptions):
    max_iter: Annotated[StrictInt, Field(ge=1, le=_MAX_ITER)] = 200


class _LDAOptions(_TopicOptions):
    max_iter: Annotated[StrictInt, Field(ge=1, le=_MAX_ITER)] = 10
    learning_method: Literal["batch", "online"] = "batch"


_CLUSTER_METRICS = frozenset(
    {"row_count", "n_clusters", "noise_points", "silhouette", "inertia"}
)
_TOPIC_METRICS = frozenset(
    {"row_count", "n_topics", "reconstruction_error", "perplexity", "topic_coherence"}
)


@dataclass(frozen=True)
class _ProviderSpec:
    task: ExecutableMLTask
    options_model: type[BaseModel]
    artifact_files: tuple[str, ...]
    metrics: frozenset[str]


_PROVIDERS: dict[ExecutableMLProvider, _ProviderSpec] = {
    "builtin.count": _ProviderSpec(
        task="features",
        options_model=_VocabularyFeatureOptions,
        artifact_files=("metadata.json", "vocabulary.json"),
        metrics=frozenset({"row_count", "vocabulary_size", "feature_rows"}),
    ),
    "builtin.tfidf": _ProviderSpec(
        task="features",
        options_model=_VocabularyFeatureOptions,
        artifact_files=("metadata.json", "vocabulary.json"),
        metrics=frozenset({"row_count", "vocabulary_size", "feature_rows"}),
    ),
    "builtin.hashing": _ProviderSpec(
        task="features",
        options_model=_HashingFeatureOptions,
        artifact_files=("metadata.json",),
        metrics=frozenset(
            {"row_count", "vocabulary_size", "feature_rows", "hash_buckets"}
        ),
    ),
    "builtin.naive_bayes": _ProviderSpec(
        task="classifier",
        options_model=_NaiveBayesOptions,
        artifact_files=("metadata.json", "model.json"),
        metrics=frozenset(
            {
                "row_count",
                "prediction_rows",
                "class_count",
                "vocabulary_size",
                "accuracy",
                "labeled_row_count",
            }
        ),
    ),
    "builtin.kmeans": _ProviderSpec(
        task="cluster",
        options_model=_KMeansOptions,
        artifact_files=("metadata.json", "model.json"),
        metrics=_CLUSTER_METRICS,
    ),
    "builtin.dbscan": _ProviderSpec(
        task="cluster",
        options_model=_DBSCANOptions,
        artifact_files=("metadata.json", "model.json"),
        metrics=_CLUSTER_METRICS - {"inertia"},
    ),
    "builtin.hdbscan": _ProviderSpec(
        task="cluster",
        options_model=_HDBSCANOptions,
        artifact_files=("metadata.json", "model.json"),
        metrics=_CLUSTER_METRICS - {"inertia"},
    ),
    "builtin.nmf": _ProviderSpec(
        task="topic_model",
        options_model=_NMFOptions,
        artifact_files=("metadata.json", "model.json"),
        metrics=_TOPIC_METRICS - {"perplexity"},
    ),
    "builtin.lda": _ProviderSpec(
        task="topic_model",
        options_model=_LDAOptions,
        artifact_files=("metadata.json", "model.json"),
        metrics=_TOPIC_METRICS - {"reconstruction_error"},
    ),
}

_DEFAULT_PROVIDERS: dict[ExecutableMLTask, ExecutableMLProvider] = {
    "features": "builtin.tfidf",
    "classifier": "builtin.naive_bayes",
    "cluster": "builtin.kmeans",
    "topic_model": "builtin.nmf",
}


@dataclass(frozen=True)
class ExecutableMLContract:
    task: ExecutableMLTask
    provider: ExecutableMLProvider
    options: dict[str, Any]
    artifact_path: Path
    artifact_files: tuple[str, ...]


type _PathToken = tuple[Literal["inode"], int, int] | tuple[Literal["name"], str]


@dataclass(frozen=True)
class _CanonicalPath:
    tokens: tuple[_PathToken, ...]


def validate_ml_project_contracts(
    models: list[ModelConfig],
    project: ProjectConfig,
    project_dir: Path,
) -> dict[str, ExecutableMLContract]:
    ml_models = {model.name: model for model in models if model.ml is not None}
    contracts = {
        name: validate_ml_contract(model, project, project_dir)
        for name, model in ml_models.items()
    }
    by_artifact: dict[_CanonicalPath, list[str]] = {}
    display_paths: dict[_CanonicalPath, Path] = {}
    for name, contract in contracts.items():
        identity = _canonical_path(contract.artifact_path)
        by_artifact.setdefault(identity, []).append(name)
        display_paths.setdefault(identity, contract.artifact_path)

    artifact_identities = sorted(
        by_artifact,
        key=lambda identity: display_paths[identity].as_posix(),
    )
    for index, identity in enumerate(artifact_identities):
        for other in artifact_identities[index + 1 :]:
            if _canonical_paths_overlap(identity, other):
                model_name = by_artifact[other][0]
                raise MLContractError(
                    "ML artifact paths must be dedicated directories; "
                    f"'{display_paths[identity]}' overlaps '{display_paths[other]}'",
                    model_name=model_name,
                    path=("ml", "artifact", "path"),
                )

    for identity, names in by_artifact.items():
        artifact_path = display_paths[identity]
        task_providers = {
            (contracts[name].task, contracts[name].provider) for name in names
        }
        if len(task_providers) != 1:
            formatted = sorted(
                f"{name}={contracts[name].task}/{contracts[name].provider}"
                for name in names
            )
            raise MLContractError(
                f"ML models sharing artifact path '{artifact_path}' must use one "
                f"task/provider contract; found {formatted}",
                model_name=names[1],
                path=("ml", "artifact", "path"),
            )

        writers: list[str] = []
        for name in names:
            ml = ml_models[name].ml
            assert ml is not None
            if ml.mode in {"fit", "fit_transform"}:
                writers.append(name)
        if len(writers) > 1:
            raise MLContractError(
                f"ML artifact path '{artifact_path}' has multiple fit writers: "
                f"{sorted(writers)}",
                model_name=writers[1],
                path=("ml", "artifact", "path"),
            )
        writer = writers[0] if writers else None
        for name in names:
            _validate_artifact_dependencies(
                ml_models[name],
                writer=writer,
                shares_artifact=len(names) > 1,
                artifact_path=artifact_path,
            )

    return contracts


def validate_persisted_ml_options(
    provider: ExecutableMLProvider,
    options: object,
) -> dict[str, Any]:
    if not isinstance(options, dict) or any(
        not isinstance(key, str) for key in options
    ):
        raise MLContractError(
            f"persisted options for provider '{provider}' must be an object"
        )
    typed_options = cast(dict[str, Any], options)
    normalized = _normalize_legacy_persisted_options(provider, typed_options)
    return _validate_provider_options(
        provider,
        normalized,
        surface=f"persisted options for provider '{provider}'",
    )


def validate_ml_contract(
    model: ModelConfig,
    project: ProjectConfig,
    project_dir: Path,
) -> ExecutableMLContract:
    ml = model.ml
    if ml is None:
        raise MLContractError(
            f"Model '{model.name}' does not declare an `ml:` block",
            model_name=model.name,
            path=("ml",),
        )

    if ml.task not in _DEFAULT_PROVIDERS:
        raise MLContractError(
            f"ML model '{model.name}' task '{ml.task}' is not executable; "
            "supported tasks are: classifier, features",
            model_name=model.name,
            path=("ml", "task"),
        )
    task = ml.task
    provider_name = ml.provider or _DEFAULT_PROVIDERS[task]
    if provider_name not in _PROVIDERS:
        supported = sorted(
            provider
            for provider, candidate in _PROVIDERS.items()
            if candidate.task == task
        )
        raise MLContractError(
            f"ML model '{model.name}' provider '{provider_name}' is not executable "
            f"for task '{task}'; supported providers: {supported}",
            model_name=model.name,
            path=("ml", "provider"),
        )
    provider = provider_name
    spec = _PROVIDERS[provider]
    if spec.task != task:
        supported = sorted(
            candidate
            for candidate, candidate_spec in _PROVIDERS.items()
            if candidate_spec.task == task
        )
        raise MLContractError(
            f"ML model '{model.name}' provider '{provider}' implements task "
            f"'{spec.task}', not '{task}'; supported providers: {supported}",
            model_name=model.name,
            path=("ml", "provider"),
        )

    if task not in _MATRIX_TASKS and (not ml.text_field or not ml.text_field.strip()):
        raise MLContractError(
            f"ML model '{model.name}' requires `ml.text_field`",
            model_name=model.name,
            path=("ml", "text_field"),
        )
    if task in {"features", *_MATRIX_TASKS} and ml.label_field is not None:
        raise MLContractError(
            f"ML model '{model.name}' task '{task}' does not use `ml.label_field`",
            model_name=model.name,
            path=("ml", "label_field"),
        )
    if task == "classifier" and ml.mode in {"fit", "fit_transform"}:
        if not ml.label_field or not ml.label_field.strip():
            raise MLContractError(
                f"Classifier model '{model.name}' requires `ml.label_field` for fitting",
                model_name=model.name,
                path=("ml", "label_field"),
            )

    if (
        provider in {"builtin.dbscan", "builtin.hdbscan", "builtin.lda"}
        and ml.mode in {"predict", "load_pretrained"}
    ):
        raise MLContractError(
            f"ML model '{model.name}' provider '{provider}' supports mode "
            "'fit_transform' or 'fit' only; it cannot assign new documents to a "
            "persisted model (use builtin.kmeans or builtin.nmf for prediction)",
            model_name=model.name,
            path=("ml", "mode"),
        )

    if ml.mode in {"predict", "load_pretrained"} and ml.options:
        raise MLContractError(
            f"ML model '{model.name}' mode '{ml.mode}' loads provider options from "
            "its persisted artifact; remove `ml.options`",
            model_name=model.name,
            path=("ml", "options"),
        )

    validated_options = _validate_provider_options(
        provider,
        ml.options,
        surface=f"ML model '{model.name}' options",
        model_name=model.name,
        path_prefix=("ml", "options"),
    )

    duplicate_metrics = sorted(
        metric for metric in set(ml.metrics) if ml.metrics.count(metric) > 1
    )
    if duplicate_metrics:
        raise MLContractError(
            f"ML model '{model.name}' declares duplicate metrics: {duplicate_metrics}",
            model_name=model.name,
            path=("ml", "metrics", ml.metrics.index(duplicate_metrics[0])),
        )
    unknown_metrics = sorted(set(ml.metrics) - spec.metrics)
    if unknown_metrics:
        raise MLContractError(
            f"ML model '{model.name}' has unsupported metrics for provider "
            f"'{provider}': {unknown_metrics}; supported metrics: {sorted(spec.metrics)}",
            model_name=model.name,
            path=("ml", "metrics", ml.metrics.index(unknown_metrics[0])),
        )

    if ml.artifact.external and ml.artifact.path is None:
        raise MLContractError(
            f"ML model '{model.name}' sets `ml.artifact.external: true` without an "
            "explicit artifact `path:`",
            model_name=model.name,
            path=("ml", "artifact", "external"),
        )

    configured_path = (
        ml.artifact.path
        if ml.artifact.path is not None
        else project.target_path / "artifacts" / model.name
    )
    try:
        artifact_path = resolve_within_project(
            configured_path,
            project_dir,
            surface=f"Model '{model.name}' ml.artifact.path",
            external=ml.artifact.external,
            hint="Set `external: true` on the artifact block to allow it.",
        )
    except ConfigError as e:
        raise MLContractError(
            str(e),
            model_name=model.name,
            path=("ml", "artifact", "path"),
        ) from e

    project_root = project_dir.resolve()
    if artifact_path == project_root or artifact_path == Path(artifact_path.anchor):
        raise MLContractError(
            f"ML model '{model.name}' artifact path must be a dedicated directory, "
            f"not filesystem/project root '{artifact_path}'",
            model_name=model.name,
            path=("ml", "artifact", "path"),
        )
    if artifact_path.exists() and not artifact_path.is_dir():
        raise MLContractError(
            f"ML model '{model.name}' artifact path is not a directory: {artifact_path}",
            model_name=model.name,
            path=("ml", "artifact", "path"),
        )
    _validate_artifact_path_boundaries(
        model.name,
        artifact_path,
        project,
        project_dir,
    )

    return ExecutableMLContract(
        task=task,
        provider=provider,
        options=validated_options,
        artifact_path=artifact_path,
        artifact_files=spec.artifact_files,
    )


def _validate_provider_options(
    provider: ExecutableMLProvider,
    options: dict[str, Any],
    *,
    surface: str,
    model_name: str | None = None,
    path_prefix: tuple[str | int, ...] = (),
) -> dict[str, Any]:
    try:
        validated = _PROVIDERS[provider].options_model.model_validate(options)
    except ValidationError as e:
        errors = e.errors(include_url=False)
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in errors
        )
        first_location = tuple(errors[0]["loc"]) if errors else ()
        raise MLContractError(
            f"{surface} are invalid: {details}",
            model_name=model_name,
            path=(*path_prefix, *first_location),
        ) from e
    return validated.model_dump(mode="python")


def _normalize_legacy_persisted_options(
    provider: ExecutableMLProvider,
    options: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(options)
    if provider not in {
        "builtin.count",
        "builtin.tfidf",
        "builtin.naive_bayes",
        "builtin.hashing",
    }:
        # Matrix-task providers (cluster/topic_model) shipped after the legacy
        # option normalization was introduced, so they have no legacy defaults.
        return normalized
    if provider in {"builtin.count", "builtin.tfidf", "builtin.naive_bayes"}:
        legacy_defaults: dict[str, Any] = {
            "n_features": 2**20,
            "alternate_sign": True,
        }
        if provider == "builtin.naive_bayes":
            legacy_defaults["binary"] = False
    else:
        legacy_defaults = {
            "min_df": 1,
            "max_df": None,
            "max_features": None,
            "binary": False,
        }
    for name, expected in legacy_defaults.items():
        if name not in normalized:
            continue
        if normalized[name] != expected:
            raise MLContractError(
                f"persisted options for provider '{provider}' contain unsupported "
                f"option '{name}'"
            )
        normalized.pop(name)
    return normalized


def _validate_artifact_path_boundaries(
    model_name: str,
    artifact_path: Path,
    project: ProjectConfig,
    project_dir: Path,
) -> None:
    layout_paths = [
        *(('source-paths', path) for path in project.source_paths),
        *(('model-paths', path) for path in project.model_paths),
        *(('transform-paths', path) for path in project.transform_paths),
    ]
    for label, configured in layout_paths:
        try:
            layout_path = resolve_within_project(
                configured,
                project_dir,
                surface=f"`{label}`",
            )
        except ConfigError as e:
            raise MLContractError(
                str(e),
                model_name=model_name,
                path=("ml", "artifact", "path"),
            ) from e
        if _paths_overlap(artifact_path, layout_path):
            raise MLContractError(
                f"ML model '{model_name}' artifact path '{artifact_path}' overlaps "
                f"configured {label} root '{layout_path}'",
                model_name=model_name,
                path=("ml", "artifact", "path"),
            )

    try:
        target_root = resolve_within_project(
            project.target_path,
            project_dir,
            surface="`target-path`",
        )
    except ConfigError as e:
        raise MLContractError(
            str(e),
            model_name=model_name,
            path=("ml", "artifact", "path"),
        ) from e
    if artifact_path == target_root or target_root.is_relative_to(artifact_path):
        raise MLContractError(
            f"ML model '{model_name}' artifact path must not own project target root "
            f"'{target_root}'",
            model_name=model_name,
            path=("ml", "artifact", "path"),
        )

    artifact_root = target_root / "artifacts"
    if artifact_path == artifact_root or artifact_root.is_relative_to(artifact_path):
        raise MLContractError(
            f"ML model '{model_name}' artifact path must be below, not own, shared "
            f"artifact root '{artifact_root}'",
            model_name=model_name,
            path=("ml", "artifact", "path"),
        )
    reserved_paths = [
        target_root / "docs",
        target_root / "manifest.json",
        target_root / "run_results.json",
        target_root / "sources.yml",
        artifact_root / "registry.json",
    ]
    for reserved in reserved_paths:
        if _paths_overlap(artifact_path, reserved):
            raise MLContractError(
                f"ML model '{model_name}' artifact path '{artifact_path}' overlaps "
                f"reserved target artifact '{reserved}'",
                model_name=model_name,
                path=("ml", "artifact", "path"),
            )


def _validate_artifact_dependencies(
    model: ModelConfig,
    *,
    writer: str | None,
    shares_artifact: bool,
    artifact_path: Path,
) -> None:
    assert model.ml is not None
    dependencies = [parse_ref(dependency) for dependency in model.depends_on or []]
    is_writer = model.ml.mode in {"fit", "fit_transform"}
    if is_writer:
        if len(dependencies) > 1:
            raise MLContractError(
                f"ML writer '{model.name}' may declare only its first data dependency; "
                f"unrelated extra dependencies: {dependencies[1:]}",
                model_name=model.name,
                path=("depends_on", 1),
            )
        return

    if writer is None:
        if len(dependencies) > 1:
            raise MLContractError(
                f"ML reader '{model.name}' may declare only its first data dependency; "
                f"artifact path '{artifact_path}' has no in-project writer",
                model_name=model.name,
                path=("depends_on", 1),
            )
        return

    if dependencies and dependencies[0] == writer:
        raise MLContractError(
            f"ML reader '{model.name}' must keep its data model as depends_on[0]; "
            f"artifact writer '{writer}' must be a second ordering dependency",
            model_name=model.name,
            path=("depends_on", 0),
        )
    ordering_dependencies = dependencies[1:]
    if ordering_dependencies != [writer]:
        if shares_artifact:
            raise MLContractError(
                f"ML reader '{model.name}' sharing artifact path '{artifact_path}' "
                f"must declare writer '{writer}' as its only dependency after "
                "depends_on[0]",
                model_name=model.name,
                path=("depends_on", 1),
            )
        raise MLContractError(
            f"ML reader '{model.name}' has unrelated extra dependencies: "
            f"{ordering_dependencies}",
            model_name=model.name,
            path=("depends_on", 1),
        )


def _paths_overlap(first: Path, second: Path) -> bool:
    return _canonical_paths_overlap(_canonical_path(first), _canonical_path(second))


def _canonical_paths_overlap(first: _CanonicalPath, second: _CanonicalPath) -> bool:
    shared = min(len(first.tokens), len(second.tokens))
    return first.tokens[:shared] == second.tokens[:shared]


def _canonical_path(path: Path) -> _CanonicalPath:
    absolute = path.resolve(strict=False)
    anchor = Path(absolute.anchor)
    try:
        anchor_stat = anchor.stat()
    except OSError:
        return _CanonicalPath((('name', _normalize_component(absolute.as_posix(), False)),))

    tokens: list[_PathToken] = [("inode", anchor_stat.st_dev, anchor_stat.st_ino)]
    current = anchor
    parts = absolute.relative_to(anchor).parts
    for index, part in enumerate(parts):
        candidate = current / part
        try:
            candidate_stat = candidate.stat()
        except OSError:
            case_insensitive = _filesystem_is_case_insensitive(current)
            tokens.extend(
                ("name", _normalize_component(remaining, case_insensitive))
                for remaining in parts[index:]
            )
            break
        tokens.append(("inode", candidate_stat.st_dev, candidate_stat.st_ino))
        current = candidate
    return _CanonicalPath(tuple(tokens))


def _filesystem_is_case_insensitive(existing_path: Path) -> bool:
    directory = existing_path if existing_path.is_dir() else existing_path.parent
    while True:
        try:
            entries = directory.iterdir()
            for entry in entries:
                try:
                    entry_stat = entry.lstat()
                except OSError:
                    continue
                if stat.S_ISLNK(entry_stat.st_mode):
                    continue
                variant = _ascii_case_variant(entry.name)
                if variant is None:
                    continue
                alternate = directory / variant
                try:
                    alternate_stat = alternate.lstat()
                except FileNotFoundError:
                    try:
                        current_stat = entry.lstat()
                    except OSError:
                        continue
                    if (
                        current_stat.st_dev != entry_stat.st_dev
                        or current_stat.st_ino != entry_stat.st_ino
                    ):
                        continue
                    return False
                except OSError:
                    continue
                return (
                    alternate_stat.st_dev == entry_stat.st_dev
                    and alternate_stat.st_ino == entry_stat.st_ino
                )
        except OSError:
            pass
        parent = directory.parent
        if parent == directory:
            return True
        if not _same_filesystem(directory, parent):
            return True
        directory = parent


def _ascii_case_variant(name: str) -> str | None:
    variant = "".join(
        character.swapcase() if character.isascii() and character.isalpha() else character
        for character in name
    )
    return variant if variant != name else None


def _same_filesystem(first: Path, second: Path) -> bool:
    try:
        return first.stat().st_dev == second.stat().st_dev
    except OSError:
        return False


def _normalize_component(component: str, case_insensitive: bool) -> str:
    if not case_insensitive:
        return component
    return unicodedata.normalize("NFC", component).casefold()
