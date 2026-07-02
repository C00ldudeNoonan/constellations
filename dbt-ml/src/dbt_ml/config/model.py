from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from .identifiers import validate_node_name


class ExtractionConfig(BaseModel):
    backend: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class TransformConfig(BaseModel):
    type: str
    module: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class MLArtifactConfig(BaseModel):
    path: Path | None = None
    include_metrics: bool = True

    @field_serializer("path")
    def _serialize_path(self, path: Path | None) -> str | None:
        return path.as_posix() if path is not None else None


class MLConfig(BaseModel):
    task: Literal[
        "features",
        "classifier",
        "regressor",
        "cluster",
        "topic_model",
        "nlp",
    ]
    mode: Literal["fit_transform", "fit", "predict", "load_pretrained"] = "fit_transform"
    provider: str | None = None
    text_field: str | None = None
    label_field: str | None = None
    artifact: MLArtifactConfig = Field(default_factory=MLArtifactConfig)
    metrics: list[str] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)


class FieldConfig(BaseModel):
    name: str
    description: str | None = None


class ModelConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: str
    description: str | None = None
    source: str | None = None
    depends_on: list[str] | None = None
    extraction: ExtractionConfig | None = None
    transform: TransformConfig | None = None
    ml: MLConfig | None = None
    fields: list[FieldConfig] = Field(default_factory=list)
    materialization: Literal["full", "incremental"] = "full"
    on_schema_change: Literal["fail", "ignore", "append_new_columns"] = "fail"
    tests: list[Any] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        return validate_node_name(v, kind="Model", reserve_internal=True)

    @model_validator(mode="after")
    def _validate_single_kind(self) -> ModelConfig:
        kinds = [
            label
            for label, block in (
                ("extraction", self.extraction),
                ("transform", self.transform),
                ("ml", self.ml),
            )
            if block is not None
        ]
        if len(kinds) > 1:
            raise ValueError(
                f"Model '{self.name}' declares multiple kind blocks "
                f"({', '.join(kinds)}); exactly one of extraction/transform/ml "
                "is allowed"
            )
        return self

    @property
    def kind_block_count(self) -> int:
        return sum(b is not None for b in (self.extraction, self.transform, self.ml))


class ModelFile(BaseModel):
    version: int = 2
    models: list[ModelConfig]

    @model_validator(mode="after")
    def _validate_models_have_kind(self) -> ModelFile:
        # Bare ModelConfig (no kind block) is allowed programmatically (DAG
        # fixtures, docs tooling); models loaded from project YAML must
        # declare what they run.
        missing = [m.name for m in self.models if m.kind_block_count == 0]
        if missing:
            raise ValueError(
                f"Models missing an extraction/transform/ml block: "
                f"{', '.join(sorted(missing))}"
            )
        return self
