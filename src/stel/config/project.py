from __future__ import annotations

from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from .identifiers import DEFAULT_DUCKDB_FILENAME, DEFAULT_SCHEMA_NAME, validate_node_name
from .vocabulary import RelationTypeDef, Vocabulary
from .yaml_diagnostics import ConfigPath, YamlProvenance


class DuckDBConfig(BaseModel):
    """Deprecated inline warehouse config used only when a project declares no
    `profile:`. Prefer a profiles.yml `warehouse:` block; this is slated for
    removal once the legacy no-profile path goes away."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    path: Path = Path("./target") / DEFAULT_DUCKDB_FILENAME
    schema_name: str = Field(default=DEFAULT_SCHEMA_NAME, alias="schema")


class ExtractionDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_backend: str = "json"


class ProjectConfig(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True, extra="forbid", protected_namespaces=()
    )

    _yaml_provenance: YamlProvenance | None = PrivateAttr(default=None)

    name: str
    version: str = "0.1.0"
    profile: str | None = None
    duckdb: DuckDBConfig = Field(default_factory=DuckDBConfig)
    extraction: ExtractionDefaults = Field(default_factory=ExtractionDefaults)

    source_paths: list[Path] = Field(
        default_factory=lambda: [Path("sources")], alias="source-paths"
    )
    model_paths: list[Path] = Field(
        default_factory=lambda: [Path("models")], alias="model-paths"
    )
    transform_paths: list[Path] = Field(
        default_factory=lambda: [Path("transforms")], alias="transform-paths"
    )
    target_path: Path = Field(default=Path("target"), alias="target-path")
    # Declared domain vocabularies (issue #625): a closed, ordered label set
    # a `type: enum` field can point at with `values_from: vocab.<name>`
    # instead of repeating `values:` inline. Keyed by the name used there.
    vocabularies: dict[str, Vocabulary] = Field(default_factory=dict)
    # Entity classes and the relation types allowed between them (issue
    # #626), checked against `RelationRule`/`ModelAssertionExtractorOptions`
    # at compile time. Declaring neither adds no constraint — only a project
    # that opts in gets one.
    classes: tuple[str, ...] = Field(default_factory=tuple)
    relations: tuple[RelationTypeDef, ...] = Field(default_factory=tuple)

    @field_validator("vocabularies")
    @classmethod
    def _validate_vocabulary_names(cls, v: dict[str, Vocabulary]) -> dict[str, Vocabulary]:
        for name in v:
            validate_node_name(name, kind="Vocabulary")
        return v

    @field_validator("classes")
    @classmethod
    def _validate_classes(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in v)
        for name in normalized:
            validate_node_name(name, kind="Class")
        if len(normalized) != len(set(normalized)):
            raise ValueError("classes must be unique")
        return normalized

    @model_validator(mode="after")
    def _validate_term_classes(self) -> ProjectConfig:
        # A term's `class:` is a membership claim against the same declaration
        # relations are checked against (issue #629). Undeclared, it would
        # attach a class no relation or agent-facing list knows about.
        for vocab_name, vocabulary in self.vocabularies.items():
            for term in vocabulary.terms:
                if term.entity_class is None:
                    continue
                if term.entity_class not in self.classes:
                    raise ValueError(
                        f"vocabulary '{vocab_name}' term '{term.label}' declares "
                        f"`class: {term.entity_class}`, which is not declared under "
                        f"`classes:`. Declared: {sorted(self.classes) or '(none)'}"
                    )
        return self

    @model_validator(mode="after")
    def _validate_relations(self) -> ProjectConfig:
        if not self.relations:
            return self
        declared_classes = set(self.classes)
        seen: set[tuple[str, str, str]] = set()
        for relation in self.relations:
            for class_name, role in (
                (relation.subject_class, "subject_class"),
                (relation.object_class, "object_class"),
            ):
                if class_name not in declared_classes:
                    raise ValueError(
                        f"relation '{relation.name}' declares `{role}: {class_name}`, "
                        "which is not declared under `classes:`"
                    )
            key = (relation.name, relation.subject_class, relation.object_class)
            if key in seen:
                raise ValueError(
                    f"relation '{relation.name}' ({relation.subject_class} -> "
                    f"{relation.object_class}) declared twice"
                )
            seen.add(key)
        return self

    @property
    def yaml_provenance(self) -> YamlProvenance | None:
        return self._yaml_provenance

    def format_yaml_diagnostic(
        self,
        message: str,
        *,
        relative_path: ConfigPath = (),
    ) -> str:
        if self._yaml_provenance is None:
            return message
        return self._yaml_provenance.format_message(
            message,
            relative_path=relative_path,
        )
