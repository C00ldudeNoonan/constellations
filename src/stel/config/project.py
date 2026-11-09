from __future__ import annotations

from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationInfo,
    field_validator,
    model_validator,
)

from .class_hierarchy import ClassDeclaration, class_names, validate_classes
from .identifiers import DEFAULT_DUCKDB_FILENAME, DEFAULT_SCHEMA_NAME
from .ontology import OntologyImport
from .relation_characteristics import check_relation_characteristics
from .search_partition import SearchGroupConfig
from .vocabulary import RelationTypeDef, Vocabulary, check_declaration
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
    # Ontology files imported into the declaration below (issue #767). An
    # authoring input, like a vocabulary's `from:`: `load_project` resolves it
    # into `vocabularies:`, `classes:` and `relations:`, and nothing reads it
    # after that. Declared before them so their validators can see it.
    ontologies: tuple[OntologyImport, ...] = Field(default_factory=tuple, exclude=True)
    # Declared domain vocabularies (issue #625): a closed, ordered label set
    # a `type: enum` field can point at with `values_from: vocab.<name>`
    # instead of repeating `values:` inline. Keyed by the name used there.
    vocabularies: dict[str, Vocabulary] = Field(default_factory=dict)
    # Entity classes and the relation types allowed between them (issue
    # #626), checked against `RelationRule`/`ModelAssertionExtractorOptions`
    # at compile time. Declaring neither adds no constraint — only a project
    # that opts in gets one.
    classes: tuple[ClassDeclaration, ...] = Field(default_factory=tuple)
    relations: tuple[RelationTypeDef, ...] = Field(default_factory=tuple)
    # Segmented search groups (ADR-0037, issue #771): one logical serving name
    # over `search:` models that each publish one partition of a shared
    # upstream. Checked as a whole at compile time (`search_groups.py`), since
    # coverage is a property of the members together, not of any one of them.
    search_groups: tuple[SearchGroupConfig, ...] = Field(default_factory=tuple)

    # A project that imports an ontology may refer to an imported class from
    # a hand-written relation, term or `subclass_of:`, so every cross-reference
    # check below waits for the import: `stel.ontology_import` merges it, then
    # calls `validate_merged_declaration`. Without `ontologies:` nothing waits.

    @field_validator("classes")
    @classmethod
    def _validate_classes(
        cls, v: tuple[ClassDeclaration, ...], info: ValidationInfo
    ) -> tuple[ClassDeclaration, ...]:
        normalized = tuple(value.strip() if isinstance(value, str) else value for value in v)
        if not info.data.get("ontologies"):
            validate_classes(normalized)
        return normalized

    @model_validator(mode="after")
    def _validate_declaration(self) -> ProjectConfig:
        # Vocabulary names and term classes are checked against the project
        # around them, in the function the serving catalog also applies to a
        # compiled declaration (issue #669), so the two cannot disagree.
        if not self.ontologies:
            check_declaration(self.vocabularies, class_names(self.classes))
        return self

    @model_validator(mode="after")
    def _validate_relations(self) -> ProjectConfig:
        if not self.ontologies:
            self._check_relations()
        return self

    def validate_merged_declaration(self) -> None:
        """Run the checks construction deferred for `ontologies:`; raises ValueError."""
        validate_classes(self.classes)
        check_declaration(self.vocabularies, class_names(self.classes))
        self._check_relations()

    def _check_relations(self) -> None:
        if not self.relations:
            return
        declared_classes = set(class_names(self.classes))
        seen: set[tuple[str, str, str]] = set()
        functional_by_name: dict[str, bool] = {}
        iri_by_name: dict[str, str] = {}
        for relation in self.relations:
            if functional_by_name.setdefault(relation.name, relation.functional) != (
                relation.functional
            ):
                raise ValueError(
                    f"relation '{relation.name}' is declared `functional` on one "
                    "pairing and not on another; `functional` belongs to the "
                    "relation type, so every pairing must agree"
                )
            if relation.iri is not None and iri_by_name.setdefault(
                relation.name, relation.iri
            ) != relation.iri:
                raise ValueError(
                    f"relation '{relation.name}' declares two IRIs, "
                    f"{iri_by_name[relation.name]} and {relation.iri}; a relation "
                    "name is one relation type"
                )
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
        check_relation_characteristics(self.relations)

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
