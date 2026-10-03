"""Project-level vocabulary-alias-source checks (issue #627).

An `alias_table` resolver's `aliases: vocab.<name>` names a project-declared
vocabulary, not an upstream model — nothing in the generic transform-options
validation (`validate_options`, project-agnostic by contract) can check that
the name is actually declared under `vocabularies:`. This does, at compile
time, before source discovery or any provider call.
"""

from __future__ import annotations

from .config.model import ModelConfig
from .config.project import ProjectConfig
from .text.linking import (
    AliasTableResolverOptions,
    parse_entity_link_options,
    parse_vocabulary_alias_source,
)

LINK_TRANSFORM_MODULE = "stel.text.transforms.link_entities"


class LinkContractError(ValueError):
    def __init__(
        self, message: str, *, model_name: str, path: tuple[str | int, ...] = ()
    ) -> None:
        super().__init__(message)
        self.model_name = model_name
        self.path = path


def validate_link_project_contracts(
    models: list[ModelConfig], project: ProjectConfig
) -> None:
    for model in models:
        if (
            model.transform is None
            or model.transform.type != "python"
            or model.transform.module != LINK_TRANSFORM_MODULE
        ):
            continue
        try:
            options = parse_entity_link_options(model.transform.options)
        except Exception:
            # Malformed options are reported by the transform's own
            # validate_options hook with a better, options-shape-specific
            # message; this check only adds a constraint on top of options
            # that already parse.
            continue
        if not isinstance(options, AliasTableResolverOptions):
            continue
        vocab_name = parse_vocabulary_alias_source(options.aliases)
        if vocab_name is None:
            continue
        if vocab_name not in project.vocabularies:
            raise LinkContractError(
                f"Model '{model.name}' declares `aliases: vocab.{vocab_name}`, "
                "which is not declared under `vocabularies:`. Available: "
                f"{sorted(project.vocabularies) or '(none declared)'}",
                model_name=model.name,
                path=("transform", "options", "aliases"),
            )
