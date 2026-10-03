"""Project-level relation-type and class-pairing checks (issue #626).

`RelationRule` names a subject label, an object label, and a relation type
(`text/relations.py`) with nothing to check them against: `relation_type` is
a free string, and `ModelAssertionExtractorOptions.relation_types` is an
unconstrained allow-list the author types out by hand. Two rules in one
project can assert `owns` and `owned_by` for the same fact, and the run
succeeds.

A project's `classes:`/`relations:` declaration — building on #625's
vocabulary — gives them something to check against. A project that declares
neither behaves exactly as it did before this existed: this adds a
constraint only where one is declared, never a new requirement.

Only models whose `transform:` is the built-in relation-extraction module
(`stel.text.transforms.extract_relations`) are in scope. A custom transform
module remains free to assert whatever relation types it wants, same as
before — this checks a specific, known options shape, not transform options
in general.
"""

from __future__ import annotations

from .config.model import ModelConfig
from .config.project import ProjectConfig
from .text.relations import (
    ModelAssertionExtractorOptions,
    RelationRule,
    RuleExtractorOptions,
    parse_relation_options,
)

RELATION_TRANSFORM_MODULE = "stel.text.transforms.extract_relations"

# relation_type -> the (subject_class, object_class) pairings declared for it.
_AllowedPairs = dict[str, set[tuple[str, str]]]


class RelationContractError(ValueError):
    def __init__(
        self, message: str, *, model_name: str, path: tuple[str | int, ...] = ()
    ) -> None:
        super().__init__(message)
        self.model_name = model_name
        self.path = path


def validate_relation_project_contracts(
    models: list[ModelConfig], project: ProjectConfig
) -> None:
    if not project.classes and not project.relations:
        return
    allowed_pairs: _AllowedPairs = {}
    for relation in project.relations:
        allowed_pairs.setdefault(relation.name, set()).add(
            (relation.subject_class, relation.object_class)
        )

    for model in models:
        if (
            model.transform is None
            or model.transform.type != "python"
            or model.transform.module != RELATION_TRANSFORM_MODULE
        ):
            continue
        try:
            options = parse_relation_options(model.transform.options)
        except Exception:
            # Malformed options are reported by the transform's own
            # validate_options hook with a better, options-shape-specific
            # message; this check only adds a constraint on top of options
            # that already parse.
            continue
        if isinstance(options, RuleExtractorOptions):
            for index, rule in enumerate(options.rules):
                _validate_rule(model, index, rule, allowed_pairs)
        elif isinstance(options, ModelAssertionExtractorOptions):
            _validate_model_assertion_types(model, options, allowed_pairs)


def _validate_rule(
    model: ModelConfig,
    index: int,
    rule: RelationRule,
    allowed_pairs: _AllowedPairs,
) -> None:
    pairs = allowed_pairs.get(rule.relation_type)
    if pairs is None:
        raise RelationContractError(
            f"Model '{model.name}' rule {index} asserts relation type "
            f"'{rule.relation_type}', which `relations:` does not declare. "
            f"Declared: {sorted(allowed_pairs) or '(none)'}",
            model_name=model.name,
            path=("transform", "options", "rules", index, "relation_type"),
        )
    pair = (rule.subject_label, rule.object_label)
    if pair not in pairs:
        allowed = sorted(f"{subject} -> {obj}" for subject, obj in pairs)
        raise RelationContractError(
            f"Model '{model.name}' rule {index} asserts '{rule.relation_type}' "
            f"from '{rule.subject_label}' to '{rule.object_label}', which "
            f"`relations:` does not allow. Declared pairings for "
            f"'{rule.relation_type}': {allowed}",
            model_name=model.name,
            path=("transform", "options", "rules", index),
        )


def _validate_model_assertion_types(
    model: ModelConfig,
    options: ModelAssertionExtractorOptions,
    allowed_pairs: _AllowedPairs,
) -> None:
    undeclared = sorted(set(options.relation_types) - set(allowed_pairs))
    if undeclared:
        raise RelationContractError(
            f"Model '{model.name}' model_assertion extractor may assert "
            f"{undeclared}, which `relations:` does not declare. Declared: "
            f"{sorted(allowed_pairs) or '(none)'}",
            model_name=model.name,
            path=("transform", "options", "relation_types"),
        )
