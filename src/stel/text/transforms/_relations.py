"""Shared execution for the built-in relation-extraction child-table transform.

The driver owns mention identity, column validation, the evidence-text opt-in,
row shaping, and the stable ``relation_id``. The pairing itself is delegated to
the extractor selected by the ``extractor`` option; see ``stel.text.relations``.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import polars as pl

from ...backends.llm_backend import extract_fields_with_usage
from ...budget import BudgetGuard
from ...config.profile import DEFAULT_LLM_PROVIDER, LLMConfig
from ...execution.cost import estimate_cost
from ...providers import get_inference_provider, resolve_provider_model
from ...transforms import IncrementalContract, TransformContext
from ..relations import (
    Mention,
    ModelAssertionExtractorOptions,
    RelationAssertion,
    RelationExtractor,
    RelationInference,
    _order_key,
    get_relation_extractor,
    parse_relation_options,
    relation_id,
)

_RELATION_SCHEMA: dict[str, pl.DataType] = {
    "relation_id": pl.String(),
    "document_id": pl.String(),
    "subject_mention_id": pl.String(),
    "object_mention_id": pl.String(),
    "relation_type": pl.String(),
    # False for symmetric co-occurrence; a typed extractor sets True.
    "directed": pl.Boolean(),
    # The method category ("co_occurrence" / "rule" / "model_assertion").
    "method": pl.String(),
    "status": pl.String(),
    # Reserved for score-producing extractors; co-occurrence emits null rather
    # than implying a calibrated confidence.
    "confidence": pl.Float64(),
    "sentence_index": pl.Int64(),
    "subject_start": pl.Int64(),
    "subject_end": pl.Int64(),
    "object_start": pl.Int64(),
    "object_end": pl.Int64(),
    "subject_label": pl.String(),
    "object_label": pl.String(),
    "extractor": pl.String(),
    "extractor_version": pl.String(),
}


def validate_relation_options(options: Mapping[str, Any]) -> None:
    parse_relation_options(options)


def declared_relation_dependencies(options: Mapping[str, Any]) -> tuple[str, ...]:
    return (parse_relation_options(options).mentions,)


def declared_relation_incremental_contract(
    options: Mapping[str, Any],
) -> IncrementalContract:
    """One relation child table per document; re-analyzing a changed document
    replaces exactly its relation rows (issue #218). Child rows are keyed by
    ``relation_id``."""
    parsed = parse_relation_options(options)
    return IncrementalContract(
        parent_key="document_id",
        child_key="relation_id",
        parent_source=parsed.mentions,
        parent_source_key=parsed.document_id_field,
    )


def run_relations(
    deps: dict[str, pl.DataFrame],
    ctx: TransformContext,
) -> pl.DataFrame:
    options = parse_relation_options(ctx.options)
    extractor = get_relation_extractor(options.extractor)
    infer = _build_inference(ctx) if extractor.requires_inference() else None
    frame = _mentions_frame(deps, options)
    schema = _output_schema(options)
    grouped = _grouped_mentions(frame, options, extractor)
    if not grouped:
        return pl.DataFrame(schema=schema)

    rows: list[dict[str, Any]] = []
    for document_id in sorted(grouped):
        for relation in extractor.extract(grouped[document_id], options, infer=infer):
            rows.append(_relation_row(document_id, relation, options, extractor))
    return pl.DataFrame(rows, schema=schema, strict=False)


# --- LLM inference for the model_assertion extractor (issue #240) -------------

_RELATION_FIELDS_SPEC: list[dict[str, Any]] = [
    {"name": "subject_mention_id", "type": "string",
     "description": "id of the subject mention, copied verbatim from the input"},
    {"name": "object_mention_id", "type": "string",
     "description": "id of the object mention, copied verbatim from the input"},
    {"name": "relation_type", "type": "string",
     "description": "one of the allowed relation types, or omit the pair"},
    {"name": "confidence", "type": "number",
     "description": "confidence between 0 and 1"},
]


def _build_inference(ctx: TransformContext) -> RelationInference:
    """Resolve the governed inference provider from the profile's `llm:` block
    (never from the project YAML). Mirrors the `llm:` model kind's resolution so
    provider/model/credential/endpoint all come from operator-owned config."""
    llm: LLMConfig | None = ctx.llm
    provider_name = llm.provider if llm is not None else DEFAULT_LLM_PROVIDER
    provider_options = dict(llm.provider_options) if llm is not None else {}
    provider = (
        get_inference_provider(provider_name, profile_options=provider_options)
        if provider_options
        else get_inference_provider(provider_name)
    )
    model = resolve_provider_model(provider, llm.model if llm is not None else None)
    # Charge and enforce `llm.budget` for every provider call, like the `llm:`
    # kind — the run ledger is shared across the invocation (issue #240).
    guard = BudgetGuard(None, ctx.run_budget, cost_estimator=_cost_estimator(llm))
    return _ProviderRelationInference(
        provider=provider_name,
        model=model,
        api_key_env=llm.api_key_env if llm is not None else None,
        base_url=llm.base_url if llm is not None else None,
        provider_options=provider_options or None,
        cache_path=str(llm.cache_path) if llm is not None and llm.cache_path else None,
        timeout_seconds=llm.timeout_seconds if llm is not None else None,
        guard=guard,
    )


def _cost_estimator(
    llm: LLMConfig | None,
) -> Callable[[Mapping[str, Any]], float] | None:
    """Per-call USD estimate from the profile's pricing, for spend caps when the
    provider does not self-report cost (sync path, so no batch discount)."""
    if llm is None or llm.pricing is None:
        return None
    pricing = llm.pricing

    def _estimate(metrics: Mapping[str, Any]) -> float:
        return estimate_cost(dict(metrics), pricing)

    return _estimate


class _ProviderRelationInference:
    """`RelationInference` backed by the shared structured-LLM core
    (`extract_fields_with_usage`, `output_cardinality: many`), so caching,
    retries, and credential resolution match the `llm:`/`backend: llm` paths.
    Only the shaped assertions are returned — raw provider output never escapes."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_key_env: Any,
        base_url: str | None,
        provider_options: dict[str, Any] | None,
        cache_path: str | None,
        timeout_seconds: float | None,
        guard: BudgetGuard,
    ) -> None:
        self._provider = provider
        self._model = model
        self._api_key_env = api_key_env
        self._base_url = base_url
        self._provider_options = provider_options
        self._cache_path = cache_path
        self._timeout_seconds = timeout_seconds
        self._guard = guard

    def assert_relations(
        self,
        pairs: Sequence[tuple[Mention, Mention]],
        options: ModelAssertionExtractorOptions,
    ) -> list[RelationAssertion]:
        if not pairs:
            return []
        kwargs: dict[str, Any] = {
            "fields_spec": _RELATION_FIELDS_SPEC,
            "provider": self._provider,
            "model": self._model,
            "system": _relation_system_prompt(options),
            "output_cardinality": "many",
            "api_key_env": self._api_key_env,
            "base_url": self._base_url,
            "provider_options": self._provider_options,
        }
        if self._cache_path is not None:
            kwargs["cache_path"] = self._cache_path
        if self._timeout_seconds is not None:
            kwargs["timeout_seconds"] = self._timeout_seconds
        output, _usage = extract_fields_with_usage(
            _relation_inference_content(pairs, options),
            **kwargs,
            budget=self._guard if self._guard.active else None,
        )
        items = output.get("items")
        if not isinstance(items, list):
            # A missing/malformed `items` is a provider-response error, not "no
            # relations": returning [] would let an incremental run replace the
            # document's children with nothing and advance state, silently
            # deleting relations. Raise so the run fails and retries.
            raise ValueError(
                "relation inference provider returned a malformed response "
                "(expected a list of assertions)"
            )
        assertions: list[RelationAssertion] = []
        for item in items:
            if not isinstance(item, Mapping):
                continue
            subject_id = item.get("subject_mention_id")
            object_id = item.get("object_mention_id")
            relation_type = item.get("relation_type")
            confidence = item.get("confidence")
            if not (
                isinstance(subject_id, str)
                and isinstance(object_id, str)
                and isinstance(relation_type, str)
            ):
                continue
            if isinstance(confidence, bool) or not isinstance(confidence, int | float):
                continue
            confidence = float(confidence)
            # The schema only asks for a number; drop values outside the
            # documented unit interval rather than publish an invalid confidence
            # that would also drive threshold status assignment.
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                continue
            assertions.append(
                RelationAssertion(
                    subject_mention_id=subject_id,
                    object_mention_id=object_id,
                    relation_type=relation_type,
                    confidence=confidence,
                )
            )
        return assertions


def _relation_system_prompt(options: ModelAssertionExtractorOptions) -> str:
    allowed = ", ".join(options.relation_types)
    base = (
        "You extract typed relations between entity mentions in one document. "
        "For each candidate pair, decide whether one of these relation types "
        f"holds: {allowed}. Only use those types; if none applies, omit the pair. "
        "Copy subject_mention_id and object_mention_id verbatim from the input "
        "and set the direction so the relation reads subject -> object. Give a "
        "confidence between 0 and 1."
    )
    return f"{base}\n\n{options.prompt}" if options.prompt else base


def _relation_inference_content(
    pairs: Sequence[tuple[Mention, Mention]],
    options: ModelAssertionExtractorOptions,
) -> str:
    lines = ["Candidate mention pairs:"]
    for index, (subject, obj) in enumerate(pairs):
        lines.append(
            f"{index + 1}. "
            f"A(id={subject.mention_id}, label={subject.label}, "
            f"text={subject.text!r}) "
            f"B(id={obj.mention_id}, label={obj.label}, text={obj.text!r})"
        )
    return "\n".join(lines)


def _mentions_frame(
    deps: dict[str, pl.DataFrame], options: Any
) -> pl.DataFrame:
    expected = {options.mentions}
    if set(deps) != expected:
        raise ValueError(
            "Relation extraction expects the dependency named by the `mentions` "
            f"option ({sorted(expected)}); got: {sorted(deps)}"
        )
    return deps[options.mentions]


def _required_columns(options: Any, extractor: RelationExtractor) -> list[str]:
    required = [
        options.mention_id_field,
        options.document_id_field,
        options.sentence_index_field,
        options.start_field,
        options.end_field,
        *extractor.required_mention_columns(options),
    ]
    if options.label_field is not None:
        required.append(options.label_field)
    if options.include_mention_text or extractor.text_required():
        required.append(options.mention_text_field)
    return list(dict.fromkeys(required))


def _output_schema(options: Any) -> dict[str, pl.DataType]:
    schema = dict(_RELATION_SCHEMA)
    if options.include_mention_text:
        schema["subject_text"] = pl.String()
        schema["object_text"] = pl.String()
    return schema


def _grouped_mentions(
    frame: pl.DataFrame,
    options: Any,
    extractor: RelationExtractor,
) -> dict[str, list[Mention]]:
    required = _required_columns(options, extractor)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        text_hint = (
            " If mention text is absent, the upstream NLP transform may need "
            "`include_text: true`."
            if options.mention_text_field in missing
            and (options.include_mention_text or extractor.text_required())
            else ""
        )
        raise ValueError(
            f"Mentions model '{options.mentions}' is missing configured columns "
            f"{missing}; got: {sorted(frame.columns)}.{text_hint}"
        )

    want_text = options.include_mention_text or extractor.text_required()
    label_allow = set(options.labels)
    grouped: dict[str, list[Mention]] = {}
    seen_ids: set[str] = set()
    for row in frame.iter_rows(named=True):
        mention_id = _required_id(row, options.mention_id_field, "Mention ID")
        if mention_id in seen_ids:
            raise ValueError(
                f"Mention ID column '{options.mention_id_field}' contains duplicate "
                f"value '{mention_id}'"
            )
        seen_ids.add(mention_id)
        document_id = _required_id(row, options.document_id_field, "Document ID")
        label = (
            _optional_str(row, options.label_field, "Label")
            if options.label_field is not None
            else None
        )
        if label_allow and label not in label_allow:
            continue
        mention = Mention(
            mention_id=mention_id,
            sentence_index=_optional_int(row, options.sentence_index_field),
            start=_required_int(row, options.start_field),
            end=_required_int(row, options.end_field),
            label=label,
            text=(
                _optional_str(row, options.mention_text_field, "Mention text")
                if want_text
                else None
            ),
        )
        grouped.setdefault(document_id, []).append(mention)

    for mentions in grouped.values():
        mentions.sort(key=_order_key)
    return grouped


def _relation_row(
    document_id: str,
    relation: Any,
    options: Any,
    extractor: RelationExtractor,
) -> dict[str, Any]:
    subject = relation.subject
    obj = relation.object
    row: dict[str, Any] = {
        "relation_id": relation_id(
            document_id=document_id,
            subject_mention_id=subject.mention_id,
            object_mention_id=obj.mention_id,
            relation_type=relation.relation_type,
            method=extractor.method,
        ),
        "document_id": document_id,
        "subject_mention_id": subject.mention_id,
        "object_mention_id": obj.mention_id,
        "relation_type": relation.relation_type,
        "directed": relation.directed,
        "method": extractor.method,
        "status": relation.status,
        "confidence": relation.confidence,
        "sentence_index": relation.sentence_index,
        "subject_start": subject.start,
        "subject_end": subject.end,
        "object_start": obj.start,
        "object_end": obj.end,
        "subject_label": subject.label,
        "object_label": obj.label,
        "extractor": extractor.name,
        "extractor_version": extractor.version,
    }
    if options.include_mention_text:
        row["subject_text"] = subject.text
        row["object_text"] = obj.text
    return row


def _required_id(row: Mapping[str, Any], field_name: str, description: str) -> str:
    raw = row[field_name]
    if raw is None or not str(raw).strip():
        raise ValueError(
            f"{description} column '{field_name}' contains null or empty values"
        )
    return str(raw)


def _required_int(row: Mapping[str, Any], field_name: str) -> int:
    raw = row[field_name]
    if raw is None or isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(
            f"Column '{field_name}' must contain non-null integer offsets"
        )
    return raw


def _optional_int(row: Mapping[str, Any], field_name: str) -> int | None:
    raw = row[field_name]
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"Column '{field_name}' must contain integers or nulls")
    return raw


def _optional_str(
    row: Mapping[str, Any], field_name: str, description: str
) -> str | None:
    raw = row[field_name]
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError(f"{description} column '{field_name}' must contain strings or nulls")
    return raw
