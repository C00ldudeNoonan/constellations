"""Shared structured-map execution core for native `llm:` models (issue #144).

A native `llm:` model maps a prompt over the rows of an upstream warehouse
relation, producing typed, agent-ready rows. This module owns the
provider-neutral execution primitive — resolving the provider/model, building
the output schema from the model's ``fields``, and delegating the actual
structured completion (with caching, retries, and usage accounting) to the same
core the ``backend: llm`` extraction path uses (``extract_fields_with_usage``).
Both paths therefore share one provider/cache/retry implementation.

Credentials stay operator-owned: provider, model, base URL, cache location, and
the api-key environment reference are read from the resolved profile, never from
the project YAML ``llm:`` block.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .backends.llm_backend import extract_fields_with_usage
from .config.model import FieldConfig, LLMTransformConfig
from .config.profile import DEFAULT_LLM_PROVIDER
from .credentials import CredentialReference
from .hashing import canonical_fingerprint
from .profile import ResolvedProfile
from .providers import (
    get_inference_provider,
    profile_options_fingerprint,
    resolve_provider_model,
)

# FieldConfig data types -> JSON-schema types requested from the provider. The
# warehouse column dtype is handled separately by the runner; here we only need
# a valid structured-output schema.
_JSON_SCHEMA_TYPES: dict[str, str] = {
    "string": "string",
    "integer": "integer",
    "float": "number",
    "boolean": "boolean",
    "date": "string",
    "timestamp": "string",
    "json": "object",
    # An enum is a string the provider is constrained to choose from; the
    # closed set rides alongside as `enum` (issue #304).
    "enum": "string",
}


class LLMMapError(Exception):
    """A native `llm:` model failed to resolve or execute."""


@dataclass(frozen=True, slots=True)
class LLMMapRuntime:
    """Fully resolved, artifact-safe execution parameters for one `llm:` model.

    Everything needed to run the model deterministically, plus a ``config_hash``
    that ties incremental state and cached rows to the exact provider identity,
    prompt, schema, and sampling parameters. No secrets: ``api_key_env`` is a
    reference to an environment variable, never a resolved value.
    """

    provider: str
    model: str
    implementation: str
    output_cardinality: Literal["one", "many"]
    system_prompt: str
    temperature: float
    max_tokens: int
    max_retries: int
    max_concurrent: int
    field_names: tuple[str, ...]
    fields_spec: tuple[tuple[tuple[str, Any], ...], ...]
    api_key_env: CredentialReference | None
    base_url: str | None
    cache_path: str | None
    timeout_seconds: float | None
    provider_options: tuple[tuple[str, Any], ...] | None
    config_hash: str

    def identity(self) -> dict[str, str]:
        """Artifact-safe descriptor for manifest/docs and code_version."""
        return {
            "provider": self.provider,
            "model": self.model,
            "implementation": self.implementation,
            "output_cardinality": self.output_cardinality,
            "config_hash": self.config_hash,
        }

    def _fields_spec(self) -> list[dict[str, Any]]:
        return [dict(items) for items in self.fields_spec]

    def _provider_options(self) -> dict[str, Any] | None:
        if self.provider_options is None:
            return None
        return dict(self.provider_options)


def _build_field_spec(field: FieldConfig) -> dict[str, Any]:
    data_type = field.data_type or "string"
    schema_type = _JSON_SCHEMA_TYPES.get(data_type, "string")
    spec: dict[str, Any] = {"name": field.name, "type": schema_type}
    if field.description:
        spec["description"] = field.description
    if field.values:
        spec["enum"] = list(field.values)
    return spec


def build_fields_spec(fields: Sequence[FieldConfig]) -> list[dict[str, Any]]:
    """Translate the model's declared `fields:` into a provider output schema."""
    return [_build_field_spec(field) for field in fields]


def _resolve_provider_name(
    config: LLMTransformConfig, resolved: ResolvedProfile | None
) -> str:
    if config.provider != "default":
        return config.provider
    if resolved is not None and resolved.llm is not None:
        return resolved.llm.provider
    return DEFAULT_LLM_PROVIDER


def resolve_llm_runtime(
    config: LLMTransformConfig,
    fields: Sequence[FieldConfig],
    resolved: ResolvedProfile | None,
) -> LLMMapRuntime:
    """Resolve the concrete provider/model/credential context for an `llm:` model.

    Node-level `provider`/`model` of ``"default"`` fall back to the profile's
    LLM configuration; operator-owned base URL, cache path, timeout, provider
    options, and api-key reference come from the profile only.
    """
    provider_name = _resolve_provider_name(config, resolved)
    profile_llm = resolved.llm if resolved is not None else None
    matches_profile_provider = (
        profile_llm is not None and provider_name == profile_llm.provider
    )
    provider_options = (
        profile_llm.provider_options
        if matches_profile_provider and profile_llm is not None
        else None
    )
    try:
        provider = (
            get_inference_provider(provider_name, profile_options=provider_options)
            if provider_options
            else get_inference_provider(provider_name)
        )
    except Exception as error:  # provider not found / misconfigured
        raise LLMMapError(str(error)) from error

    model_hint: str | None = None if config.model == "default" else config.model
    if model_hint is None and matches_profile_provider and profile_llm is not None:
        model_hint = profile_llm.model
    try:
        model = resolve_provider_model(provider, model_hint)
    except Exception as error:
        raise LLMMapError(str(error)) from error
    implementation = provider.implementation_identity()

    api_key_env = profile_llm.api_key_env if profile_llm is not None else None
    base_url = (
        profile_llm.base_url
        if matches_profile_provider and profile_llm is not None
        else None
    )
    cache_path = (
        str(profile_llm.cache_path)
        if profile_llm is not None and profile_llm.cache_path is not None
        else None
    )
    timeout_seconds = (
        profile_llm.timeout_seconds
        if profile_llm is not None and "timeout_seconds" in profile_llm.model_fields_set
        else None
    )

    fields_spec = build_fields_spec(fields)
    config_hash = canonical_fingerprint(
        {
            "provider": provider_name,
            "model": model,
            "implementation": implementation,
            "prompt": config.prompt,
            "output_cardinality": config.output_cardinality,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "fields": fields_spec,
            # Endpoint and semantic provider options resolve from the profile and
            # reach the provider and cache key at execution time, so they are part
            # of the output identity: a base_url or provider_options change must
            # invalidate incremental state (mirrors the extraction cache key).
            "base_url": base_url,
            "provider_options": profile_options_fingerprint(
                getattr(provider, "profile_options", None)
            ),
        },
        domain="llm-map-config",
        version=1,
    )
    return LLMMapRuntime(
        provider=provider_name,
        model=model,
        implementation=implementation,
        output_cardinality=config.output_cardinality,
        system_prompt=config.prompt,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        max_retries=config.max_retries,
        max_concurrent=config.max_concurrent,
        field_names=tuple(field.name for field in fields),
        fields_spec=tuple(tuple(spec.items()) for spec in fields_spec),
        api_key_env=api_key_env,
        base_url=base_url,
        cache_path=cache_path,
        timeout_seconds=timeout_seconds,
        provider_options=(
            tuple(provider_options.items()) if provider_options else None
        ),
        config_hash=config_hash,
    )


def _project_object(obj: Mapping[str, Any], field_names: Sequence[str]) -> dict[str, Any]:
    # Keep only declared output fields; fill missing with None. Never retain raw
    # provider keys outside the declared schema.
    return {name: obj.get(name) for name in field_names}


def execute_map_item(
    content: str,
    runtime: LLMMapRuntime,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one input row through the provider and return its projected output rows.

    ``output_cardinality: one`` yields exactly one row; ``many`` yields zero or
    more, projected to the declared field names. The second element is the
    provider usage/accounting metrics for the call.
    """
    kwargs: dict[str, Any] = {
        "fields_spec": runtime._fields_spec(),
        "provider": runtime.provider,
        "model": runtime.model,
        "system": runtime.system_prompt,
        "temperature": runtime.temperature,
        "max_tokens": runtime.max_tokens,
        "max_retries": runtime.max_retries,
        "max_concurrent": runtime.max_concurrent,
        "output_cardinality": runtime.output_cardinality,
        "api_key_env": runtime.api_key_env,
        "base_url": runtime.base_url,
        "provider_options": runtime._provider_options(),
    }
    if runtime.cache_path is not None:
        kwargs["cache_path"] = runtime.cache_path
    if runtime.timeout_seconds is not None:
        kwargs["timeout_seconds"] = runtime.timeout_seconds

    output, usage = extract_fields_with_usage(content, **kwargs)

    if runtime.output_cardinality == "one":
        objects: list[Mapping[str, Any]] = [output]
    else:
        items = output.get("items")
        if not isinstance(items, list):
            raise LLMMapError(
                "provider returned a malformed list for output_cardinality: many"
            )
        objects = []
        for item in items:
            if not isinstance(item, Mapping):
                raise LLMMapError(
                    "provider returned a malformed list for output_cardinality: many"
                )
            objects.append(item)
    rows = [_project_object(obj, runtime.field_names) for obj in objects]
    return rows, usage
