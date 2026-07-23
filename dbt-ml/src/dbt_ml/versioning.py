from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .backends import get_backend, validate_backend_options
from .backends.options import LLMBackendOptions
from .config.loader import ConfigError
from .config.model import (
    AgentContextConfig,
    ChunkConfig,
    EmbedConfig,
    ExtractionConfig,
    FieldConfig,
    LLMTransformConfig,
    MLConfig,
    ModelConfig,
    SearchConfig,
    TransformConfig,
)
from .config.profile import DEFAULT_LLM_PROVIDER
from .config.project import ProjectConfig
from .dag import parse_ref
from .embedding import EmbeddingIdentity
from .hashing import HASH_DIGEST_SIZE, canonical_json
from .llm_map import LLMMapError, resolve_llm_runtime
from .paths import resolve_within_project
from .profile import (
    ResolvedProfile,
    resolve_embedding_options,
    resolve_llm_options,
)
from .providers import (
    get_inference_provider,
    profile_options_fingerprint,
    resolve_provider_model,
)
from .sql_models import SQL_COMPILER_CONTRACT_VERSION

_HASH_CHUNK_SIZE = 1024 * 1024
_NON_SEMANTIC_EXTRACTION_OPTIONS = frozenset(
    {
        "api_key_env",
        "batch",
        "batch_poll_max_seconds",
        "batch_poll_seconds",
        "batch_size",
        "batch_timeout_seconds",
        "budget",
        "cache_path",
        "max_concurrent",
        "max_retries",
        "on_partial_batch",
        # The raw mapping may hold execution/credential fields that must not
        # invalidate state; the semantic subset re-enters identity through the
        # inference descriptor's provider_options_identity (issue #71).
        "provider_options",
        "timeout_seconds",
    }
)


def compute_content_hash(path: Path) -> str:
    return _hash_file(path)


def compute_document_id(scope: str, relative_path: str) -> str:
    return hashlib.blake2b(
        f"{scope}:{relative_path}".encode(), digest_size=HASH_DIGEST_SIZE
    ).hexdigest()


def compute_code_version(
    *,
    extraction: ExtractionConfig | None,
    transform: TransformConfig | None,
    ml: MLConfig | None = None,
    chunk: ChunkConfig | None = None,
    embed: EmbedConfig | None = None,
    llm: LLMTransformConfig | None = None,
    search: SearchConfig | None = None,
    depends_on: list[str] | None = None,
    fields: list[FieldConfig] | None = None,
    effective_extraction: Mapping[str, Any] | None = None,
    effective_transform: Mapping[str, Any] | None = None,
    effective_embedding: Mapping[str, Any] | None = None,
    effective_llm: Mapping[str, Any] | None = None,
    project_dir: Path,
    agent_context: AgentContextConfig | None = None,
    unique_key: str | None = None,
    on_schema_change: str | None = None,
) -> str:
    payload: dict[str, Any] = {
        # flush_every shapes execution (memory/flush cadence), never output
        # content — including it would invalidate every model's incremental
        # state on upgrade. ModelConfig.warehouse_options stays out for the
        # same reason (issue #91): partitioning/clustering shape physical
        # layout, not row content, and applying a layout change needs
        # --full-refresh regardless.
        "extraction": (
            dict(effective_extraction)
            if effective_extraction is not None
            else extraction.model_dump(exclude={"flush_every"})
            if extraction
            else None
        ),
        "transform": (
            dict(effective_transform)
            if effective_transform is not None
            else transform.model_dump()
            if transform
            else None
        ),
        # artifact.external is boundary policy, not code identity (see
        # flush_every above).
        "ml": ml.model_dump(mode="json", exclude={"artifact": {"external"}})
        if ml
        else None,
        "chunk": chunk.model_dump() if chunk else None,
        "embed": (
            dict(effective_embedding)
            if effective_embedding is not None
            else {
                **embed.model_dump(exclude={"batch_size", "max_retries"}),
                "identity": EmbeddingIdentity.from_config(embed).to_dict(),
            }
            if embed
            else None
        ),
        # For llm map models the resolved runtime identity (provider
        # implementation, model, prompt, schema, and sampling parameters, folded
        # into config_hash) is the code identity; execution-only tuning like
        # concurrency/retries is excluded, matching how embed folds its identity.
        "llm": (
            dict(effective_llm)
            if effective_llm is not None
            else llm.model_dump(exclude={"max_concurrent", "max_retries"})
            if llm
            else None
        ),
        "search": search.model_dump(mode="python") if search else None,
        "agent_context": (
            agent_context.model_dump(mode="json") if agent_context else None
        ),
        "depends_on": depends_on or None,
        "fields": [
            {"name": field.name, "data_type": field.data_type} for field in fields
        ]
        if fields
        else None,
    }
    if transform and transform.module:
        module_file = resolve_module_file(transform.module, project_dir)
        if module_file.exists():
            payload["transform_code_hash"] = _hash_file(module_file)
        else:
            payload["transform_code_hash"] = "missing"

    if transform and transform.type == "sql" and transform.path:
        # Raw source-SQL content + template contract version drive state
        # selection; the compiled, target-specific SQL is recorded separately in
        # the manifest so DuckDB vs BigQuery spelling never churns code_version.
        # unique_key/on_schema_change are ModelConfig-level (issue #142): scoped
        # to this block only, so other model kinds' incremental state is
        # untouched.
        payload["sql_contract"] = SQL_COMPILER_CONTRACT_VERSION
        if unique_key is not None:
            payload["sql_unique_key"] = unique_key
            payload["sql_on_schema_change"] = on_schema_change
        try:
            sql_file = resolve_within_project(
                transform.path, project_dir, surface="transform.path"
            )
        except ConfigError:
            payload["transform_sql_hash"] = "missing"
        else:
            payload["transform_sql_hash"] = (
                _hash_file(sql_file) if sql_file.exists() else "missing"
            )

    canonical = canonical_json(payload)
    return hashlib.blake2b(
        canonical.encode(), digest_size=HASH_DIGEST_SIZE
    ).hexdigest()


def compute_model_code_version(
    model: ModelConfig,
    project: ProjectConfig,
    project_dir: Path,
    *,
    resolved: ResolvedProfile | None = None,
) -> str:
    effective_extraction: dict[str, Any] | None = None
    effective_transform: dict[str, Any] | None = None
    resolved_extraction = _resolve_extraction_options(model, project, resolved)
    if resolved_extraction is not None:
        backend_name, canonical_options = resolved_extraction
        semantic_options = (
            {
                key: value
                for key, value in canonical_options.items()
                if key not in _NON_SEMANTIC_EXTRACTION_OPTIONS
            }
            if backend_name == "llm"
            else canonical_options
        )
        backend = get_backend(backend_name)
        effective_extraction = {
            "backend": backend_name,
            "backend_version": backend.version(),
            "backend_implementation": backend.implementation_identity(),
            "options": semantic_options,
        }
        if backend_name == "llm":
            effective_extraction["inference"] = _inference_descriptor(
                canonical_options
            )
    if model.transform is not None and model.transform.uses_llm:
        effective_transform = model.transform.model_dump()
        effective_transform["inference"] = _profile_inference_descriptor(resolved)
        effective_transform["llm_helper_implementation"] = get_backend(
            "llm"
        ).implementation_identity()
        effective_transform["system_prompt_fingerprint"] = (
            _profile_system_prompt_fingerprint(resolved)
        )

    effective_llm: dict[str, Any] | None = None
    if model.llm is not None:
        try:
            effective_llm = resolve_llm_runtime(
                model.llm, model.fields, resolved
            ).identity()
        except LLMMapError:
            # Provider unresolved (e.g. offline docs tooling) — fall back to the
            # raw block so the version still computes deterministically.
            effective_llm = None
    effective_embedding: dict[str, Any] | None = None
    if model.embed is not None:
        embedding_options = resolve_embedding_options(
            model.embed.provider,
            resolved,
        )
        effective_embedding = {
            **model.embed.model_dump(exclude={"batch_size", "max_retries"}),
            "identity": EmbeddingIdentity.from_config(
                model.embed,
                profile_options=embedding_options.provider_options,
            ).to_dict(),
        }

    return compute_code_version(
        extraction=model.extraction,
        transform=model.transform,
        ml=model.ml,
        chunk=model.chunk,
        embed=model.embed,
        llm=model.llm,
        search=model.search,
        agent_context=model.agent_context,
        depends_on=(
            [parse_ref(dependency) for dependency in model.depends_on]
            if (
                model.chunk is not None
                or model.embed is not None
                or model.llm is not None
                or model.search is not None
            )
            and model.depends_on
            else None
        ),
        fields=model.fields,
        effective_extraction=effective_extraction,
        effective_transform=effective_transform,
        effective_embedding=effective_embedding,
        effective_llm=effective_llm,
        project_dir=project_dir,
        unique_key=model.unique_key,
        on_schema_change=model.on_schema_change,
    )


def describe_model_inference(
    model: ModelConfig,
    project: ProjectConfig,
    *,
    resolved: ResolvedProfile | None = None,
) -> dict[str, str] | None:
    """Return the effective, artifact-safe inference implementation descriptor."""
    if model.transform is not None and model.transform.uses_llm:
        return _profile_inference_descriptor(resolved)
    resolved_extraction = _resolve_extraction_options(model, project, resolved)
    if resolved_extraction is None:
        return None
    backend_name, canonical_options = resolved_extraction
    if backend_name != "llm":
        return None
    return _inference_descriptor(canonical_options)


def describe_model_embedding(
    model: ModelConfig,
    *,
    resolved: ResolvedProfile | None = None,
) -> dict[str, str | int] | None:
    if model.embed is None:
        return None
    options = resolve_embedding_options(model.embed.provider, resolved)
    return EmbeddingIdentity.from_config(
        model.embed,
        profile_options=options.provider_options,
    ).to_dict()


def describe_model_llm(
    model: ModelConfig,
    *,
    resolved: ResolvedProfile | None = None,
) -> dict[str, str] | None:
    """Artifact-safe resolved-inference descriptor for a native `llm:` model."""
    if model.llm is None:
        return None
    try:
        return resolve_llm_runtime(model.llm, model.fields, resolved).identity()
    except LLMMapError:
        return None


def _resolve_extraction_options(
    model: ModelConfig,
    project: ProjectConfig,
    resolved: ResolvedProfile | None,
) -> tuple[str, dict[str, Any]] | None:
    if model.extraction is None:
        return None
    backend_name = model.extraction.backend or project.extraction.default_backend
    options = model.extraction.options
    if backend_name == "llm" and resolved is not None:
        options = resolve_llm_options(options, resolved)
    return backend_name, validate_backend_options(backend_name, options)


def _inference_descriptor(options: Mapping[str, Any]) -> dict[str, str]:
    effective = LLMBackendOptions.model_validate(options)
    return _provider_descriptor(
        effective.provider,
        effective.model,
        base_url=effective.base_url,
        provider_options=effective.provider_options,
    )


def _profile_inference_descriptor(
    resolved: ResolvedProfile | None,
) -> dict[str, str]:
    provider_name = (
        resolved.llm.provider
        if resolved is not None and resolved.llm is not None
        else DEFAULT_LLM_PROVIDER
    )
    model = (
        resolved.llm.model
        if resolved is not None and resolved.llm is not None
        else None
    )
    base_url = (
        resolved.llm.base_url
        if resolved is not None and resolved.llm is not None
        else None
    )
    provider_options = (
        resolved.llm.provider_options
        if resolved is not None and resolved.llm is not None
        else {}
    )
    return _provider_descriptor(
        provider_name, model, base_url=base_url, provider_options=provider_options
    )


def _profile_system_prompt_fingerprint(
    resolved: ResolvedProfile | None,
) -> str:
    system_prompt = (
        resolved.llm.system_prompt
        if resolved is not None and resolved.llm is not None
        else None
    )
    canonical = canonical_json({"system_prompt": system_prompt})
    return hashlib.blake2b(
        canonical.encode(), digest_size=HASH_DIGEST_SIZE
    ).hexdigest()


def _provider_descriptor(
    provider_name: str,
    model: str | None,
    *,
    base_url: str | None = None,
    provider_options: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    if provider_options:
        provider = get_inference_provider(
            provider_name, profile_options=provider_options
        )
    else:
        provider = get_inference_provider(provider_name)
    descriptor = {
        "provider": provider_name,
        "model": resolve_provider_model(provider, model),
        "implementation": provider.implementation_identity(),
    }
    resolved_base_url = provider.resolve_base_url(base_url)
    if resolved_base_url is not None:
        descriptor["endpoint_identity"] = hashlib.blake2b(
            canonical_json({"base_url": resolved_base_url}).encode(),
            digest_size=HASH_DIGEST_SIZE,
        ).hexdigest()
    # Semantic provider_options change provider behavior, so they are part
    # of the transformation's identity; execution and credential fields
    # never reach this fingerprint.
    options_fingerprint = profile_options_fingerprint(
        getattr(provider, "profile_options", None)
    )
    if options_fingerprint is not None:
        descriptor["provider_options_identity"] = options_fingerprint
    return descriptor


def resolve_module_file(module: str, project_dir: Path) -> Path:
    """Resolve a dotted module path (e.g. 'transforms.summarize') to a .py file
    relative to the project directory."""
    parts = module.split(".")
    relative_path = Path(*parts).with_suffix(".py")
    return resolve_within_project(
        relative_path,
        project_dir,
        surface=f"Python module '{module}'",
    )


def _hash_file(path: Path) -> str:
    h = hashlib.blake2b(digest_size=HASH_DIGEST_SIZE)
    with path.open("rb") as f:
        while chunk := f.read(_HASH_CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()
