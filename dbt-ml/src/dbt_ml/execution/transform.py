from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

from ..adapters import AdapterError, WarehouseAdapter, WarehouseCapability
from ..agent_context import (
    AgentContextValidationError,
    empty_agent_context_frame,
    validate_agent_context_frame,
)
from ..config.model import ModelConfig
from ..config.profile import DEFAULT_LLM_PROVIDER
from ..dag import parse_ref
from ..paths import resolve_within_project
from ..profile import ResolvedProfile
from ..providers import get_inference_provider, resolve_provider_model
from ..sql_models import compile_sql, discover_refs, read_sql_source
from ..transforms import TransformContext, load_transform, transform_call_arity
from .contracts import ModelRunResult, RunError
from .errors import artifact_error_text
from .warehouse import warehouse_options

log = logging.getLogger(__name__)


def run_sql_model(
    *,
    model: ModelConfig,
    project_dir: Path,
    adapter: WarehouseAdapter,
    resolved: ResolvedProfile,
    full_refresh: bool = False,
) -> ModelRunResult:
    """Compile a SQL transform's refs to warehouse relations and materialize it
    with an adapter-owned CTAS/merge — upstream rows never enter the dbt-ml
    process.

    For `materialization: incremental` (#142), the SQL is compiled twice
    differently depending on whether the target already exists and
    `--full-refresh` isn't active: `is_incremental()` renders False (and
    `materialize_sql_full` runs a plain CTAS) on the first run or under
    `--full-refresh`; otherwise it renders True and the compiled delta query is
    merged into the existing target via `materialize_sql_incremental`."""
    assert model.transform is not None and model.transform.path is not None
    if WarehouseCapability.SQL_MODEL_MATERIALIZATION not in adapter.capabilities():
        raise RunError(
            f"Adapter '{adapter.adapter_type()}' does not support SQL models "
            "(SQL_MODEL_MATERIALIZATION)."
        )

    resolved_path = resolve_within_project(
        model.transform.path,
        project_dir,
        surface=f"model '{model.name}' transform.path",
    )
    sql_text = read_sql_source(resolved_path, model_name=model.name)
    refs = discover_refs(sql_text, model_name=model.name)
    # Each ref resolves to the upstream model's adapter-quoted target relation.
    relations = {name: adapter.table_ref(name) for name in refs}

    is_incremental_model = model.materialization == "incremental"
    target_exists = is_incremental_model and adapter.relation_exists(model.name)
    # Compile-time is_incremental() reflects what will actually run: False on
    # the first run or --full-refresh, even though the model is *configured*
    # incremental, so the .sql file's own branch stays truthful.
    run_incremental = is_incremental_model and target_exists and not full_refresh
    select_sql = compile_sql(
        sql_text,
        model_name=model.name,
        relations=relations,
        target_name=resolved.target_name,
        target_type=resolved.warehouse.type,
        this=adapter.table_ref(model.name) if is_incremental_model else None,
        is_incremental=run_incremental if is_incremental_model else None,
    )

    try:
        if run_incremental:
            assert model.unique_key is not None  # enforced at compile time
            if (
                WarehouseCapability.SQL_INCREMENTAL_MATERIALIZATION
                not in adapter.capabilities()
            ):
                raise RunError(
                    f"Adapter '{adapter.adapter_type()}' does not support "
                    "incremental SQL models (SQL_INCREMENTAL_MATERIALIZATION)."
                )
            result = adapter.materialize_sql_incremental(
                model.name,
                select_sql,
                unique_key=model.unique_key,
                on_schema_change=model.on_schema_change,
                options=warehouse_options(adapter, model),
            )
        else:
            result = adapter.materialize_sql_full(
                model.name, select_sql, options=warehouse_options(adapter, model)
            )
    except AdapterError as e:
        raise RunError(f"SQL model '{model.name}' failed: {e}") from e

    metrics: dict[str, Any] = {
        "compiled_sql": select_sql,
        "relation": result.relation,
        "refs": list(refs),
    }
    if is_incremental_model:
        metrics["is_incremental"] = run_incremental
    if result.job_metadata:
        metrics["job"] = result.job_metadata
    return ModelRunResult(
        model_name=model.name,
        materialization=model.materialization,
        kind="sql",
        rows_written=result.rows_written,
        rows_inserted=result.rows_inserted or 0,
        rows_updated=result.rows_updated or 0,
        metrics=metrics,
    )


def run_transform_model(
    *,
    model: ModelConfig,
    project_dir: Path,
    adapter: WarehouseAdapter,
    resolved: ResolvedProfile,
) -> ModelRunResult:
    assert model.transform is not None
    if model.materialization == "incremental":
        raise RunError(
            f"Transform model '{model.name}' declares `materialization: incremental`, "
            "but transforms only support `full` today. Set `materialization: full` "
            "(or omit it) — see issue #53."
        )
    if model.transform.type != "python":
        raise RunError(
            f"Model '{model.name}': only `type: python` transforms are supported in v1"
        )
    if not model.transform.module:
        raise RunError(f"Model '{model.name}': transform requires a `module:`")
    if not model.depends_on:
        raise RunError(
            f"Transform model '{model.name}' must declare `depends_on:` for v1"
        )

    provider_name: str | None = None
    provider_model: str | None = None
    provider_implementation: str | None = None
    if model.transform.uses_llm:
        provider_name = (
            resolved.llm.provider
            if resolved.llm is not None
            else DEFAULT_LLM_PROVIDER
        )
        selected_model = resolved.llm.model if resolved.llm is not None else None
        try:
            provider = get_inference_provider(provider_name)
            provider_model = resolve_provider_model(provider, selected_model)
            provider_implementation = provider.implementation_identity()
        except Exception as e:
            raise RunError(
                f"Transform model '{model.name}' could not initialize inference: "
                f"{artifact_error_text(e)}"
            ) from e

    transform_fn = load_transform(model.transform.module, project_dir)
    deps: dict[str, pl.DataFrame] = {}
    for dep_ref in model.depends_on:
        dep_name = parse_ref(dep_ref)
        deps[dep_name] = adapter.read_table(dep_name)

    try:
        if transform_call_arity(transform_fn) == 2:
            ctx = TransformContext(
                project_dir=project_dir,
                profile_name=resolved.profile_name,
                target_name=resolved.target_name,
                warehouse=resolved.warehouse,
                llm=resolved.llm,
                options=dict(model.transform.options),
            )
            output = transform_fn(deps, ctx)
        else:
            output = transform_fn(deps)
    except RunError:
        raise
    except Exception as e:
        log.debug("transform failed for %s", model.name, exc_info=True)
        raise RunError(
            f"Transform model '{model.name}' failed: {artifact_error_text(e)}"
        ) from e

    if not isinstance(output, pl.DataFrame):
        raise RunError(
            f"Transform '{model.transform.module}' must return a polars.DataFrame"
        )

    adapter.materialize_full(
        model.name,
        _validate_agent_context_output(output, model),
        options=warehouse_options(adapter, model),
    )

    return ModelRunResult(
        model_name=model.name,
        materialization=model.materialization,
        kind="transform",
        provider=provider_name,
        provider_model=provider_model,
        provider_implementation=provider_implementation,
        rows_written=output.height,
    )


def _validate_agent_context_output(
    frame: pl.DataFrame, model: ModelConfig
) -> pl.DataFrame:
    if model.agent_context is None:
        return frame
    if frame.is_empty() and not frame.columns:
        frame = empty_agent_context_frame(model.agent_context.grain)
    try:
        validate_agent_context_frame(frame, model.agent_context.grain)
    except AgentContextValidationError as error:
        raise RunError(f"Model '{model.name}' produced invalid {error}") from error
    return frame
