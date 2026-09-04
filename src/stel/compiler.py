from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from pathlib import Path

from .adapters import (
    AdapterError,
    WarehouseAdapter,
    WarehouseCapability,
    adapter_capabilities,
)
from .backends import BackendOptionsError, list_backends, validate_backend_options
from .config.loader import ConfigError
from .config.model import (
    DEFAULT_VECTOR_INDEX,
    ModelConfig,
    ModelKind,
    protect_model_llm_credential_option,
)
from .config.project import ProjectConfig
from .config.source import SourceConfig
from .config.yaml_diagnostics import ConfigPath
from .dag import DAGError, ProjectDAG, is_dbt_ref, parse_dbt_ref, parse_ref
from .embedding import resolve_search_embedding_identity
from .ml_contracts import MLContractError, validate_ml_project_contracts
from .paths import resolve_within_project
from .post_extract import validate_post_extract_contract
from .profile import ResolvedProfile
from .prompts import PromptError, resolve_prompt
from .providers import (
    ProviderConfigurationError,
    ProviderNotFoundError,
    get_embedding_provider,
    get_inference_provider,
)
from .retrieval import (
    PUBLISHER_FENCING_FEATURES,
    RetrievalCapabilityError,
    RetrievalFeature,
    StoreRole,
    create_store,
    store_class,
)
from .sql_models import (
    SqlModelError,
    discover_refs,
    read_sql_source,
    validate_single_select,
)
from .test_specs import (
    TestSpecError,
    declared_accepted_values,
    enum_test_drift,
    has_model_tests,
    parse_test_spec,
)
from .transforms import transform_requires_llm, validate_transform_contract

log = logging.getLogger(__name__)

_MODULE_PATTERN = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")


def validate_project_contract(
    project: ProjectConfig,
    sources: list[SourceConfig],
    models: list[ModelConfig],
    project_dir: Path,
) -> ProjectDAG:
    default_backend = project.extraction.default_backend
    for model in models:
        extraction = model.extraction
        if extraction is not None and (extraction.backend or default_backend) == "llm":
            protect_model_llm_credential_option(model)

    source_names = {source.name for source in sources}
    model_names = {model.name for model in models}
    models_by_name = {model.name: model for model in models}
    search_names = {model.name for model in models if model.search is not None}
    duplicates = source_names & model_names
    if duplicates:
        duplicate_model = next(model for model in models if model.name in duplicates)
        raise _model_error(
            duplicate_model,
            f"Source and model names must be unique; duplicated: {sorted(duplicates)}",
            ("name",),
        )

    available_backends = set(list_backends())
    if default_backend not in available_backends:
        raise ConfigError(
            project.format_yaml_diagnostic(
                f"Default extraction backend '{default_backend}' is not registered. "
                f"Available: {sorted(available_backends)}",
                relative_path=("extraction", "default_backend"),
            )
        )

    # SQL transforms derive their `depends_on` from the `ref()` calls in the
    # `.sql` file; populate it before edge validation and DAG construction so the
    # existing lineage/selector/state machinery treats them like any transform.
    for model in models:
        if model.transform is not None and model.transform.type == "sql":
            _prepare_sql_transform(model, project_dir)

    for model in models:
        _validate_prompt(model, project_dir)
        _validate_tests(model, source_names, model_names, project_dir)
        _validate_model_edges(model, source_names, model_names, search_names)
        _validate_materialization(model)
        if (
            model.search is not None
            and model.search.vector is not None
            and model.search.vector.embedding == "inherit"
        ):
            try:
                resolve_search_embedding_identity(model, models_by_name)
            except (
                ValueError,
                ProviderNotFoundError,
                ProviderConfigurationError,
            ) as error:
                raise _model_error(
                    model,
                    str(error),
                    ("search", "vector", "embedding"),
                ) from error
        _validate_retrieval_tests(model, model_names)
        if model.extraction is not None:
            backend = model.extraction.backend or default_backend
            if backend not in available_backends:
                raise _model_error(
                    model,
                    f"Extraction model '{model.name}' uses unregistered backend "
                    f"'{backend}'. Available: {sorted(available_backends)}",
                    ("extraction", "backend"),
                )
            if backend == "llm" and "api_key_env" in model.extraction.options:
                raise _model_error(
                    model,
                    "llm option 'api_key_env' is operator-owned configuration; "
                    "set it under `llm:` in profiles.yml, not in model "
                    "extraction options",
                    ("extraction", "options", "api_key_env"),
                )
            try:
                canonical_options = validate_backend_options(
                    backend, model.extraction.options
                )
            except BackendOptionsError as e:
                error_path = getattr(e, "path", ("options",))
                raise _model_error(
                    model,
                    f"Extraction model '{model.name}' has {e}",
                    ("extraction", *error_path),
                ) from e
            # Provider checks here only apply when the model pins one — the
            # canonical default may not be the effective provider, which the
            # profile selects. resolve_llm_options re-validates registration
            # and batch capability against the resolved profile.
            if backend == "llm" and "provider" in model.extraction.options:
                provider_name = str(canonical_options["provider"])
                try:
                    provider = get_inference_provider(provider_name)
                except (ProviderNotFoundError, ProviderConfigurationError) as e:
                    raise _model_error(
                        model,
                        str(e),
                        ("extraction", "options", "provider"),
                    ) from e
                if canonical_options.get("batch") and not provider.supports_native_batch:
                    raise _model_error(
                        model,
                        f"Inference provider '{provider_name}' does not support "
                        "native batch execution",
                        ("extraction", "options", "batch"),
                    )
            if backend == "llm" and "cache_path" in model.extraction.options:
                try:
                    resolve_within_project(
                        model.extraction.options["cache_path"],
                        project_dir,
                        surface=f"Model '{model.name}' llm cache_path",
                        hint="Set llm.cache_path in profiles.yml for locations "
                        "outside the project.",
                    )
                except ConfigError as e:
                    raise _model_error(
                        model,
                        str(e),
                        ("extraction", "options", "cache_path"),
                    ) from e
            _validate_post_extract(model, project_dir)
        if model.transform is not None:
            _validate_transform(model, project_dir)
        if model.embed is not None:
            try:
                get_embedding_provider(model.embed.provider)
            except (ProviderNotFoundError, ProviderConfigurationError) as e:
                raise _model_error(
                    model,
                    str(e),
                    ("embed", "provider"),
                ) from e
        if model.llm is not None and model.llm.provider != "default":
            # `default` defers to the profile's LLM provider, resolved at run
            # time; a concrete name must resolve to a registered provider now.
            try:
                get_inference_provider(model.llm.provider)
            except (ProviderNotFoundError, ProviderConfigurationError) as e:
                raise _model_error(
                    model,
                    str(e),
                    ("llm", "provider"),
                ) from e
    try:
        validate_ml_project_contracts(models, project, project_dir)
    except MLContractError as e:
        implicated = next(
            (model for model in models if model.name == e.model_name),
            None,
        )
        if implicated is None:
            raise ConfigError(str(e)) from e
        raise _model_error(implicated, str(e), e.path) from e

    try:
        dag = ProjectDAG(sources, models)
    except DAGError as e:
        raise ConfigError(f"Invalid project DAG: {e}") from e
    for name in search_names:
        if dag.successors[name]:
            model = next(item for item in models if item.name == name)
            raise _model_error(
                model,
                f"Search resource '{name}' must be a leaf serving sink",
                ("depends_on",),
            )
    return dag


def validate_warehouse_capabilities(
    models: list[ModelConfig], adapter: str | WarehouseAdapter
) -> None:
    if isinstance(adapter, str):
        active_adapter = None
        adapter_type = adapter
    else:
        active_adapter = adapter
        adapter_type = adapter.adapter_type()
    available = adapter_capabilities(adapter_type)
    for model in models:
        required: dict[WarehouseCapability, str] = {}
        # An Iceberg-format target (issue #163) is created and replaced through a
        # non-atomic explicit-DDL path, so it is gated by ICEBERG_TABLE_FORMAT
        # rather than ATOMIC_FULL_REPLACE. An active adapter validates and merges
        # effective options here; string-only callers retain the legacy raw check.
        parsed_options: object | None = None
        if active_adapter is not None:
            try:
                parsed_options = active_adapter.parse_warehouse_options(
                    model.warehouse_options,
                    model_name=model.name,
                )
            except AdapterError as error:
                raise _model_error(
                    model,
                    str(error),
                    ("warehouse_options",),
                ) from None
        is_iceberg = (
            getattr(parsed_options, "table_format", None) == "iceberg"
            if active_adapter is not None
            else model.warehouse_options.get("table_format") == "iceberg"
        )
        if model.search is not None:
            required[WarehouseCapability.STREAMING_TABULAR_READS] = (
                "bounded search-index publication reads"
            )
            required[WarehouseCapability.PAGED_STATE_RECONCILIATION] = (
                "bounded publication-state reconciliation"
            )
        elif model.materialization == "full":
            if is_iceberg:
                required[WarehouseCapability.ICEBERG_TABLE_FORMAT] = (
                    "iceberg full materialization"
                )
            else:
                required[WarehouseCapability.ATOMIC_FULL_REPLACE] = (
                    "full materialization"
                )
        else:
            required[WarehouseCapability.ATOMIC_KEYED_UPSERT] = (
                "incremental materialization"
            )
            if is_iceberg:
                required[WarehouseCapability.ICEBERG_TABLE_FORMAT] = (
                    "iceberg incremental materialization"
                )
        if model.extraction is not None:
            required[WarehouseCapability.TYPED_EMPTY_RELATIONS] = (
                "empty extraction results"
            )
            required[WarehouseCapability.CHUNKED_WRITES] = (
                "bounded extraction writes"
            )
        if (
            model.transform is not None
            or model.ml is not None
            or model.chunk is not None
            or model.embed is not None
            or model.llm is not None
        ):
            required[WarehouseCapability.TABULAR_READS] = (
                f"{_kind_label(model).lower()} input reads"
            )
        if model.embed is not None:
            # Embed reads its upstream as a stream rather than one frame
            # (issue #410), on every run and not only on resume. Required at
            # preflight so a warehouse without it fails before credentials and
            # provider spend, not partway through a corpus.
            required[WarehouseCapability.STREAMING_TABULAR_READS] = (
                "bounded embed input reads"
            )
        if model.llm is not None:
            # Native llm uses projected snapshots for both its validation
            # pass and flush-window generation (issue #424). Declare that
            # dependency before runtime resolution can touch provider config.
            required[WarehouseCapability.STREAMING_TABULAR_READS] = (
                "bounded llm input reads"
            )
        # Derived enum checks count: they are schema tests the adapter must
        # support, and finding that out after materialization would move a
        # predictable configuration failure past warehouse mutation (#304).
        if has_model_tests(model) and model.search is None:
            required[WarehouseCapability.SQL_SCHEMA_TESTS] = "model tests"
        if (
            model.materialization == "incremental"
            and model.on_schema_change == "append_new_columns"
        ):
            required[WarehouseCapability.SCHEMA_EVOLUTION] = (
                "on_schema_change=append_new_columns"
            )

        missing = sorted(set(required) - available, key=lambda item: item.value)
        if not missing:
            continue
        details = ", ".join(
            f"{capability.value} ({required[capability]})"
            for capability in missing
        )
        raise _model_error(
            model,
            f"Warehouse adapter '{adapter_type}' cannot execute model "
            f"'{model.name}'; missing capabilities: {details}",
        )


def validate_retrieval_capabilities(
    models: list[ModelConfig],
    project: ProjectConfig,
    resolved: ResolvedProfile,
) -> None:
    search_models = [model for model in models if model.search is not None]
    if not search_models:
        return
    if resolved.retrieval is None:
        raise ConfigError(
            "Selected search resources require a `retrieval:` block in the active profile"
        )
    seen_collections: dict[tuple[str, str], str] = {}
    for model in search_models:
        search = model.search
        assert search is not None
        alias = search.store or resolved.retrieval.default
        config = resolved.retrieval.stores.get(alias)
        if config is None:
            raise _model_error(
                model,
                f"Search resource '{model.name}' selects unknown retrieval store "
                f"'{alias}'. Available: {sorted(resolved.retrieval.stores)}",
                ("search", "store"),
            )
        if search.access == "public" and not resolved.retrieval.allow_public_indexes:
            raise _model_error(
                model,
                f"Search resource '{model.name}' is public but the active profile does "
                "not set retrieval.allow_public_indexes: true",
                ("search", "access"),
            )
        if search.index_options:
            raise _model_error(
                model,
                f"Retrieval store '{config.type}' does not accept index_options in the "
                "reference implementation",
                ("search", "index_options"),
            )
        cls = store_class(config.type)
        capabilities = cls.capabilities()
        if not capabilities.features & PUBLISHER_FENCING_FEATURES:
            raise _model_error(
                model,
                f"Retrieval store '{config.type}' declares no publisher fencing "
                "proof; a warehouse fencing token alone cannot exclude a stale "
                "writer from an independent store (issue #152)",
                ("search", "store"),
            )
        if search.access == "governed":
            if "strong" not in capabilities.consistency_modes:
                raise _model_error(
                    model,
                    f"Governed search publication requires strong read-after-write "
                    f"consistency, which retrieval store '{config.type}' does not "
                    "declare",
                    ("search", "access"),
                )
            if RetrievalFeature.METADATA_FILTERING not in capabilities.features:
                raise _model_error(
                    model,
                    f"Governed search indexes require mandatory policy prefilters, "
                    f"which retrieval store '{config.type}' cannot execute",
                    ("search", "access"),
                )
        required: dict[RetrievalFeature, str] = {
            RetrievalFeature.KEYED_UPSERT: "incremental publication",
            RetrievalFeature.KEYED_DELETE: "stale-record deletion",
            RetrievalFeature.DURABLE_WRITE_ACK: "receipt-gated warehouse state",
            RetrievalFeature.ATOMIC_BATCH_MUTATION: "exact whole-batch receipts",
            RetrievalFeature.INDEX_READINESS: "post-publication index validation",
        }
        if search.on_index_change == "online":
            required[RetrievalFeature.PRIVATE_GENERATION_BUILD] = (
                "safe generation replacement for `on_index_change: online`"
            )
        if search.vector is not None:
            required[
                RetrievalFeature.APPROXIMATE_VECTOR_SEARCH
                if search.vector.search == "approximate"
                else RetrievalFeature.EXACT_VECTOR_SEARCH
            ] = f"{search.vector.search} vector search"
            if search.vector.metric not in capabilities.distance_metrics:
                raise _model_error(
                    model,
                    f"Retrieval store '{config.type}' does not support distance metric "
                    f"'{search.vector.metric}'",
                    ("search", "vector", "metric"),
                )
        if search.full_text is not None:
            required[RetrievalFeature.FULL_TEXT_SEARCH] = "full-text index"
        if any(attribute.filter_role != "none" for attribute in search.attributes):
            required[RetrievalFeature.METADATA_FILTERING] = "typed metadata filtering"
        try:
            capabilities.require(required, store_type=config.type)
        except RetrievalCapabilityError as error:
            raise _model_error(model, str(error), ("search",)) from None
        if search.batch_size > capabilities.max_batch_size:
            raise _model_error(
                model,
                f"Search batch_size exceeds retrieval store '{config.type}' limit of "
                f"{capabilities.max_batch_size}",
                ("search", "batch_size"),
            )
        if (
            search.vector is not None
            and capabilities.max_dimensions is not None
            and search.vector.dimensions > capabilities.max_dimensions
        ):
            raise _model_error(
                model,
                f"Search vector dimensions exceed retrieval store '{config.type}' "
                f"limit of {capabilities.max_dimensions}",
                ("search", "vector", "dimensions"),
            )
        store = create_store(
            config,
            project_name=project.name,
            target_name=resolved.target_name,
            alias=alias,
            # Validation only: capabilities and refusals, no data access.
            role=StoreRole.INSPECT,
        )
        # Asked of the constructed store, not of its capability set: some
        # refusals depend on the resolved store config rather than the store
        # type. This has to happen before any publish, because a compatible
        # vector-search change is applied to the *live* collection — an index
        # the store turns out to refuse would be discovered after every row had
        # been republished and the serving pointer cleared (Codex review, #461).
        refusal = store.index_config_refusal(
            vector_search=search.vector.search if search.vector else None,
            vector_index=search.vector.index if search.vector else None,
        )
        if refusal is not None:
            # A non-default index type is the more specific choice, so a
            # refusal under one is reported against it; otherwise the search
            # mode is what asked for an index at all.
            chose_index = (
                search.vector is not None and search.vector.index != DEFAULT_VECTOR_INDEX
            )
            field = "index" if chose_index else "search"
            raise _model_error(model, refusal, ("search", "vector", field))
        logical = search.collection or model.name
        physical = store.physical_collection(logical)
        key = (store.safe_descriptor().safe_target_identity, physical)
        previous = seen_collections.get(key)
        if previous is not None:
            raise _model_error(
                model,
                f"Search resources '{previous}' and '{model.name}' resolve to the same "
                "retrieval collection",
                ("search", "collection"),
            )
        seen_collections[key] = model.name


def validate_warehouse_operation_capabilities(
    adapter_type: str,
    required: Mapping[WarehouseCapability, str],
    *,
    operation: str,
) -> None:
    """Preflight a non-model warehouse operation before adapter construction."""
    available = adapter_capabilities(adapter_type)
    missing = sorted(set(required) - available, key=lambda item: item.value)
    if not missing:
        return
    details = ", ".join(
        f"{capability.value} ({required[capability]})"
        for capability in missing
    )
    raise ConfigError(
        f"Warehouse adapter '{adapter_type}' cannot execute {operation}; "
        f"missing capabilities: {details}"
    )


def _validate_prompt(model: ModelConfig, project_dir: Path) -> None:
    """Resolve a versioned prompt reference at compile time (issue #303).

    Deliberately not left to run time: the artifact-safe descriptor path
    swallows resolution failures so offline docs tooling still works, so a
    misspelled version would otherwise survive `compile` and surface only
    after source discovery and credentials — a typo that costs a corpus
    instead of nothing.
    """
    if model.llm is None or isinstance(model.llm.prompt, str):
        return
    try:
        resolve_prompt(model.llm, project_dir, model_name=model.name)
    except PromptError as error:
        raise _model_error(model, str(error), ("llm", "prompt")) from error


def _validate_tests(
    model: ModelConfig,
    source_names: set[str],
    model_names: set[str],
    project_dir: Path,
) -> None:
    if model.search is not None and model.tests:
        raise _model_error(
            model,
            "Search resources do not run warehouse schema tests; portable retrieval "
            "tests are delivered with the #135 query contract",
            ("tests",),
        )
    for index, spec in enumerate(model.tests):
        try:
            parsed = parse_test_spec(spec)
        except TestSpecError as e:
            raise _model_error(
                model,
                f"Model '{model.name}' test[{index}] is invalid: {e}",
                ("tests", index),
            ) from e
        if parsed.name == "python":
            module_path = parsed.argument
            assert isinstance(module_path, str)
            _validate_python_test(model, index, module_path, project_dir)
        target = parsed.relationship_target
        if target is None:
            continue
        if target in source_names:
            raise _model_error(
                model,
                f"Model '{model.name}' relationships test target '{target}' is a "
                "source; relationship targets must be models",
                ("tests", index),
            )
        if target not in model_names:
            raise _model_error(
                model,
                f"Model '{model.name}' relationships test references unknown model "
                f"'{target}'",
                ("tests", index),
            )
    _warn_on_enum_test_drift(model)


def _warn_on_enum_test_drift(model: ModelConfig) -> None:
    """Warn when a hand-written accepted_values disagrees with a field's enum.

    An explicit check on an `enum` field is redundant — the derived one already
    covers it (issue #304) — so the only thing a disagreement can mean is that
    one of the two lists has drifted. Which one is right is the author's call,
    so this reports rather than decides.
    """
    drift = enum_test_drift(model.fields, declared_accepted_values(model.tests))
    for name, values, explicit in drift:
        log.warning(
            "Model '%s' field '%s' declares `values: %s` but an "
            "accepted_values test allows %s. The declared set is what the "
            "provider schema and the prompt use, so the test is checking a "
            "different taxonomy than the model was asked for.",
            model.name,
            name,
            sorted(values),
            sorted(map(str, explicit)),
        )


def _validate_python_test(
    model: ModelConfig, index: int, module_path: str, project_dir: Path
) -> None:
    if not _MODULE_PATTERN.fullmatch(module_path):
        raise _model_error(
            model,
            f"Model '{model.name}' test[{index}] python module '{module_path}' is not "
            "a valid dotted Python module path",
            ("tests", index),
        )
    # Local import avoids a compiler <-> checks package import cycle.
    from .checks.python import CustomTestError, load_python_test

    try:
        load_python_test(module_path, project_dir)
    except CustomTestError as e:
        raise _model_error(
            model,
            f"Model '{model.name}' test[{index}] python module '{module_path}' is "
            f"invalid: {e}",
            ("tests", index),
        ) from e


def _validate_model_edges(
    model: ModelConfig,
    source_names: set[str],
    model_names: set[str],
    search_names: set[str],
) -> None:
    if model.kind_block_count != 1:
        raise _model_error(
            model,
            f"Model '{model.name}' must declare exactly one of "
            f"{'/'.join(ModelKind)}",
        )

    has_dbt_ref = model.source is not None and is_dbt_ref(model.source)
    if has_dbt_ref:
        # A `dbt_ref('...')` source names a dbt-built table (reverse direction,
        # #177): transform-only, resolved by dbt in embedded mode, never
        # validated against the stel graph. `depends_on:` may additionally
        # name stel models feeding the same transform — validated like any
        # other transform's below, since the dbt_ref alone already satisfies
        # "at least one input."
        if model.transform is None or model.transform.type != "python":
            # SQL transforms execute via run_sql_model against warehouse-native
            # relations (or the embedded CaptureAdapter's scratch database in
            # dbt-duckdb mode); neither path resolves a dbt_ref/upstream frame
            # the way run_transform_model's python-deps injection does, so this
            # would compile but fail at dbt-build time (Codex review, #177).
            raise _model_error(
                model,
                f"{_kind_label(model)} model '{model.name}' may not use a "
                "`dbt_ref(...)` source; it is supported only on `type: python` "
                "transform models",
                ("source",),
            )
        assert model.source is not None
        target = parse_dbt_ref(model.source)
        if target in model_names or target in source_names or target in search_names:
            raise _model_error(
                model,
                f"Transform model '{model.name}' dbt_ref('{target}') names a stel "
                "node; a dbt_ref must name a dbt-built table outside the stel graph",
                ("source",),
            )
    elif model.extraction is not None:
        if not model.source:
            raise _model_error(
                model,
                f"Extraction model '{model.name}' must declare exactly one `source:`",
                ("source",),
            )
        if model.depends_on is not None:
            raise _model_error(
                model,
                f"Extraction model '{model.name}' must use `source:`, not `depends_on:`",
                ("depends_on",),
            )
        target = parse_ref(model.source)
        if target in model_names:
            raise _model_error(
                model,
                f"Extraction model '{model.name}' source '{target}' is a model; "
                "extraction sources must reference source nodes",
                ("source",),
            )
        if target not in source_names:
            raise _model_error(
                model,
                f"Extraction model '{model.name}' references unknown source '{target}'",
                ("source",),
            )
        return
    elif model.source is not None:
        raise _model_error(
            model,
            f"{_kind_label(model)} model '{model.name}' must use `depends_on:`, "
            "not `source:`",
            ("source",),
        )

    dependencies = model.depends_on or []
    if model.search is not None and len(dependencies) != 1:
        raise _model_error(
            model,
            f"Search resource '{model.name}' must declare exactly one `depends_on:` model",
            ("depends_on",),
        )
    if model.transform is not None and not dependencies and not has_dbt_ref:
        raise _model_error(
            model,
            f"Transform model '{model.name}' must declare at least one `depends_on:` model",
            ("depends_on",),
        )
    if model.ml is not None and not dependencies:
        raise _model_error(
            model,
            f"ML model '{model.name}' must declare at least one `depends_on:` model",
            ("depends_on",),
        )
    if model.chunk is not None and len(dependencies) != 1:
        raise _model_error(
            model,
            f"Chunk model '{model.name}' must declare exactly one `depends_on:` model",
            ("depends_on",),
        )
    if model.embed is not None and len(dependencies) != 1:
        raise _model_error(
            model,
            f"Embed model '{model.name}' must declare exactly one `depends_on:` model",
            ("depends_on",),
        )
    if model.llm is not None and len(dependencies) != 1:
        raise _model_error(
            model,
            f"llm model '{model.name}' must declare exactly one `depends_on:` model",
            ("depends_on",),
        )

    dependency_targets = [parse_ref(dependency) for dependency in dependencies]
    duplicate_targets = sorted(
        target for target in set(dependency_targets) if dependency_targets.count(target) > 1
    )
    if duplicate_targets:
        raise _model_error(
            model,
            f"{_kind_label(model)} model '{model.name}' declares duplicate "
            f"dependencies: {duplicate_targets}",
            ("depends_on",),
        )

    for index, target in enumerate(dependency_targets):
        if target in source_names:
            raise _model_error(
                model,
                f"{_kind_label(model)} model '{model.name}' dependency '{target}' is "
                "a source; non-extraction models must depend on models",
                ("depends_on", index),
            )
        if target not in model_names:
            raise _model_error(
                model,
                f"{_kind_label(model)} model '{model.name}' references unknown model "
                f"'{target}'",
                ("depends_on", index),
            )
        if model.search is not None and target in search_names:
            raise _model_error(
                model,
                f"Search resource '{model.name}' cannot depend on search resource '{target}'",
                ("depends_on", index),
            )
        if model.search is None and target in search_names:
            raise _model_error(
                model,
                f"Warehouse model '{model.name}' cannot depend on search resource '{target}'",
                ("depends_on", index),
            )


def _validate_retrieval_tests(model: ModelConfig, model_names: set[str]) -> None:
    if not model.retrieval_tests:
        return
    if model.search is None:
        raise _model_error(
            model,
            f"Model '{model.name}' declares `retrieval_tests`, which only "
            "applies to `search:` models (issue #137)",
            ("retrieval_tests",),
        )
    names = [t.name for t in model.retrieval_tests]
    if len(names) != len(set(names)):
        raise _model_error(
            model,
            f"Search model '{model.name}' declares duplicate retrieval_tests "
            f"names: {sorted({n for n in names if names.count(n) > 1})}",
            ("retrieval_tests",),
        )
    for test in model.retrieval_tests:
        golden = parse_ref(test.golden_set)
        if golden not in model_names:
            raise _model_error(
                model,
                f"Retrieval test '{test.name}' on '{model.name}' references "
                f"unknown golden_set model '{golden}'",
                ("retrieval_tests",),
            )
        if golden == model.name:
            raise _model_error(
                model,
                f"Retrieval test '{test.name}' on '{model.name}' cannot use "
                "the search model itself as its golden_set",
                ("retrieval_tests",),
            )
        if test.mode is not None and test.mode not in model.search.query.modes:
            raise _model_error(
                model,
                f"Retrieval test '{test.name}' on '{model.name}' sets "
                f"`mode: {test.mode}`, which the search index does not declare "
                f"in `query.modes` ({sorted(model.search.query.modes)})",
                ("retrieval_tests",),
            )


def _validate_materialization(model: ModelConfig) -> None:
    if model.search is not None:
        if model.materialization not in ("incremental", "full"):
            raise _model_error(
                model,
                "Search resources support `materialization: incremental` or "
                "`materialization: full`",
                ("materialization",),
            )
        if model.warehouse_options:
            raise _model_error(
                model,
                "Search resources cannot declare warehouse_options",
                ("warehouse_options",),
            )
        return
    is_incremental_sql_transform = (
        model.transform is not None
        and model.transform.type == "sql"
        and model.materialization == "incremental"
    )
    # Python transforms may be incremental when they declare an
    # `IncrementalContract` (issue #218); `_validate_transform` enforces that a
    # contract is present and agrees with `depends_on`. SQL transforms keep the
    # `unique_key` route (#142); every other model kind is full-only.
    is_incremental_python_transform = (
        model.transform is not None
        and model.transform.type == "python"
        and model.materialization == "incremental"
    )
    if (
        model.transform is not None
        and model.materialization != "full"
        and not (is_incremental_sql_transform or is_incremental_python_transform)
    ):
        raise _model_error(
            model,
            f"Transform model '{model.name}' only supports `materialization: full`",
            ("materialization",),
        )
    if model.unique_key is not None and not is_incremental_sql_transform:
        raise _model_error(
            model,
            f"Model '{model.name}' declares `unique_key`, which only applies to "
            "`transform: {type: sql}` with `materialization: incremental` (#142)",
            ("unique_key",),
        )
    if model.ml is not None and model.materialization != "full":
        raise _model_error(
            model,
            f"ML model '{model.name}' only supports `materialization: full`",
            ("materialization",),
        )


def _prepare_sql_transform(model: ModelConfig, project_dir: Path) -> None:
    """Resolve + read a SQL transform's `.sql` file, discover its literal refs,
    and populate `depends_on` from them (validating agreement with any explicit
    `depends_on`). Runs before edge validation and DAG construction."""
    assert model.transform is not None
    transform = model.transform
    if not transform.path:
        raise _model_error(
            model,
            f"SQL transform model '{model.name}' requires a `path:` to a .sql file",
            ("transform", "path"),
        )
    if model.agent_context is not None:
        # The Python transform path validates agent_context outputs
        # (_validate_agent_context_output) before materializing; there is no
        # warehouse-side equivalent yet, so a SQL model must not advertise an
        # unverified agent_context contract. Reject until #145's contract check
        # can run against the materialized relation.
        raise _model_error(
            model,
            f"SQL transform model '{model.name}' does not support `agent_context` "
            "yet: SQL outputs are not validated against the contract before "
            "publication. Use a python transform, or drop agent_context.",
            ("agent_context",),
        )
    if model.materialization == "incremental" and not model.unique_key:
        raise _model_error(
            model,
            f"SQL transform model '{model.name}' declares `materialization: "
            "incremental` and requires a `unique_key:` column (#142)",
            ("unique_key",),
        )
    try:
        resolved = resolve_within_project(
            transform.path, project_dir, surface=f"model '{model.name}' transform.path"
        )
        sql_text = read_sql_source(resolved, model_name=model.name)
        validate_single_select(sql_text, model_name=model.name)
        refs = discover_refs(sql_text, model_name=model.name)
    except (SqlModelError, ConfigError) as e:
        raise _model_error(model, str(e), ("transform", "path")) from e

    if not refs:
        raise _model_error(
            model,
            f"SQL transform model '{model.name}' must ref() at least one upstream "
            "model",
            ("transform", "path"),
        )

    if model.depends_on:
        declared = {parse_ref(dep) for dep in model.depends_on}
        if declared != set(refs):
            raise _model_error(
                model,
                f"SQL transform model '{model.name}' declares depends_on "
                f"{sorted(declared)} but its SQL ref()s {sorted(refs)}; they must "
                "match (depends_on is optional for SQL models).",
                ("depends_on",),
            )
    # Canonical ref() form so parse_ref and the DAG treat it uniformly.
    model.depends_on = [f"ref('{name}')" for name in refs]


def _validate_transform(model: ModelConfig, project_dir: Path) -> None:
    assert model.transform is not None
    transform = model.transform
    if transform.type == "sql":
        # Source, refs, and depends_on were validated in _prepare_sql_transform.
        return
    if transform.type != "python":
        raise _model_error(
            model,
            f"Transform model '{model.name}' has unsupported type '{transform.type}'; "
            "supported: python, sql",
            ("transform", "type"),
        )
    if not transform.module:
        raise _model_error(
            model,
            f"Transform model '{model.name}' requires a `module:`",
            ("transform", "module"),
        )
    if not _MODULE_PATTERN.fullmatch(transform.module):
        raise _model_error(
            model,
            f"Transform model '{model.name}' module '{transform.module}' is not a "
            "valid dotted Python module path",
            ("transform", "module"),
        )
    # A dbt_ref('...') source (#177) forms no stel DAG edge, but the module's
    # declared_dependencies must still name it alongside any `depends_on:`
    # stel models — run_transform_model resolves both into the same `deps`
    # dict the transform receives, so the contract is validated against the
    # union, not either alone.
    declared_dependencies = (
        [parse_dbt_ref(model.source)] if model.source and is_dbt_ref(model.source) else []
    ) + [parse_ref(dependency) for dependency in model.depends_on or []]
    try:
        validate_transform_contract(
            transform.module,
            project_dir,
            transform.options,
            declared_dependencies,
            materialization=model.materialization,
        )
    except (Exception, SystemExit) as e:
        raise _model_error(
            model,
            f"Transform model '{model.name}' module '{transform.module}' is invalid: {e}",
            ("transform", "module"),
        ) from e
    if (
        not transform.uses_llm
        and transform_requires_llm(transform.module, project_dir, transform.options)
    ):
        raise _model_error(
            model,
            f"Transform model '{model.name}' uses an inference-based extractor and "
            "requires `transform.uses_llm: true` so an `llm:` provider is resolved "
            "and the model reprocesses when the provider or model changes",
            ("transform", "uses_llm"),
        )


def _validate_post_extract(model: ModelConfig, project_dir: Path) -> None:
    assert model.extraction is not None
    hook = model.extraction.post_extract
    if hook is None:
        return
    if not _MODULE_PATTERN.fullmatch(hook.module):
        raise _model_error(
            model,
            f"Extraction model '{model.name}' post_extract module "
            f"'{hook.module}' is not a valid dotted Python module path",
            ("extraction", "post_extract", "module"),
        )
    validation_failed = False
    try:
        validate_post_extract_contract(hook.module, project_dir, hook.options)
    except (Exception, SystemExit):
        # Hook options can contain prompts or other sensitive project values.
        # Leave the except block before raising so neither the original
        # exception nor its traceback survives as __context__.
        validation_failed = True
    if validation_failed:
        raise _model_error(
            model,
            f"Extraction model '{model.name}' post_extract module "
            f"'{hook.module}' could not be loaded or validated; ensure the "
            "project module exists, defines run(fields[, ctx]), and accepts "
            "the configured options",
            ("extraction", "post_extract", "module"),
        ) from None


def _model_error(
    model: ModelConfig,
    message: str,
    relative_path: ConfigPath = (),
) -> ConfigError:
    return ConfigError(
        model.format_yaml_diagnostic(message, relative_path=relative_path)
    )


def _kind_label(model: ModelConfig) -> str:
    if model.transform is not None:
        return "Transform"
    if model.ml is not None:
        return "ML"
    if model.chunk is not None:
        return "Chunk"
    if model.embed is not None:
        return "Embed"
    if model.llm is not None:
        return "llm"
    if model.search is not None:
        return "Search"
    if model.eval is not None:
        return "Eval"
    return "Unknown"
