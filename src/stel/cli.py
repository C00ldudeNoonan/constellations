from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

import click

from ._distribution import distribution_version
from .adapters import (
    AdapterError,
    apply_name_migration,
    create_adapter,
    plan_name_migration,
)
from .checks import TestResult, run_project_tests
from .cli_services.context import CONFIG_ERRORS as _CONFIG_ERRORS
from .cli_services.context import ConfigClickError
from .cli_services.context import build_dag_or_click as _build_dag
from .cli_services.context import load_project_or_click as _load
from .cli_services.watch import run_watch as _run_watch
from .compiler import validate_project_contract, validate_warehouse_capabilities
from .concept_cloud import (
    ConceptCloudExportError,
    demo_export,
    export_concept_cloud,
    placeholder_export,
    write_concept_cloud,
)
from .config import ConfigError
from .config.identifiers import PROJECT_FILENAME
from .config.model import ModelConfig
from .config.project import ProjectConfig
from .config.source import SourceConfig
from .credentials import CredentialReference, CredentialReferenceError
from .dag import SelectionError, parse_ref
from .dbt_export import write_dbt_sources
from .docs import DocsError, generate_docs, serve_docs
from .freshness import check_freshness
from .logging_setup import configure_verbose_logging, resolve_verbosity
from .manifest import write_manifest, write_run_results
from .optional_dependencies import OptionalDependencyError
from .paths import resolve_within_project
from .profile import (
    PROFILES_FILENAME,
    ProfileError,
    apply_source_path_overrides,
    resolve_llm_options,
    resolve_profile,
)
from .progress import OutputLevel, configure_progress, get_reporter
from .prompts import (
    PromptLockError,
    check_lock,
    lock_path,
    read_lock,
    write_lock,
)
from .providers import (
    get_inference_provider,
)
from .retrieval.servability import (
    DEFAULT_CONTEXT_TIMEOUT_SECONDS,
    MAX_CONTEXT_TIMEOUT_SECONDS,
)
from .retrieval_eval import (
    RetrievalEvalError,
    run_retrieval_evaluation,
    write_retrieval_eval_artifact,
)
from .runner import (
    BuildResult,
    ModelRunResult,
    RunError,
    build_project,
    clean_project,
    run_project,
)
from .search import (
    SearchError,
    SearchFilter,
    SearchFilterOperator,
    SearchMode,
    SearchRequest,
    SearchResult,
)
from .search import (
    search as run_search,
)
from .sources import SourceError
from .suggest import DEFAULT_MIN_EVIDENCE
from .synth import (
    generate_arxiv_papers,
    generate_invoice_pdfs,
    generate_invoice_texts,
    generate_invoices,
    generate_posts,
    generate_product_pages,
    generate_support_emails,
    generate_support_tickets,
)


def _context_override(
    name: str,
    *,
    resolve_path: bool = False,
) -> Callable[[click.Context, click.Parameter, Any], Any]:
    def callback(
        ctx: click.Context,
        _param: click.Parameter,
        value: Any,
    ) -> Any:
        if value is not None:
            ctx.ensure_object(dict)
            ctx.obj[name] = value.resolve() if resolve_path else value
        return value

    return callback


def _verbose_option(command: Callable[..., Any]) -> Callable[..., Any]:
    """``-v`` for progress output. Enables per-source discovery lines,
    per-model start/finish lines, and a live progress bar on a TTY (non-TTY
    stderr gets plain log lines instead). Also honored via ``STEL_VERBOSE``.
    Extra ``v``s are silently capped — verbose is always INFO-level; the flag
    intentionally does not expose DEBUG."""
    return click.option(
        "-v",
        "--verbose",
        "verbose",
        count=True,
        help=(
            "Enable progress output: per-source discovery, per-model "
            "start/finish, and a TTY progress bar. Also honored via "
            "STEL_VERBOSE=1."
        ),
    )(command)


def _configure_output(
    verbose_count: int, *, bars_safe: bool = True, json_output: bool = False
) -> None:
    """Pick the output level for this invocation and install its channels.

    ``--json`` is the machine path, so it goes quiet: the payload on stdout is
    the whole output, and a caller who wants narration alongside it can add
    ``-v`` for the log channel on stderr. Otherwise the default is the running
    ledger (issue #404) and ``-v`` layers detail on top of it — discovery lines,
    progress bars on a TTY, and the forwarded INFO log.

    On a TTY with bars live the log handler is routed through the reporter so
    records defer past a bar instead of writing over it (issue #403).
    ``bars_safe=False`` (``run --threads N`` over multiple models) drops the
    bars: each model would open its own ``click.progressbar`` on the one stderr
    and their redraws would interleave. INFO records are emitted atomically
    under the logging lock, so parallel models stay individually intact."""
    verbosity = resolve_verbosity(verbose_count)
    if json_output:
        level = OutputLevel.QUIET
    elif verbosity > 0:
        level = OutputLevel.VERBOSE
    else:
        level = OutputLevel.NORMAL
    bars = configure_progress(level, bars_safe=bars_safe)
    configure_verbose_logging(
        verbosity, reporter=get_reporter() if bars else None
    )


def _project_context_options(command: Callable[..., Any]) -> Callable[..., Any]:
    """Allow dbt-style global options after a project-aware subcommand."""
    command = click.option(
        "--target",
        default=None,
        expose_value=False,
        callback=_context_override("target"),
        help="Target name within the active profile.",
    )(command)
    command = click.option(
        "--profiles-dir",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        default=None,
        expose_value=False,
        callback=_context_override("profiles_dir", resolve_path=True),
        help="Directory containing profiles.yml. Overrides discovery.",
    )(command)
    command = click.option(
        "--project-dir",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        default=None,
        expose_value=False,
        callback=_context_override("project_dir", resolve_path=True),
        help="Path to the stel project (where stel_project.yml lives).",
    )(command)
    return command


@click.group()
@click.option(
    "--project-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path.cwd,
    help="Path to the stel project (where stel_project.yml lives).",
)
@click.option(
    "--profiles-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Directory containing profiles.yml. Overrides discovery.",
)
@click.option("--target", default=None, help="Target name within the active profile.")
@click.version_option(
    version=distribution_version(),
    prog_name="stel",
    message="%(prog)s %(version)s",
)
@click.pass_context
def cli(
    ctx: click.Context,
    project_dir: Path,
    profiles_dir: Path | None,
    target: str | None,
) -> None:
    ctx.ensure_object(dict)
    ctx.obj["project_dir"] = project_dir.resolve()
    ctx.obj["profiles_dir"] = profiles_dir.resolve() if profiles_dir else None
    ctx.obj["target"] = target


@cli.command()
@_project_context_options
@click.pass_context
def compile(ctx: click.Context) -> None:
    """Parse YAML, validate DAG, write target/manifest.json."""
    project_dir: Path = ctx.obj["project_dir"]
    profiles_dir = ctx.obj["profiles_dir"]
    target = ctx.obj["target"]
    project, sources, models = _load(project_dir)
    try:
        dag = validate_project_contract(project, sources, models, project_dir)
        resolved = resolve_profile(
            project, project_dir, target=target, profiles_dir=profiles_dir
        )
        adapter = create_adapter(resolved.warehouse, project_dir=project_dir)
        validate_warehouse_capabilities(models, adapter)
        manifest_path = write_manifest(
            project_dir, target=target, profiles_dir=profiles_dir
        )
    except _CONFIG_ERRORS as e:
        raise ConfigClickError(str(e)) from e

    click.echo(f"Project: {project.name} v{project.version}")
    click.echo(f"  Sources: {len(sources)}")
    click.echo(f"  Models:  {len(models)}")
    click.echo("")
    click.echo("Execution order:")
    for i, name in enumerate(dag.execution_order(), 1):
        click.echo(f"  {i}. {name}")
    click.echo("")
    click.echo(f"Wrote {manifest_path.relative_to(project_dir)}")

    # Surface backend-related warnings (e.g. a missing LLM credential).
    for warning in _compile_warnings(project, models, project_dir, target, profiles_dir):
        click.echo(f"warning: {warning}", err=True)


def _compile_warnings(
    project: ProjectConfig,
    models: list[ModelConfig],
    project_dir: Path,
    target: str | None,
    profiles_dir: Path | None,
) -> list[str]:
    out: list[str] = []
    uses_llm = any(
        (
            model.extraction is not None
            and (
                model.extraction.backend or project.extraction.default_backend
            )
            == "llm"
        )
        or (model.transform is not None and model.transform.uses_llm)
        for model in models
    )

    if uses_llm:
        try:
            resolved = resolve_profile(
                project, project_dir, target=target, profiles_dir=profiles_dir
            )
        except ProfileError:
            return out

        missing: set[str] = set()
        for model in models:
            if model.extraction is not None and (
                model.extraction.backend or project.extraction.default_backend
            ) == "llm":
                options = resolve_llm_options(model.extraction.options, resolved)
            elif model.transform is not None and model.transform.uses_llm:
                options = resolve_llm_options({}, resolved)
            else:
                continue
            provider = get_inference_provider(str(options["provider"]))
            if options.get("batch") and not options.get("cache_path"):
                out.append(
                    f"Model '{model.name}' uses batch extraction without "
                    "llm.cache_path; interrupted batch jobs cannot be resumed "
                    "and completed responses are not cached."
                )
            env_value = options.get("api_key_env") or provider.default_credential_env
            if env_value is None:
                if not provider.requires_credentials:
                    continue
                out.append(
                    f"Inference provider '{provider.name()}' requires credentials; "
                    "configure llm.api_key_env in profiles.yml."
                )
                continue
            try:
                reference = (
                    env_value
                    if isinstance(env_value, CredentialReference)
                    else CredentialReference.from_env_name(env_value)
                )
            except (TypeError, CredentialReferenceError):
                missing.add(provider.name())
                continue
            if not reference.is_available():
                missing.add(provider.name())

        for provider_name in sorted(missing):
            out.append(
                f"Inference provider '{provider_name}' credential environment "
                "variable is not set; models using LLM inference will fail at "
                "run time."
            )

    return out


Seeder = Callable[[int, Path, int], list[Path]]


_SEEDERS_BY_BACKEND: dict[str, Seeder] = {
    "json": generate_invoices,
    "markdown": generate_posts,
    "llm": generate_invoice_texts,
    "pdf": generate_invoice_pdfs,
    "html": generate_product_pages,
    "email": generate_support_emails,
}

_SEEDERS_BY_TYPE: dict[str, Seeder] = {
    "invoices": generate_invoices,
    "posts": generate_posts,
    "invoice_texts": generate_invoice_texts,
    "invoice_pdfs": generate_invoice_pdfs,
    "product_pages": generate_product_pages,
    "tickets": generate_support_tickets,
    "emails": generate_support_emails,
    "arxiv": generate_arxiv_papers,
}


_AVAILABLE_TEMPLATES = ("json", "pdf", "markdown", "html")


@cli.command()
@click.argument("name")
@click.option(
    "--template",
    "template",
    type=click.Choice(_AVAILABLE_TEMPLATES, case_sensitive=False),
    default="json",
    show_default=True,
    help="Which backend to scaffold for.",
)
def init(name: str, template: str) -> None:
    """Scaffold a new stel project at ./<name>/."""
    target = Path.cwd() / name
    if target.exists():
        raise click.ClickException(f"{target} already exists")

    template_dir = Path(__file__).parent / "templates" / template
    if not template_dir.is_dir():
        raise click.ClickException(f"Template directory missing: {template_dir}")

    shutil.copytree(template_dir, target)
    for path in target.rglob(".gitkeep"):
        path.unlink()

    for filename in (PROJECT_FILENAME, PROFILES_FILENAME):
        path = target / filename
        if path.exists():
            path.write_text(
                path.read_text(encoding="utf-8").replace("__PROJECT_NAME__", name),
                encoding="utf-8",
            )

    click.echo(f"Created stel project at {target} (template: {template})")
    click.echo("")
    click.echo("Next:")
    click.echo(f"  cd {name}")
    if template == "json":
        click.echo("  uv run stel seed --count 20")
    else:
        click.echo(
            f"  # drop your {template} files into ./data/, "
            "or `stel seed --count 20` for synthetic data"
        )
    click.echo("  uv run stel run")
    click.echo("  uv run stel test")


@cli.command()
@click.argument("model_name")
@click.option("--limit", default=10, show_default=True, help="Number of rows to show.")
@_project_context_options
@click.pass_context
def show(ctx: click.Context, model_name: str, limit: int) -> None:
    """Pretty-print rows from a materialized model table."""
    project_dir: Path = ctx.obj["project_dir"]
    profiles_dir = ctx.obj["profiles_dir"]
    target = ctx.obj["target"]
    project, _, models = _load(project_dir)
    selected_model = next((model for model in models if model.name == model_name), None)
    if selected_model is not None and selected_model.search is not None:
        raise click.ClickException(
            f"Search resource '{model_name}' has no warehouse relation. Inspect its "
            "serving_resource descriptor in target/manifest.json or query it with "
            "`stel search`."
        )
    try:
        resolved = resolve_profile(
            project, project_dir, target=target, profiles_dir=profiles_dir
        )
    except ProfileError as e:
        raise ConfigClickError(str(e)) from e

    try:
        with create_adapter(resolved.warehouse, project_dir=project_dir) as adapter:
            tables = adapter.list_tables()
            if model_name not in tables:
                raise click.ClickException(
                    f"Table '{model_name}' not found in the "
                    f"{adapter.adapter_type()} target. "
                    f"Run `stel run` first. Available: {tables or '(none)'}"
                )
            df = adapter.read_table(model_name, limit=limit)
    except AdapterError as e:
        raise click.ClickException(str(e)) from e

    stdout = click.get_text_stream("stdout")
    click.echo(_safe_console_text(str(df), stdout), file=stdout)


@cli.command("search")
@click.option("--model", "model_name", required=True, help="Search-index model name.")
@click.option("--query", default=None, help="Query text for text or embedded search.")
@click.option(
    "--mode",
    type=click.Choice([mode.value for mode in SearchMode], case_sensitive=False),
    default=SearchMode.HYBRID.value,
    show_default=True,
)
@click.option("--limit", type=click.IntRange(1, 1000), default=10, show_default=True)
@click.option(
    "--candidate-limit",
    type=click.IntRange(1, 1000),
    default=None,
    help="Candidates requested per retrieval mode before fusion.",
)
@click.option(
    "--filter",
    "filter_values",
    type=(
        str,
        click.Choice([operator.value for operator in SearchFilterOperator]),
        str,
    ),
    multiple=True,
    help=(
        "Repeatable FIELD OP VALUE filter. The 'in' and 'array_contains_any' "
        "operators take a JSON array. An array[string] field must use "
        "'array_contains_any', which matches when the row's list shares any "
        "value with the array given."
    ),
)
@click.option(
    "--field",
    "fields",
    multiple=True,
    help="Declared return field to include. Repeat for multiple fields.",
)
@click.option(
    "--vector",
    "vector_json",
    default=None,
    help="Precomputed query vector as a JSON array.",
)
@click.option(
    "--output",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
)
@_project_context_options
@click.pass_context
def search_command(
    ctx: click.Context,
    model_name: str,
    query: str | None,
    mode: str,
    limit: int,
    candidate_limit: int | None,
    filter_values: tuple[tuple[str, str, str], ...],
    fields: tuple[str, ...],
    vector_json: str | None,
    output_format: str,
) -> None:
    """Query a published retrieval index without provider-specific request shapes."""
    project_dir: Path = ctx.obj["project_dir"]
    profiles_dir = ctx.obj["profiles_dir"]
    target = ctx.obj["target"]
    try:
        vector = _parse_search_vector(vector_json)
        filters = tuple(
            _parse_search_filter(field, operator, value)
            for field, operator, value in filter_values
        )
        request = SearchRequest(
            model=model_name,
            query=query,
            vector=vector,
            mode=SearchMode(mode),
            limit=limit,
            candidate_limit=candidate_limit,
            filters=filters,
            fields=fields,
        )
        results = run_search(
            project_dir,
            request,
            target=target,
            profiles_dir=profiles_dir,
        )
    except (*_CONFIG_ERRORS, ValueError) as error:
        raise ConfigClickError(str(error)) from error
    except SearchError as error:
        raise click.ClickException(str(error)) from error

    if output_format == "json":
        click.echo(json.dumps([result.to_dict() for result in results], indent=2))
        return
    _echo_search_table(results)


def _parse_search_vector(value: str | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    try:
        decoded = json.loads(value)
        if (
            not isinstance(decoded, list)
            or not decoded
            or any(isinstance(item, bool) or not isinstance(item, int | float) for item in decoded)
        ):
            raise ValueError
        return tuple(float(item) for item in decoded)
    except (TypeError, ValueError, json.JSONDecodeError):
        raise ValueError("--vector must be a non-empty JSON array of numbers") from None


_JSON_ARRAY_FILTER_OPERATORS = frozenset(
    {SearchFilterOperator.IN, SearchFilterOperator.ARRAY_CONTAINS_ANY}
)


def _parse_search_filter(field: str, operator: str, value: str) -> SearchFilter:
    resolved_operator = SearchFilterOperator(operator)
    if resolved_operator not in _JSON_ARRAY_FILTER_OPERATORS:
        return SearchFilter(field, resolved_operator, value)
    name = resolved_operator.value
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        raise ValueError(f"Search filter '{name}' values must be a JSON array") from None
    if not isinstance(decoded, list) or not decoded:
        raise ValueError(
            f"Search filter '{name}' values must be a non-empty JSON array"
        )
    return SearchFilter(field, resolved_operator, tuple(decoded))


def _echo_search_table(results: list[SearchResult]) -> None:
    if not results:
        click.echo("No results.")
        return
    provenance = results[0].provenance
    click.echo(
        f"Index: {provenance.unique_id}  Target: {provenance.target}  "
        f"Store: {provenance.store_type}  Collection: {provenance.logical_collection}"
    )
    dynamic_fields: list[str] = []
    for result in results:
        for values in (result.text, result.metadata, result.display):
            for field in values:
                if field not in dynamic_fields:
                    dynamic_fields.append(field)
    headers = ["rank", "score", "record_id", "document_id", "chunk_id", *dynamic_fields]
    rows: list[list[str]] = []
    for result in results:
        values = {**result.text, **result.metadata, **result.display}
        rows.append(
            [
                str(result.rank),
                f"{result.score:.6f}",
                result.record_id,
                result.document_id or "",
                result.chunk_id or "",
                *[_display_search_value(values.get(field)) for field in dynamic_fields],
            ]
        )
    widths = [
        min(80, max(len(headers[index]), *(len(row[index]) for row in rows)))
        for index in range(len(headers))
    ]
    click.echo("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    click.echo("  ".join("-" * width for width in widths))
    for row in rows:
        click.echo(
            "  ".join(
                value[: widths[index]].ljust(widths[index])
                for index, value in enumerate(row)
            )
        )


def _display_search_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, default=str, sort_keys=True)
    return str(value).replace("\n", " ")


def _safe_console_text(text: str, stream: object | None = None) -> str:
    target = stream or click.get_text_stream("stdout")
    encoding = getattr(target, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="replace").decode(
        encoding, errors="replace"
    )


@cli.command()
@click.option("--count", default=20, show_default=True, help="Number of documents to generate.")
@click.option("--seed", default=42, show_default=True, help="Random seed for deterministic output.")
@click.option(
    "--source",
    "source_name",
    default=None,
    help="Source name to seed (required if the project has multiple sources).",
)
@click.option(
    "--type",
    "data_type",
    type=click.Choice(sorted(_SEEDERS_BY_TYPE), case_sensitive=False),
    default=None,
    help="Synthetic data shape. Defaults based on the source's backend.",
)
@_project_context_options
@click.pass_context
def seed(
    ctx: click.Context,
    count: int,
    seed: int,
    source_name: str | None,
    data_type: str | None,
) -> None:
    """Generate synthetic documents into the source's data path.

    If --type is not given, the seeder is chosen by the backend of the model
    consuming the source: json → invoices, markdown → posts, pdf → invoice_pdfs,
    html → product_pages, llm → invoice_texts.
    """
    project_dir: Path = ctx.obj["project_dir"]
    profiles_dir = ctx.obj["profiles_dir"]
    target = ctx.obj["target"]
    project, sources, models = _load(project_dir)
    try:
        resolved = resolve_profile(
            project, project_dir, target=target, profiles_dir=profiles_dir
        )
        sources = apply_source_path_overrides(sources, resolved)
    except ProfileError as e:
        raise ConfigClickError(str(e)) from e
    source = _pick_source(sources, source_name)
    if _is_remote_source_path(source.path):
        raise ConfigClickError(
            f"`stel seed` only supports local source paths; source "
            f"'{source.name}' points to remote path '{source.path}'. "
            "Use a local target source_paths override, or seed remote storage "
            "outside stel."
        )

    if data_type:
        seeder = _SEEDERS_BY_TYPE[data_type]
        label = data_type
    else:
        backend_name = _backend_for_source(source, models)
        backend_seeder = _SEEDERS_BY_BACKEND.get(backend_name)
        if backend_seeder is None:
            raise click.ClickException(
                f"No default seeder for backend '{backend_name}'. "
                f"Pass --type explicitly. Available: {sorted(_SEEDERS_BY_TYPE)}"
            )
        seeder = backend_seeder
        label = backend_name

    try:
        output_dir = resolve_within_project(
            source.path,
            project_dir,
            surface=f"Source '{source.name}' path",
            external=source.external,
            hint="Set `external: true` on the source to allow it.",
        )
    except ConfigError as e:
        raise ConfigClickError(str(e)) from e
    try:
        paths = seeder(count, output_dir, seed)
    except OptionalDependencyError as e:
        raise ConfigClickError(str(e)) from e
    click.echo(f"Wrote {len(paths)} {label} documents to {output_dir}")


@cli.command()
@_project_context_options
@click.pass_context
def graph(ctx: click.Context) -> None:
    """Print a Mermaid diagram of the project DAG."""
    project_dir: Path = ctx.obj["project_dir"]
    _, sources, models = _load(project_dir)
    dag = _build_dag(sources, models)
    click.echo(dag.to_mermaid())


class _ResourceListRow(TypedDict):
    name: str
    resource_type: str
    kind: str
    tags: list[str]


@cli.command(name="ls")
@click.option("--select", "select", default=None, help="Selector expression for models.")
@click.option("--exclude", default=None, help="Selector expression for models to skip.")
@click.option(
    "--resource-type",
    type=click.Choice(
        ["model", "search_index", "source", "all"], case_sensitive=False
    ),
    default="model",
    show_default=True,
    help="Which resources to list. Selectors apply to models.",
)
@click.option(
    "--output",
    type=click.Choice(["name", "json"], case_sensitive=False),
    default="name",
    show_default=True,
    help="Output format.",
)
@_project_context_options
@click.pass_context
def ls(
    ctx: click.Context,
    select: str | None,
    exclude: str | None,
    resource_type: str,
    output: str,
) -> None:
    """List project resources (models/sources) matching a selector."""
    project_dir: Path = ctx.obj["project_dir"]
    _, sources, models = _load(project_dir)
    dag = _build_dag(sources, models)
    models_by_name = {m.name: m for m in models}

    rows: list[_ResourceListRow] = []
    if resource_type in ("model", "search_index", "all"):
        try:
            selected = dag.select_models(select=select, exclude=exclude)
        except SelectionError as e:
            raise click.ClickException(str(e)) from e
        for name in selected:
            model = models_by_name[name]
            is_search = model.search is not None
            if resource_type == "model" and is_search:
                continue
            if resource_type == "search_index" and not is_search:
                continue
            rows.append(
                {
                    "name": name,
                    "resource_type": "search_index" if is_search else "model",
                    "kind": _model_kind(model),
                    "tags": sorted(model.tags),
                }
            )
    if resource_type in ("source", "all"):
        for s in sources:
            rows.append(
                {
                    "name": s.name,
                    "resource_type": "source",
                    "kind": "source",
                    "tags": sorted(s.tags),
                }
            )

    if not rows:
        click.echo("No resources matched.")
        return

    if output == "json":
        click.echo(json.dumps(rows, indent=2))
        return
    for row in rows:
        tags = ",".join(row["tags"]) if row["tags"] else "-"
        click.echo(f"{row['name']:<24}{row['resource_type']:<10}{row['kind']:<12}{tags}")


def _model_kind(model: ModelConfig) -> str:
    if model.extraction is not None:
        return "extraction"
    if model.ml is not None:
        return "ml"
    if model.transform is not None:
        return "transform"
    if model.chunk is not None:
        return "chunk"
    if model.embed is not None:
        return "embed"
    if model.llm is not None:
        return "llm"
    if model.search is not None:
        return "search"
    if model.eval is not None:
        return "eval"
    return "unknown"


@cli.command()
@click.option(
    "--full-refresh", is_flag=True, help="Ignore incremental state and reprocess everything."
)
@click.option(
    "--select",
    "select",
    default=None,
    help="Selector expression (e.g. 'raw_invoices+', '+invoice_summary', '+name+').",
)
@click.option(
    "--exclude", default=None, help="Selector expression for nodes to exclude."
)
@click.option(
    "--watch",
    is_flag=True,
    help="Watch source paths and re-run on file changes (Ctrl-C to stop).",
)
@click.option(
    "--threads",
    type=int,
    default=1,
    show_default=True,
    help="Parallel worker threads per extraction model.",
)
@click.option(
    "--state",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Previous manifest.json (or its directory) for state:modified selection.",
)
@click.option(
    "--source-filter",
    "source_filter",
    multiple=True,
    metavar="GLOB",
    help="Process only source documents whose relative path matches this glob "
    "(repeatable, e.g. --source-filter 'AAPL/*'). Additive/upsert-only: a "
    "filtered run never deletes and cannot be combined with --full-refresh.",
)
@click.option(
    "--read-filter",
    "read_filter",
    multiple=True,
    nargs=3,
    metavar="FIELD OP VALUE",
    help="Narrow what transform parent reads and embed source reads see, as a "
    "typed predicate pushed down to the warehouse (repeatable; ops: eq, ne, "
    "lt, le, gt, ge, in -- 'in' takes a JSON array). Additive/upsert-only, "
    "like --source-filter: a filtered run never deletes and cannot be "
    "combined with --full-refresh.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Print the run_results.json payload to stdout instead of the table.",
)
@_verbose_option
@_project_context_options
@click.pass_context
def run(
    ctx: click.Context,
    full_refresh: bool,
    select: str | None,
    exclude: str | None,
    watch: bool,
    threads: int,
    state: Path | None,
    source_filter: tuple[str, ...],
    read_filter: tuple[tuple[str, str, str], ...],
    json_output: bool,
    verbose: int,
) -> None:
    """Extract and materialize selected models into the configured warehouse."""
    project_dir: Path = ctx.obj["project_dir"]
    profiles_dir = ctx.obj["profiles_dir"]
    target = ctx.obj["target"]
    # `run --threads N` executes independent models concurrently, each with its
    # own progress bar on one stderr — fall back to interleave-safe log lines.
    _configure_output(verbose, bars_safe=threads <= 1, json_output=json_output)

    if watch:
        _run_watch(
            project_dir,
            profiles_dir=profiles_dir,
            target=target,
            full_refresh=full_refresh,
            select=select,
            exclude=exclude,
            threads=threads,
            source_filter=source_filter,
            read_filter=read_filter,
        )
        return

    start = time.monotonic()
    try:
        results = run_project(
            project_dir,
            full_refresh=full_refresh,
            select=select,
            exclude=exclude,
            target=target,
            profiles_dir=profiles_dir,
            threads=threads,
            state=state,
            source_filter=source_filter,
            read_filter=read_filter,
        )
    except _CONFIG_ERRORS as e:
        raise ConfigClickError(str(e)) from e
    except RunError as e:
        raise click.ClickException(str(e)) from e
    elapsed = round(time.monotonic() - start, 3)

    write_manifest(project_dir, target=target, profiles_dir=profiles_dir)
    results_path = write_run_results(
        project_dir,
        results,
        target=target,
        profiles_dir=profiles_dir,
        invocation="run",
        elapsed_seconds=elapsed,
    )

    if json_output:
        click.echo(results_path.read_text(encoding="utf-8"))
        if any(r.errors for r in results):
            ctx.exit(1)
        return

    if not results:
        click.echo("No models selected.")
        return

    header = (
        f"{'model':<22}{'kind':<12}{'mater.':<14}"
        f"{'processed':>10}{'skipped':>10}{'deleted':>9}{'rows':>8}{'time(s)':>10}"
    )
    click.echo(header)
    click.echo("-" * len(header))
    for r in results:
        click.echo(
            f"{r.model_name:<22}{r.kind:<12}{r.materialization:<14}"
            f"{r.documents_processed:>10}{r.documents_skipped:>10}"
            f"{r.documents_deleted:>9}{r.rows_written:>8}{r.duration_seconds:>10.3f}"
        )
        if r.status is not None:
            click.echo(f"  STATUS: {r.status}", err=True)
        for err in r.errors:
            click.echo(f"  ERROR: {err}", err=True)
        _echo_warnings(r)

    usage_lines = [
        f"{r.model_name:<22}"
        f"{_usage_summary(r.metrics, provider=r.provider, model=r.provider_model)}"
        for r in results
        if "api_calls" in r.metrics or "provider_calls" in r.metrics
    ]
    if usage_lines:
        click.echo("")
        for line in usage_lines:
            click.echo(line)

    if any(r.errors for r in results):
        ctx.exit(1)


_MAX_WARNING_LINES = 5


def _echo_warnings(r: ModelRunResult) -> None:
    """Backend warnings under the model's summary row, capped so a corpus-wide
    papercut (one warning per document) can't flood the terminal. The full set
    is always in run_results.json. Warnings never change the exit code."""
    shown = list(r.warnings.items())[:_MAX_WARNING_LINES]
    for message, count in shown:
        suffix = f" ({count} documents)" if count > 1 else ""
        click.echo(f"  WARNING: {message}{suffix}", err=True)
    hidden = len(r.warnings) - len(shown)
    if hidden > 0:
        click.echo(
            f"  ... {hidden} more distinct warnings (see run_results.json)", err=True
        )


def _usage_summary(
    m: dict[str, object],
    *,
    provider: str | None = None,
    model: str | None = None,
) -> str:
    """One-line provider usage: calls, cache hits, tokens, and optional cost."""
    if "api_calls" in m:
        # Native `llm:` models and `backend: llm` extraction report api_calls;
        # embed models report only provider_calls.
        parts = [
            f"llm: {m.get('api_calls', 0)} calls, "
            f"{m.get('cache_hits', 0)} cache hits"
        ]
    elif "provider_calls" in m:
        parts = [
            f"embedding: {m.get('provider_calls', 0)} calls, "
            f"{m.get('cache_hits', 0)} cache hits"
        ]
    else:
        parts = [
            f"llm: {m.get('api_calls', 0)} calls, "
            f"{m.get('cache_hits', 0)} cache hits"
        ]
    if provider is not None:
        identity = f"provider={provider}"
        if model is not None:
            identity += f" model={model}"
        parts.append(identity)
    tokens_in = m.get("input_tokens", 0)
    tokens_out = m.get("output_tokens", 0)
    if tokens_in or tokens_out:
        parts.append(f"{tokens_in:,} in / {tokens_out:,} out tokens")
    if (reported_cost := m.get("reported_cost_usd")) is not None:
        parts.append(f"${reported_cost:.4f} reported")
    if (estimated_cost := m.get("estimated_cost_usd")) is not None:
        parts.append(f"~${estimated_cost:.4f} estimated")
    return "  ".join(parts)


@cli.command()
@click.option(
    "--full-refresh", is_flag=True, help="Ignore incremental state and reprocess everything."
)
@click.option("--select", "select", default=None, help="Selector expression.")
@click.option("--exclude", default=None, help="Selector expression for nodes to exclude.")
@click.option(
    "--threads",
    type=int,
    default=1,
    show_default=True,
    help="Parallel worker threads per extraction model.",
)
@click.option(
    "--store-failures",
    is_flag=True,
    help="Persist failing test rows to stel_test_failures__* tables.",
)
@click.option(
    "--state",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Previous manifest.json (or its directory) for state:modified selection.",
)
@click.option(
    "--source-filter",
    "source_filter",
    multiple=True,
    metavar="GLOB",
    help="Process only source documents whose relative path matches this glob "
    "(repeatable, e.g. --source-filter 'AAPL/*'). Additive/upsert-only: a "
    "filtered run never deletes and cannot be combined with --full-refresh.",
)
@click.option(
    "--read-filter",
    "read_filter",
    multiple=True,
    nargs=3,
    metavar="FIELD OP VALUE",
    help="Narrow what transform parent reads and embed source reads see, as a "
    "typed predicate pushed down to the warehouse (repeatable; ops: eq, ne, "
    "lt, le, gt, ge, in -- 'in' takes a JSON array). Additive/upsert-only, "
    "like --source-filter: a filtered run never deletes and cannot be "
    "combined with --full-refresh.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Print the run_results.json payload to stdout instead of the table.",
)
@_verbose_option
@_project_context_options
@click.pass_context
def build(
    ctx: click.Context,
    full_refresh: bool,
    select: str | None,
    exclude: str | None,
    threads: int,
    store_failures: bool,
    state: Path | None,
    source_filter: tuple[str, ...],
    read_filter: tuple[tuple[str, str, str], ...],
    json_output: bool,
    verbose: int,
) -> None:
    """Run and test each model in dependency order; downstream models are skipped
    when an upstream model errors or fails a test."""
    project_dir: Path = ctx.obj["project_dir"]
    profiles_dir = ctx.obj["profiles_dir"]
    target = ctx.obj["target"]
    _configure_output(verbose, json_output=json_output)
    start = time.monotonic()
    try:
        result = build_project(
            project_dir,
            full_refresh=full_refresh,
            select=select,
            exclude=exclude,
            target=target,
            profiles_dir=profiles_dir,
            threads=threads,
            store_failures=store_failures,
            state=state,
            source_filter=source_filter,
            read_filter=read_filter,
        )
    except _CONFIG_ERRORS as e:
        raise ConfigClickError(str(e)) from e
    except RunError as e:
        raise click.ClickException(str(e)) from e
    elapsed = round(time.monotonic() - start, 3)

    # Hard test failures don't populate ModelRunResult.errors, so feed them in
    # explicitly or a failing test on a leaf model would report success.
    test_failures: dict[str, list[str]] = {}
    for t in result.test_results:
        if t.is_hard_failure:
            test_failures.setdefault(t.model_name, []).append(_test_failure_label(t))

    write_manifest(project_dir, target=target, profiles_dir=profiles_dir)
    results_path = write_run_results(
        project_dir,
        result.run_results,
        target=target,
        profiles_dir=profiles_dir,
        invocation="build",
        skipped=result.skipped,
        elapsed_seconds=elapsed,
        test_failures=test_failures,
    )

    failed_tests = sum(1 for t in result.test_results if t.status == "fail")
    errored_models = sum(1 for r in result.run_results if r.errors)

    if json_output:
        click.echo(results_path.read_text(encoding="utf-8"))
    else:
        _echo_build(result)

    if failed_tests or errored_models or result.skipped:
        ctx.exit(1)


def _echo_build(result: BuildResult) -> None:
    if not result.run_results and not result.skipped:
        click.echo("No models selected.")
        return

    rheader = f"{'model':<22}{'kind':<12}{'rows':>8}{'time(s)':>10}  status"
    click.echo(rheader)
    click.echo("-" * len(rheader))
    for r in result.run_results:
        status = r.status.upper() if r.status else ("ERROR" if r.errors else "ok")
        click.echo(
            f"{r.model_name:<22}{r.kind:<12}{r.rows_written:>8}"
            f"{r.duration_seconds:>10.3f}  {status}"
        )
        for err in r.errors:
            click.echo(f"  ERROR: {err}", err=True)
        _echo_warnings(r)
    for name in result.skipped:
        click.echo(f"{name:<22}{'-':<12}{'-':>8}{'-':>10}  SKIPPED (upstream failed)")

    if result.test_results:
        click.echo("")
        theader = f"{'model':<22}{'test':<14}{'column':<22}{'status':<8}{'message'}"
        click.echo(theader)
        click.echo("-" * 90)
        for t in result.test_results:
            click.echo(
                f"{t.model_name:<22}{t.test_name:<14}{(t.column or ''):<22}"
                f"{t.status:<8}{_test_message(t)}"
            )


def _test_message(t: TestResult) -> str:
    if t.failures_table:
        return f"{t.message} [stored {t.failure_count} rows in {t.failures_table}]"
    return t.message


def _test_failure_label(t: TestResult) -> str:
    """Compact identifier for a failed test, for the run_results payload."""
    name = f"{t.test_name}({t.column})" if t.column else t.test_name
    return f"{name}: {t.message}" if t.message else name


@cli.command()
@click.option("--select", "select", default=None, help="Selector expression for models to test.")
@click.option("--exclude", default=None, help="Selector expression for models to skip.")
@click.option(
    "--store-failures",
    is_flag=True,
    help="Persist failing test rows to stel_test_failures__* tables.",
)
@click.option(
    "--state",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Previous manifest.json (or its directory) for state:modified selection.",
)
@_project_context_options
@click.pass_context
def test(
    ctx: click.Context,
    select: str | None,
    exclude: str | None,
    store_failures: bool,
    state: Path | None,
) -> None:
    """Run schema tests against materialized tables."""
    project_dir: Path = ctx.obj["project_dir"]
    profiles_dir = ctx.obj["profiles_dir"]
    target = ctx.obj["target"]
    try:
        results = run_project_tests(
            project_dir,
            select=select,
            exclude=exclude,
            target=target,
            profiles_dir=profiles_dir,
            store_failures=store_failures,
            state=state,
        )
    except _CONFIG_ERRORS as e:
        raise ConfigClickError(str(e)) from e

    if not results:
        click.echo("No tests defined.")
        return

    passed = sum(1 for r in results if r.status == "pass")
    warned = sum(1 for r in results if r.status == "warn")
    failed = sum(1 for r in results if r.status == "fail")
    header = f"{'model':<22}{'test':<14}{'column':<22}{'status':<8}{'message'}"
    click.echo(header)
    click.echo("-" * 90)
    for r in results:
        click.echo(
            f"{r.model_name:<22}{r.test_name:<14}{(r.column or ''):<22}"
            f"{r.status:<8}{_test_message(r)}"
        )
    click.echo("-" * 90)
    summary = f"{passed} passed"
    if warned:
        summary += f", {warned} warned"
    summary += f", {failed} failed (of {len(results)})"
    click.echo(summary)
    if failed:
        ctx.exit(1)


@cli.command("eval")
@click.option(
    "--select", "select", default=None, help="Selector expression for search models to evaluate."
)
@click.option("--exclude", default=None, help="Selector expression for models to skip.")
@click.option("--json", "as_json", is_flag=True, help="Print the eval artifact to stdout.")
@_verbose_option
@_project_context_options
@click.pass_context
def eval_(
    ctx: click.Context,
    select: str | None,
    exclude: str | None,
    as_json: bool,
    verbose: int,
) -> None:
    """Run golden-set retrieval evaluations declared as `retrieval_tests:` on
    search models (issue #137): recall/precision/hit-rate/MRR/NDCG@k against
    labeled queries, plus hard policy-filter assertions. Writes
    target-path/retrieval_eval.json."""
    project_dir: Path = ctx.obj["project_dir"]
    profiles_dir = ctx.obj["profiles_dir"]
    target = ctx.obj["target"]
    _configure_output(verbose, json_output=as_json)
    try:
        results = run_retrieval_evaluation(
            project_dir,
            select=select,
            exclude=exclude,
            target=target,
            profiles_dir=profiles_dir,
        )
    except _CONFIG_ERRORS as e:
        raise ConfigClickError(str(e)) from e
    except RetrievalEvalError as e:
        raise click.ClickException(str(e)) from e

    project, _sources, _models = _load(project_dir)
    artifact_path = write_retrieval_eval_artifact(project_dir, project, results)
    failed = sum(1 for r in results if r.status == "fail")

    if not results:
        click.echo("No retrieval_tests defined.")
        return

    if as_json:
        click.echo(artifact_path.read_text(encoding="utf-8"))
        if failed:
            ctx.exit(1)
        return

    header = f"{'model':<22}{'test':<26}{'status':<8}{'thresholds'}"
    click.echo(header)
    click.echo("-" * 90)
    for r in results:
        threshold_summary = ", ".join(
            f"{t.key}={t.actual:.3f}(>={t.min:.3f}){'*' if t.status != 'pass' else ''}"
            for t in r.thresholds
        )
        click.echo(f"{r.model_name:<22}{r.test_name:<26}{r.status:<8}{threshold_summary}")
        for violation in r.policy_violations:
            click.echo(
                f"    POLICY VIOLATION [{violation.query_id}] {violation.kind}: "
                f"{list(violation.ids)}"
            )
    click.echo("-" * 90)
    passed = sum(1 for r in results if r.status == "pass")
    warned = sum(1 for r in results if r.status == "warn")
    summary = f"{passed} passed"
    if warned:
        summary += f", {warned} warned"
    summary += f", {failed} failed (of {len(results)})"
    click.echo(summary)
    click.echo(f"Wrote {artifact_path}")
    if failed:
        ctx.exit(1)


@cli.command("emit-dbt-sources")
@click.option(
    "--source-name",
    default=None,
    help="dbt source name (default: dbt_ml_<project-name>).",
)
@click.option("--select", "select", default=None, help="Selector expression.")
@click.option("--exclude", default=None, help="Selector expression for nodes to skip.")
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output file (default: <target-path>/sources.yml).",
)
@click.option(
    "--dagster-meta",
    is_flag=True,
    help="Stamp meta.dagster.asset_key on each table so the emitted sources map "
    "cleanly onto dagster-dbt assets. Ignored by pure dbt.",
)
@_project_context_options
@click.pass_context
def emit_dbt_sources(
    ctx: click.Context,
    source_name: str | None,
    select: str | None,
    exclude: str | None,
    output: Path | None,
    dagster_meta: bool,
) -> None:
    """Write a dbt-compatible sources.yml declaring stel's materialized tables.

    Drop the output into a dbt project using the matching warehouse adapter so
    models can refer to stel tables via `{{ source(...) }}`. With
    --dagster-meta, each table also carries a Dagster asset key for the
    dagster-dbt integration.
    """
    project_dir: Path = ctx.obj["project_dir"]
    profiles_dir = ctx.obj["profiles_dir"]
    target = ctx.obj["target"]
    try:
        path = write_dbt_sources(
            project_dir,
            source_name=source_name,
            select=select,
            exclude=exclude,
            output=output,
            target=target,
            profiles_dir=profiles_dir,
            dagster_meta=dagster_meta,
        )
    except _CONFIG_ERRORS as e:
        raise ConfigClickError(str(e)) from e
    click.echo(f"Wrote {path}")


@cli.command("codegen")
@click.option(
    "--output",
    "output",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory in the dbt project to write the shims + schema.yml into "
    "(e.g. <dbt-project>/models/stel).",
)
@click.option(
    "--source-name",
    default=None,
    help="Name used in the generated schema.yml (default: dbt_ml_<project-name>).",
)
@_project_context_options
@click.pass_context
def codegen(
    ctx: click.Context,
    output: Path,
    source_name: str | None,
) -> None:
    """Generate dbt Python-model shims + schema.yml from this stel project (#177).

    Emits one dbt Python model per extraction/transform model — each calls
    stel in-process via `stel.dbt_embed.materialize` — plus a schema.yml
    carrying the models' fields and tests. Point --output at a directory under
    your dbt project's model-paths so `dbt build` runs stel as native dbt
    nodes. Set STEL_PROJECT_DIR to this project when running dbt.
    """
    from .dbt_embed.codegen import generate_dbt_models

    project_dir: Path = ctx.obj["project_dir"]
    try:
        written = generate_dbt_models(project_dir, output, source_name=source_name)
    except _CONFIG_ERRORS as e:
        raise ConfigClickError(str(e)) from e
    for path in written:
        click.echo(f"Wrote {path}")


@cli.group()
@_project_context_options
def mcp() -> None:
    """Serve governed document context to MCP clients."""


@mcp.command("serve")
@click.option(
    "--timeout-seconds",
    # Bounded here as well as in the settings model so an operator raising the
    # deadline learns the ceiling from the CLI, instead of from a validation
    # error after the server has been told to start (issue #461).
    type=click.FloatRange(min=0.1, max=MAX_CONTEXT_TIMEOUT_SECONDS),
    default=DEFAULT_CONTEXT_TIMEOUT_SECONDS,
    show_default=True,
)
@click.option(
    "--max-concurrency",
    type=click.IntRange(1, 64),
    default=4,
    show_default=True,
)
@click.option(
    "--max-requests-per-minute",
    type=click.IntRange(1, 100_000),
    default=120,
    show_default=True,
)
@click.option(
    "--max-requests-per-minute-per-principal",
    type=click.IntRange(1, 100_000),
    default=None,
    help=(
        "Each caller's share of --max-requests-per-minute, keyed by subject. "
        "Unset, the global cap is all there is and one caller can exhaust it "
        "for everyone. Requests with no resolvable principal share one "
        "anonymous bucket of this size."
    ),
)
@click.option(
    "--max-response-bytes",
    type=click.IntRange(1024, 10_000_000),
    default=256_000,
    show_default=True,
)
@click.option(
    "--max-scan-rows",
    type=click.IntRange(1, 1_000_000),
    default=10_000,
    show_default=True,
)
@click.option(
    "--transport",
    type=click.Choice(["stdio", "streamable-http", "sse"]),
    default="stdio",
    show_default=True,
    help="stdio for a local client; the others serve many callers over HTTP.",
)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", type=click.IntRange(1, 65535), default=8000, show_default=True)
@click.option(
    "--jwt-issuer",
    default=None,
    help=(
        "Verify each caller's bearer token was issued by this issuer. With "
        "--jwt-audience and --jwt-jwks-uri, identity comes from the token this "
        "server verified rather than from a header it was told to trust."
    ),
)
@click.option(
    "--jwt-audience",
    default=None,
    help=(
        "The audience every accepted token must name — this deployment. "
        "Required with --jwt-issuer: without it, a token a caller legitimately "
        "holds for another service is a valid token here."
    ),
)
@click.option(
    "--jwt-jwks-uri",
    default=None,
    help="HTTPS URL of the issuer's JWKS, used to verify token signatures.",
)
@click.option(
    "--introspection-endpoint",
    default=None,
    help=(
        "HTTPS URL of the issuer's RFC 7662 introspection endpoint. Use this "
        "instead of the --jwt-* flags when the authorization server issues "
        "opaque tokens that cannot be verified locally. Costs a round trip "
        "per token, cached briefly."
    ),
)
@click.option(
    "--introspection-issuer",
    default=None,
    help=(
        "The issuer every introspected token must name, when the response "
        "carries one."
    ),
)
@click.option(
    "--introspection-audience",
    default=None,
    help=(
        "The audience every introspected token must name — this deployment. "
        "An `active` token is only valid at the issuer, not necessarily for "
        "you, so an authorization server that omits `aud` cannot be used."
    ),
)
@click.option(
    "--introspection-client-id",
    default=None,
    help="Client id this server authenticates to the introspection endpoint with.",
)
@click.option(
    "--introspection-client-secret-env",
    default=None,
    help=(
        "Name of the environment variable holding the introspection client "
        "secret. The name, not the secret: a value passed on the command line "
        "is visible in the process list and the shell history."
    ),
)
@click.option(
    "--trust-proxy-principal-headers",
    is_flag=True,
    help=(
        "Take each caller's identity from X-Stel-Principal-Id and friends. "
        "ONLY safe when a proxy in front authenticates the caller and "
        "OVERWRITES those headers — reachable directly, any caller can claim "
        "any tenant. Required for a network transport until token "
        "verification lands."
    ),
)
@click.option(
    "--grants-relation",
    type=str,
    default=None,
    help=(
        "Warehouse relation of (subject_id, attribute, value) grants. When "
        "set, authorization is looked up by authenticated subject instead of "
        "being read from caller-supplied claim headers."
    ),
)
@click.option(
    "--grant-ttl-seconds",
    type=click.FloatRange(min=1.0),
    default=None,
    show_default="60",
    help=(
        "How long a subject's grants are cached. This is the ceiling on how "
        "long a revoked grant keeps applying."
    ),
)
@_project_context_options
@click.pass_context
def mcp_serve(
    ctx: click.Context,
    transport: str,
    host: str,
    port: int,
    jwt_issuer: str | None,
    jwt_audience: str | None,
    jwt_jwks_uri: str | None,
    introspection_endpoint: str | None,
    introspection_issuer: str | None,
    introspection_audience: str | None,
    introspection_client_id: str | None,
    introspection_client_secret_env: str | None,
    trust_proxy_principal_headers: bool,
    timeout_seconds: float,
    max_concurrency: int,
    max_requests_per_minute: int,
    max_requests_per_minute_per_principal: int | None,
    max_response_bytes: int,
    max_scan_rows: int,
    grants_relation: str | None,
    grant_ttl_seconds: float | None,
) -> None:
    """Run the read-only stel MCP server.

    Defaults to stdio, where the operator running the process is the
    principal. A network transport serves many callers, so it needs an
    identity per request and refuses to start without one.
    """
    from pydantic import ValidationError

    from .mcp_server.authorization import (
        AuthorizationError,
        TrustedHeaderPrincipalResolver,
    )
    from .mcp_server.catalog import ArtifactCatalogError
    from .mcp_server.server import serve_network, serve_stdio
    from .mcp_server.service import ContextServerSettings
    from .mcp_server.tokens import TokenVerificationError

    jwt_settings = (jwt_issuer, jwt_audience, jwt_jwks_uri)
    introspection_settings = (
        introspection_endpoint,
        introspection_issuer,
        introspection_audience,
        introspection_client_id,
        introspection_client_secret_env,
    )
    verifying_jwt = any(jwt_settings)
    introspecting = any(introspection_settings)
    verifying_tokens = verifying_jwt or introspecting
    try:
        # Constructed inside the try: the per-principal-cap-under-global-cap
        # cross-field check raises pydantic's ValidationError, which is not
        # one of _CONFIG_ERRORS and was escaping as a bare traceback with exit
        # code 1 instead of the usual exit-2 configuration diagnostic (Codex
        # review, #466).
        settings = ContextServerSettings(
            timeout_seconds=timeout_seconds,
            max_concurrency=max_concurrency,
            max_requests_per_minute=max_requests_per_minute,
            max_requests_per_minute_per_principal=max_requests_per_minute_per_principal,
            max_response_bytes=max_response_bytes,
            max_scan_rows=max_scan_rows,
        )
        if transport == "stdio":
            if trust_proxy_principal_headers:
                raise ConfigClickError(
                    "--trust-proxy-principal-headers applies to a network "
                    "transport; stdio resolves its principal from the "
                    "operator's environment."
                )
            if verifying_tokens:
                # Refused rather than ignored: a flag that is accepted and
                # does nothing looks, from the outside, like authentication.
                raise ConfigClickError(
                    "Token verification (--jwt-* or --introspection-*) applies "
                    "to a network transport; stdio resolves its principal from "
                    "the operator's environment and would verify nothing."
                )
            serve_stdio(
                ctx.obj["project_dir"],
                target=ctx.obj["target"],
                profiles_dir=ctx.obj["profiles_dir"],
                grants_relation=grants_relation,
                grant_ttl_seconds=grant_ttl_seconds,
                settings=settings,
            )
            return
        if verifying_jwt and introspecting:
            raise ConfigClickError(
                "Choose one token verifier. --jwt-* verifies a JWT's signature "
                "locally against the issuer's published keys; "
                "--introspection-* asks the issuer about each token instead, "
                "which is what an opaque token needs. Configuring both leaves "
                "it ambiguous which one a rejected caller was refused by."
            )
        if verifying_jwt and not all(jwt_settings):
            missing = [
                name
                for name, value in (
                    ("--jwt-issuer", jwt_issuer),
                    ("--jwt-audience", jwt_audience),
                    ("--jwt-jwks-uri", jwt_jwks_uri),
                )
                if not value
            ]
            raise ConfigClickError(
                "JWT verification needs all three of --jwt-issuer, "
                f"--jwt-audience and --jwt-jwks-uri; missing {', '.join(missing)}. "
                "Each is a security boundary, so none of them gets a default."
            )
        if introspecting and not all(introspection_settings):
            missing = [
                name
                for name, value in (
                    ("--introspection-endpoint", introspection_endpoint),
                    ("--introspection-issuer", introspection_issuer),
                    ("--introspection-audience", introspection_audience),
                    ("--introspection-client-id", introspection_client_id),
                    (
                        "--introspection-client-secret-env",
                        introspection_client_secret_env,
                    ),
                )
                if not value
            ]
            raise ConfigClickError(
                "Token introspection needs all five of "
                "--introspection-endpoint, --introspection-issuer, "
                "--introspection-audience, --introspection-client-id and "
                f"--introspection-client-secret-env; missing {', '.join(missing)}. "
                "Each is a security boundary, so none of them gets a default."
            )
        if verifying_tokens and trust_proxy_principal_headers:
            raise ConfigClickError(
                "Choose one identity source. --trust-proxy-principal-headers "
                "believes whatever the proxy sets; JWT verification checks the "
                "caller's own token. Enabling both means the headers decide, "
                "and the token checking is decoration."
            )
        if not verifying_tokens and not trust_proxy_principal_headers:
            raise ConfigClickError(
                f"--transport {transport} serves many callers, so each request "
                "needs its own identity. No per-request principal source is "
                "configured, and the stdio default would serve every caller as "
                "one principal — with that principal's tenant filters. Pass "
                "--jwt-issuer/--jwt-audience/--jwt-jwks-uri to verify tokens "
                "locally, --introspection-* to verify them at the issuer, or "
                "--trust-proxy-principal-headers if an authenticating proxy "
                "sets them."
            )
        token_verifier = None
        if verifying_jwt:
            from .mcp_server.authorization import AccessTokenPrincipalResolver
            from .mcp_server.tokens import JwksTokenVerifier, JwtVerifierConfig

            token_verifier = JwksTokenVerifier(
                JwtVerifierConfig(
                    issuer=str(jwt_issuer),
                    audience=str(jwt_audience),
                    jwks_uri=str(jwt_jwks_uri),
                )
            )
            resolver: Any = AccessTokenPrincipalResolver()
            click.echo(
                f"Serving on {host}:{port} over {transport}. Caller identity "
                f"comes from bearer tokens verified against {jwt_issuer}.",
                err=True,
            )
        elif introspecting:
            from .mcp_server.authorization import AccessTokenPrincipalResolver
            from .mcp_server.introspection import (
                IntrospectionTokenVerifier,
                IntrospectionVerifierConfig,
            )

            # The variable's name is passed through, not its value: the secret
            # is read only where the request is built, so no long-lived object
            # here holds it. The config refuses at startup if it is unset.
            token_verifier = IntrospectionTokenVerifier(
                IntrospectionVerifierConfig(
                    issuer=str(introspection_issuer),
                    audience=str(introspection_audience),
                    introspection_endpoint=str(introspection_endpoint),
                    client_id=str(introspection_client_id),
                    client_secret_env=str(introspection_client_secret_env),
                )
            )
            resolver = AccessTokenPrincipalResolver()
            click.echo(
                f"Serving on {host}:{port} over {transport}. Caller identity "
                "comes from bearer tokens introspected at "
                f"{introspection_issuer}. A revoked token keeps working for up "
                "to the cache TTL.",
                err=True,
            )
        else:
            resolver = TrustedHeaderPrincipalResolver()
            click.echo(
                f"Serving on {host}:{port} over {transport}. Caller identity "
                "comes from proxy-set headers — verify the proxy authenticates "
                "and overwrites them.",
                err=True,
            )
        serve_network(
            ctx.obj["project_dir"],
            transport=transport,
            host=host,
            port=port,
            principal_resolver=resolver,
            token_verifier=token_verifier,
            target=ctx.obj["target"],
            profiles_dir=ctx.obj["profiles_dir"],
            grants_relation=grants_relation,
            grant_ttl_seconds=grant_ttl_seconds,
            settings=settings,
        )
    except ValidationError as error:
        raise ConfigClickError(str(error)) from error
    except (ArtifactCatalogError, AuthorizationError, *_CONFIG_ERRORS) as error:
        raise ConfigClickError(str(error)) from error
    except TokenVerificationError as error:
        # Raised at construction, before anything listens: a plaintext JWKS
        # URL or a blank issuer is a configuration mistake, not a runtime one.
        raise ConfigClickError(str(error)) from error


@cli.command("concept-cloud")
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("concept_cloud.html"),
    show_default=True,
    help="Where to write the self-contained HTML artifact.",
)
@click.option(
    "--placeholder",
    is_flag=True,
    help="Render the tiny built-in placeholder bundle.",
)
@click.option(
    "--demo",
    is_flag=True,
    help="Render the sizable built-in economic-data demo bundle (~45 entities).",
)
@click.option(
    "--linking-model",
    default=None,
    help="Entity-linking model whose canonical_id rows become the cloud.",
)
@click.option(
    "--relation-model",
    default=None,
    help="Relation model whose rows become concept-to-concept edges.",
)
@click.option(
    "--entity-model",
    default=None,
    help="NLP entity model used to enrich labels/display text.",
)
@click.option(
    "--dbt-manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Downstream dbt manifest.json to use as the DAG plane.",
)
@click.option(
    "--source-name",
    default=None,
    help=(
        "dbt source name the manifest declares, when it differs from the "
        "default (default: dbt_ml_<project-name>)."
    ),
)
@click.option(
    "--top-n",
    default=200,
    show_default=True,
    help="Cap the cloud to the N most frequent canonical concepts.",
)
@click.option(
    "--embed-model",
    default=None,
    help=(
        "Embed model over the linking mentions; bakes semantic 3D positions "
        "from mention-vector centroids (#345). Coordinates only — no vectors "
        "or text enter the bundle."
    ),
)
@click.option(
    "--with-query-log",
    is_flag=True,
    help=(
        "Derive a `retrieval` heat dimension (hot/warm/cold/never) from the "
        "MCP query log. Aggregate-only; absent log means no dimension."
    ),
)
@click.option(
    "--dimension",
    "dimensions",
    multiple=True,
    help=(
        "Categorical dimension from a concept-keyed column, as "
        "name=model.column (repeatable). Enum-field outputs fit directly."
    ),
)
@_verbose_option
@_project_context_options
@click.pass_context
def concept_cloud(
    ctx: click.Context,
    output: Path,
    placeholder: bool,
    demo: bool,
    linking_model: str | None,
    relation_model: str | None,
    entity_model: str | None,
    dbt_manifest: Path | None,
    source_name: str | None,
    top_n: int,
    embed_model: str | None,
    with_query_log: bool,
    dimensions: tuple[str, ...],
    verbose: int,
) -> None:
    """Render the self-contained 3D concept-cloud artifact (#255).

    Pass ``--linking-model`` to export a real project's concepts (optionally with
    ``--relation-model``/``--entity-model`` and a downstream ``--dbt-manifest``),
    ``--demo`` for the sizable built-in example, or ``--placeholder`` for the
    minimal one.
    """
    _configure_output(verbose)
    if placeholder or demo:
        bundle = demo_export() if demo else placeholder_export()
        written = write_concept_cloud(bundle, output)
        click.echo(
            f"Wrote {'demo' if demo else 'placeholder'} concept-cloud artifact "
            f"({len(bundle.concepts)} concepts) to {written}"
        )
        return
    if not linking_model:
        raise click.ClickException(
            "Pass --linking-model <model> to export a project's concepts, "
            "or --demo / --placeholder for a built-in bundle."
        )
    dimension_specs: dict[str, str] = {}
    for raw in dimensions:
        name, _, spec = raw.partition("=")
        if not name or not spec:
            raise click.ClickException(
                f"--dimension must be name=model.column, got {raw!r}"
            )
        dimension_specs[name] = spec
    try:
        export = export_concept_cloud(
            ctx.obj["project_dir"],
            linking_model=linking_model,
            relation_model=relation_model,
            entity_model=entity_model,
            dbt_manifest=dbt_manifest,
            source_name=source_name,
            target=ctx.obj["target"],
            profiles_dir=ctx.obj["profiles_dir"],
            top_n=top_n,
            embed_model=embed_model,
            with_query_log=with_query_log,
            dimension_specs=dimension_specs or None,
        )
    except (ConceptCloudExportError, AdapterError, *_CONFIG_ERRORS) as e:
        raise ConfigClickError(str(e)) from e
    written = write_concept_cloud(export, output)
    click.echo(
        f"Wrote concept-cloud artifact ({len(export.concepts)} concepts) to {written}"
    )


@cli.group()
@_project_context_options
def docs() -> None:
    """Generate or serve a static docs site for the project."""


@docs.command("generate")
@click.option(
    "--output",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Output dir (default: <target-path>/docs).",
)
@_project_context_options
@click.pass_context
def docs_generate(ctx: click.Context, output: Path | None) -> None:
    """Render target/docs/*.html driven by manifest.json + run_results.json."""
    project_dir: Path = ctx.obj["project_dir"]
    profiles_dir = ctx.obj["profiles_dir"]
    target = ctx.obj["target"]
    try:
        result = generate_docs(
            project_dir,
            target=target,
            profiles_dir=profiles_dir,
            output_dir=output,
        )
    except (ConfigError, ProfileError) as e:
        raise ConfigClickError(str(e)) from e
    except DocsError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"Wrote {result.pages_written} page(s) to {result.output_dir}")


@docs.command("serve")
@click.option("--port", default=8080, show_default=True, help="HTTP port.")
@_project_context_options
@click.pass_context
def docs_serve(ctx: click.Context, port: int) -> None:
    """Serve the generated docs over http.server. Ctrl-C to stop."""
    project_dir: Path = ctx.obj["project_dir"]
    try:
        serve_docs(project_dir, port=port)
    except ConfigError as e:
        raise ConfigClickError(str(e)) from e
    except DocsError as e:
        raise click.ClickException(str(e)) from e


@cli.group()
@_project_context_options
def source() -> None:
    """Inspect sources (freshness, etc.)."""


@source.command("freshness")
@_project_context_options
@click.pass_context
def source_freshness(ctx: click.Context) -> None:
    """Check source freshness against configured warn/error thresholds."""
    project_dir: Path = ctx.obj["project_dir"]
    profiles_dir = ctx.obj["profiles_dir"]
    target = ctx.obj["target"]
    try:
        results = check_freshness(
            project_dir, target=target, profiles_dir=profiles_dir
        )
    except (ConfigError, ProfileError) as e:
        raise ConfigClickError(str(e)) from e
    except SourceError as e:
        raise click.ClickException(str(e)) from e

    if not results:
        click.echo("No sources defined.")
        return

    header = f"{'source':<24}{'status':<10}{'files':>8}{'age':>10}  {'message'}"
    click.echo(header)
    click.echo("-" * 90)
    for r in results:
        age = "-" if r.newest_age_seconds is None else f"{r.newest_age_seconds:.0f}s"
        click.echo(
            f"{r.source_name:<24}{r.status:<10}{r.file_count:>8}{age:>10}  {r.message}"
        )
    click.echo("-" * 90)
    failed = sum(1 for r in results if r.status == "fail")
    warned = sum(1 for r in results if r.status == "warn")
    nodata = sum(1 for r in results if r.status == "no_data")
    passed = sum(1 for r in results if r.status == "pass")
    summary = f"{passed} pass"
    if warned:
        summary += f", {warned} warn"
    if failed:
        summary += f", {failed} fail"
    if nodata:
        summary += f", {nodata} no_data"
    click.echo(summary)
    if failed:
        ctx.exit(1)


@cli.group()
@_project_context_options
def serving() -> None:
    """Inspect and recover serving-readiness state for search indexes."""


@serving.command("status")
@click.argument("model_name")
@_project_context_options
@click.pass_context
def serving_status(ctx: click.Context, model_name: str) -> None:
    """Show the publication ledger for one search index."""
    from .cli_services.serving import serving_status as _serving_status
    from .retrieval import ServingCoordinationError

    try:
        entry = _serving_status(
            ctx.obj["project_dir"],
            profiles_dir=ctx.obj["profiles_dir"],
            target=ctx.obj["target"],
            model_name=model_name,
        )
    except (ConfigError, ProfileError) as e:
        raise ConfigClickError(str(e)) from e
    except (AdapterError, ServingCoordinationError) as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"status:            {entry.status}")
    click.echo(f"fencing_token:     {entry.fencing_token}")
    click.echo(f"active_generation: {entry.active_generation or '-'}")
    click.echo(f"active_collection: {entry.active_collection or '- (default)'}")
    click.echo(f"publisher:         {'active' if entry.publication_id else '-'}")
    click.echo(f"query_leases:      {entry.query_leases}")
    click.echo(f"safe_error_code:   {entry.safe_error_code or '-'}")
    click.echo(
        "rows:              "
        f"inserted={entry.rows_inserted} updated={entry.rows_updated} "
        f"skipped={entry.rows_skipped} deleted={entry.rows_deleted}"
    )


@serving.command("recover")
@click.argument("model_name")
@click.option(
    "--owner-terminated",
    is_flag=True,
    help=(
        "Confirm every previous publisher and query process for this scope "
        "has been terminated. Recovery is refused without this confirmation."
    ),
)
@_project_context_options
@click.pass_context
def serving_recover(
    ctx: click.Context, model_name: str, owner_terminated: bool
) -> None:
    """Explicitly reassign serving authority after terminating the old owner.

    There is no timeout-based lease stealing: recovery advances the fencing
    token so any surviving process fails its next verification, clears all
    leases, and leaves the scope failed until the next successful publish.
    """
    from .cli_services.serving import serving_recover as _serving_recover
    from .retrieval import ServingCoordinationError

    try:
        entry = _serving_recover(
            ctx.obj["project_dir"],
            profiles_dir=ctx.obj["profiles_dir"],
            target=ctx.obj["target"],
            model_name=model_name,
            owner_terminated=owner_terminated,
        )
    except (ConfigError, ProfileError) as e:
        raise ConfigClickError(str(e)) from e
    except (AdapterError, ServingCoordinationError) as e:
        raise click.ClickException(str(e)) from e
    click.echo(
        f"Recovered serving scope for '{model_name}': status={entry.status}, "
        f"fencing_token={entry.fencing_token}. Re-run `stel run` to publish."
    )


@serving.command("migrate-scope")
@click.argument("model_name")
@_project_context_options
@click.pass_context
def serving_migrate_scope(ctx: click.Context, model_name: str) -> None:
    """Move a search index's serving scope onto its logical-collection key.

    Issue #355 re-keys the retrieval serving scope from the physical
    collection to the logical one, so the ledger stays readable once a
    logical collection can have several physical generations behind it. An
    index published before that change keeps its ledger row and publication
    state under the old key, where nothing looks for it — and stel treats an
    index with unreachable state as unpublished, which means re-embedding it.

    Run this once per affected index. It is idempotent: a second run reports
    zero rows moved.
    """
    from .cli_services.serving import serving_migrate_scope as _migrate
    from .retrieval import ServingCoordinationError

    try:
        result = _migrate(
            ctx.obj["project_dir"],
            profiles_dir=ctx.obj["profiles_dir"],
            target=ctx.obj["target"],
            model_name=model_name,
        )
    except (ConfigError, ProfileError) as e:
        raise ConfigClickError(str(e)) from e
    except (AdapterError, ServingCoordinationError) as e:
        raise click.ClickException(str(e)) from e
    if not result["state_rows"] and not result["ledger_rows"]:
        click.echo(f"Nothing to migrate for '{model_name}'; scope is already current.")
        return
    click.echo(
        f"Migrated serving scope for '{model_name}': "
        f"ledger_rows={result['ledger_rows']} state_rows={result['state_rows']}"
    )


@cli.command()
@click.option(
    "--from-candidates",
    "relation",
    required=True,
    help="Relation holding candidate judgments, produced by the transcript project.",
)
@click.option(
    "--output",
    "output",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Golden-set file to draft, e.g. golden_sets/context_search.yml.",
)
@click.option(
    "--promoted-by",
    "promoted_by",
    required=True,
    help="Who is promoting these rows. Recorded on every row as provenance.",
)
@click.option(
    "--context-model",
    "context_model",
    default=None,
    help=(
        "Draft only judgments of this context model. Required when the "
        "candidates span more than one index."
    ),
)
@click.option(
    "--write",
    is_flag=True,
    help="Write the draft. Without this it is printed and nothing changes.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite an existing golden-set file, discarding its review.",
)
@_project_context_options
@click.pass_context
def promote(
    ctx: click.Context,
    relation: str,
    output: Path,
    promoted_by: str,
    context_model: str | None,
    write: bool,
    force: bool,
) -> None:
    """Draft a golden set from candidate judgments, for review (#380).

    This drafts; it does not promote. The file it writes is the artifact a
    human reads, edits and merges — nothing becomes a golden until that
    happens, and no eval reads the candidates it came from.

    Only ids an answer actually cited become `relevant_ids`. Ids that were
    returned and not cited are left out on purpose: an agent may use a chunk
    without naming it, so that is absence of evidence, not evidence of
    irrelevance.
    """
    from .cli_services.promote import promote_from_candidates

    try:
        rendered, draft = promote_from_candidates(
            ctx.obj["project_dir"],
            profiles_dir=ctx.obj["profiles_dir"],
            target=ctx.obj["target"],
            relation=relation,
            output=output,
            promoted_by=promoted_by,
            context_model=context_model,
            write=write,
            force=force,
        )
    except (ConfigError, ProfileError) as e:
        raise ConfigClickError(str(e)) from e
    except AdapterError as e:
        raise click.ClickException(str(e)) from e

    if not write:
        click.echo(rendered, nl=False)
    click.echo(
        f"{len(draft.drafted)} quer{'y' if len(draft.drafted) == 1 else 'ies'} "
        f"drafted{f' to {output}' if write else ''}."
    )
    # Shown for confirmation: a transcribed query is more faithful than a
    # remembered one, but it is still the reviewer who decides it is the
    # question this golden should ask (#380 constraint 2).
    for query in draft.drafted:
        source = "from corpus" if query.text_from_corpus else "NEEDS TEXT"
        click.echo(f"  {query.query_id} [{source}] {query.query_text}")
    needs_text = draft.needs_text
    if needs_text:
        click.echo(
            f"{len(needs_text)} quer"
            f"{'y' if len(needs_text) == 1 else 'ies'} captured no text; "
            "write the query each one asks before this set will load.",
            err=True,
        )
    for skipped in draft.skipped:
        click.echo(
            f"skipped {skipped.query_fingerprint[:12]}: {skipped.reason}", err=True
        )
    if not write:
        click.echo("Re-run with --write to draft the file.")


@cli.group()
def suggest() -> None:
    """Propose context improvements from the agent-transcript corpus (#361)."""


@suggest.command("dbt")
@click.option(
    "--from",
    "relation",
    required=True,
    help="Relation holding candidate suggestions, produced by the analysis project.",
)
@click.option(
    "--dbt-project",
    "dbt_project",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="dbt project directory to patch. Only models/**/*.yml is ever touched.",
)
@click.option(
    "--min-evidence",
    default=DEFAULT_MIN_EVIDENCE,
    show_default=True,
    help="Distinct sessions required before a gap is worth proposing a change for.",
)
@click.option(
    "--write",
    is_flag=True,
    help="Apply the patch. Without this the diff is printed and nothing changes.",
)
@_project_context_options
@click.pass_context
def suggest_dbt(
    ctx: click.Context,
    relation: str,
    dbt_project: Path,
    min_evidence: int,
    write: bool,
) -> None:
    """Propose `description:` for dbt models agents keep having to read.

    Only absent descriptions are filled and only `description:` is touched, so
    a suggestion can add context but never overwrite it. Review the diff and
    merge it like any other change; stel does not commit.
    """
    from .cli_services.suggest import suggest_dbt as _suggest_dbt

    try:
        diff, outcomes = _suggest_dbt(
            ctx.obj["project_dir"],
            profiles_dir=ctx.obj["profiles_dir"],
            target=ctx.obj["target"],
            relation=relation,
            dbt_project_dir=dbt_project,
            min_evidence=min_evidence,
            write=write,
        )
    except (ConfigError, ProfileError) as e:
        raise ConfigClickError(str(e)) from e
    except AdapterError as e:
        raise click.ClickException(str(e)) from e

    applied = [outcome for outcome in outcomes if outcome.applied]
    if not applied:
        click.echo("No suggestions to apply.")
    else:
        click.echo(diff, nl=False)
        click.echo(
            f"{len(applied)} suggestion(s) "
            f"{'applied' if write else 'proposed'}."
        )
        if not write:
            click.echo("Re-run with --write to apply.")
    skipped = [outcome for outcome in outcomes if not outcome.applied]
    for outcome in skipped:
        click.echo(f"skipped {outcome.target}: {outcome.reason}", err=True)


@cli.group()
def prompts() -> None:
    """Manage versioned prompt artifacts (issue #303)."""


@prompts.command("lock")
@click.option(
    "--force",
    is_flag=True,
    help="Re-lock released versions whose contents changed. Deliberate only.",
)
@_project_context_options
@click.pass_context
def prompts_lock(ctx: click.Context, force: bool) -> None:
    """Record every prompt version's contents in `prompts/lock.json`.

    Commit the lock: its diff is what makes a changed prompt visible in review,
    and `stel prompts check` is what makes it fail a build.
    """
    project_dir: Path = ctx.obj["project_dir"]
    try:
        added, rewritten = write_lock(project_dir, force=force)
    except PromptLockError as error:
        raise ConfigClickError(str(error)) from error
    total = len(read_lock(project_dir))
    click.echo(
        f"Locked {total} prompt version(s) in "
        f"{lock_path(project_dir).relative_to(project_dir).as_posix()} "
        f"({added} added"
        + (f", {rewritten} re-locked" if rewritten else "")
        + ")."
    )


@prompts.command("check")
@_project_context_options
@click.pass_context
def prompts_check(ctx: click.Context) -> None:
    """Fail if a released prompt version changed, or the lock is stale.

    The CI gate for prompt immutability: editing a released version should be
    a failed build, not a silent reprocess. The fix is to add the next version.
    """
    project_dir: Path = ctx.obj["project_dir"]
    try:
        drift = check_lock(project_dir)
    except PromptLockError as error:
        raise ConfigClickError(str(error)) from error
    if not drift:
        click.echo(f"{len(read_lock(project_dir))} prompt version(s) unchanged.")
        return
    lines = "\n".join(item.describe() for item in drift)
    raise ConfigClickError(f"Prompt lock check failed:\n{lines}")


@cli.group()
def providers() -> None:
    """Inspect registered and discovered inference/embedding providers."""


@providers.command("list")
@click.option(
    "--output",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
)
def providers_list(output: str) -> None:
    """List built-in and entry-point-discovered providers.

    Separately packaged providers load from the versioned entry-point groups
    (issue #71); plugins built against another provider contract version are
    shown as incompatible instead of being silently ignored.
    """
    from .providers import ProviderRegistrationError, provider_inventory

    try:
        entries = provider_inventory()
    except ProviderRegistrationError as e:
        raise click.ClickException(f"Provider plugin discovery failed: {e}") from e
    if output == "json":
        click.echo(
            json.dumps(
                [
                    {
                        "capability": entry.capability,
                        "name": entry.name,
                        "distribution": entry.distribution,
                        "status": entry.status,
                        "detail": entry.detail,
                    }
                    for entry in entries
                ],
                indent=2,
            )
        )
        return
    header = f"{'capability':<12}{'name':<20}{'distribution':<28}{'status':<14}detail"
    click.echo(header)
    click.echo("-" * 110)
    for entry in entries:
        click.echo(
            f"{entry.capability:<12}{entry.name:<20}{entry.distribution:<28}"
            f"{entry.status:<14}{entry.detail}"
        )


@cli.group()
def transcripts() -> None:
    """Convert agent-session transcripts into transcript/v1 landing files.

    Claude Code and Codex sessions become one reduced, exchange-structured
    JSON document each, ready to consume as an ordinary local json source
    (issue #360). Prose is kept; tool exhaust is reduced to name, argument
    fingerprint, outcome, and byte count.
    """


@transcripts.command("convert")
@click.argument(
    "paths",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--out",
    "out_dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Landing directory for transcript/v1 documents.",
)
@click.option(
    "--capture-context-queries",
    is_flag=True,
    default=False,
    help="Record the query text of stel search_context calls, not just its "
    "fingerprint. Off by default, mirroring the query log's separate "
    "capture_query_text opt-in.",
)
def transcripts_convert(
    paths: tuple[Path, ...], out_dir: Path, capture_context_queries: bool
) -> None:
    """Convert specific transcript files (assumed final, e.g. from a
    SessionEnd hook). The harness is detected per file."""
    from .transcripts import convert_file

    for path in paths:
        landed = convert_file(
            path, out_dir, capture_query=capture_context_queries
        )
        if landed is None:
            raise click.ClickException(
                f"Not a recognized agent transcript with conversation: {path}"
            )
        click.echo(f"wrote {landed}")


@transcripts.command("sync")
@click.option(
    "--out",
    "out_dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Landing directory for transcript/v1 documents.",
)
@click.option(
    "--claude-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Claude Code projects directory [default: ~/.claude/projects].",
)
@click.option(
    "--codex-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Codex sessions directory [default: ~/.codex/sessions].",
)
@click.option(
    "--min-idle-seconds",
    type=click.FloatRange(min=0),
    default=None,
    help="Skip transcripts modified more recently than this — the live "
    "session rule [default: 300].",
)
@click.option(
    "--capture-context-queries",
    is_flag=True,
    default=False,
    help="Record the query text of stel search_context calls, not just its "
    "fingerprint. Off by default, mirroring the query log's separate "
    "capture_query_text opt-in.",
)
def transcripts_sync(
    out_dir: Path,
    claude_dir: Path | None,
    codex_dir: Path | None,
    min_idle_seconds: float | None,
    capture_context_queries: bool,
) -> None:
    """Scan the harness directories and convert every settled transcript."""
    from .transcripts import (
        DEFAULT_MIN_IDLE_SECONDS,
        default_claude_dir,
        default_codex_dir,
        sync_transcripts,
    )

    written = sync_transcripts(
        out_dir=out_dir,
        claude_dir=claude_dir if claude_dir is not None else default_claude_dir(),
        codex_dir=codex_dir if codex_dir is not None else default_codex_dir(),
        min_idle_seconds=(
            min_idle_seconds
            if min_idle_seconds is not None
            else DEFAULT_MIN_IDLE_SECONDS
        ),
        capture_query=capture_context_queries,
    )
    click.echo(f"wrote {len(written)} transcript document(s) to {out_dir}")


@cli.command()
@_project_context_options
@click.pass_context
def clean(ctx: click.Context) -> None:
    """Remove generated files under the project's target path.

    Known local artifacts are removed without invoking a warehouse-level reset.
    Configured databases and unknown files under target-path are preserved.
    """
    project_dir: Path = ctx.obj["project_dir"]
    try:
        path = clean_project(project_dir)
    except (ConfigError, RunError) as e:
        raise ConfigClickError(str(e)) from e
    click.echo(f"Cleaned generated artifacts under {path}")


@cli.command()
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the renames without performing them.",
)
@_project_context_options
@click.pass_context
def migrate(ctx: click.Context, dry_run: bool) -> None:
    """Rename stel's internal warehouse tables to their current names.

    Needed once per target built before the #313 rename, which moved every
    internal table from `dbt_ml_*` to `stel_*`. Those tables hold incremental
    state and live publication claims, so until they are carried over every
    other command refuses to run rather than treat the project as new and
    reprocess its corpus at provider cost.

    Only tables stel owns are touched, and only inside the schema the target
    already points at. Moving a whole schema is not done here: pin `schema:`
    in the profile to keep using a pre-rename one.
    """
    project_dir: Path = ctx.obj["project_dir"]
    profiles_dir = ctx.obj["profiles_dir"]
    target = ctx.obj["target"]
    project, _, _ = _load(project_dir)
    try:
        resolved = resolve_profile(
            project, project_dir, target=target, profiles_dir=profiles_dir
        )
    except ProfileError as e:
        raise ConfigClickError(str(e)) from e

    adapter = create_adapter(resolved.warehouse, project_dir=project_dir)
    # The one caller allowed past the legacy-name guards: they exist to stop
    # everything else from running against the objects this is here to rename.
    adapter.migration_mode = True
    try:
        with adapter:
            renames = plan_name_migration(adapter)
            if not renames:
                click.echo(
                    f"Nothing to migrate: schema '{adapter.schema}' in the "
                    f"{adapter.adapter_type()} target already uses the current "
                    "internal table names."
                )
                return
            for rename in renames:
                click.echo(f"  {rename.old} -> {rename.new}")
            if dry_run:
                click.echo(f"{len(renames)} table(s) would be renamed. (--dry-run)")
                return
            applied = apply_name_migration(adapter, renames)
    except AdapterError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"Renamed {len(applied)} table(s) in schema '{adapter.schema}'.")


def _backend_for_source(source: SourceConfig, models: list[ModelConfig]) -> str:
    """Find the backend name of the (first) extraction model consuming this source."""
    for model in models:
        if (
            model.extraction is not None
            and model.source
            and parse_ref(model.source) == source.name
        ):
            return model.extraction.backend or "json"
    return "json"


def _is_remote_source_path(path: str) -> bool:
    return path.startswith("gs://")


def _pick_source(sources: list[SourceConfig], name: str | None) -> SourceConfig:
    if name:
        match = next((s for s in sources if s.name == name), None)
        if match is None:
            raise click.ClickException(
                f"Source '{name}' not found. Available: {[s.name for s in sources]}"
            )
        return match
    if len(sources) == 1:
        return sources[0]
    if not sources:
        raise click.ClickException("Project has no sources defined.")
    raise click.ClickException(
        f"Project has multiple sources; pass --source. Available: {[s.name for s in sources]}"
    )


def main() -> None:
    cli(obj={})


if __name__ == "__main__":
    main()
