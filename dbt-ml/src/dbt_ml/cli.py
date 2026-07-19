from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click

from .adapters import AdapterError, create_adapter
from .checks import TestResult, run_project_tests
from .compiler import validate_project_contract, validate_warehouse_capabilities
from .config import ConfigError, load_project
from .config.model import ModelConfig
from .config.project import ProjectConfig
from .config.source import SourceConfig
from .credentials import CredentialReference, CredentialReferenceError
from .dag import DAGError, ProjectDAG, SelectionError, parse_ref
from .dbt_export import write_dbt_sources
from .docs import DocsError, generate_docs, serve_docs
from .freshness import check_freshness
from .manifest import StateError, write_manifest, write_run_results
from .optional_dependencies import OptionalDependencyError
from .paths import resolve_within_project
from .profile import (
    ProfileError,
    apply_source_path_overrides,
    resolve_llm_options,
    resolve_profile,
)
from .providers import (
    ProviderConfigurationError,
    ProviderNotFoundError,
    get_inference_provider,
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


class ConfigClickError(click.ClickException):
    """A configuration/usage error the run never got past. Exits 2 so an
    orchestrator (issue #87) can tell a broken project apart from a run that
    started but had a model fail (exit 1)."""

    exit_code = 2


# Errors that mean the project couldn't be coherently set up → exit 2. RunError
# (a run that started but a model failed hard) stays a plain ClickException → 1.
_CONFIG_ERRORS = (
    ConfigError,
    DAGError,
    SelectionError,
    ProfileError,
    ProviderConfigurationError,
    ProviderNotFoundError,
    StateError,
    OptionalDependencyError,
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
        help="Path to the dbt-ml project (where dbt_ml_project.yml lives).",
    )(command)
    return command


@click.group()
@click.option(
    "--project-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path.cwd,
    help="Path to the dbt-ml project (where dbt_ml_project.yml lives).",
)
@click.option(
    "--profiles-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Directory containing profiles.yml. Overrides discovery.",
)
@click.option("--target", default=None, help="Target name within the active profile.")
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
        validate_warehouse_capabilities(models, resolved.warehouse.type)
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
            if not provider.requires_credentials:
                continue
            env_value = options.get("api_key_env") or provider.default_credential_env
            if env_value is None:
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
    """Scaffold a new dbt-ml project at ./<name>/."""
    target = Path.cwd() / name
    if target.exists():
        raise click.ClickException(f"{target} already exists")

    template_dir = Path(__file__).parent / "templates" / template
    if not template_dir.is_dir():
        raise click.ClickException(f"Template directory missing: {template_dir}")

    shutil.copytree(template_dir, target)
    for path in target.rglob(".gitkeep"):
        path.unlink()

    for filename in ("dbt_ml_project.yml", "profiles.yml"):
        path = target / filename
        if path.exists():
            path.write_text(path.read_text().replace("__PROJECT_NAME__", name))

    click.echo(f"Created dbt-ml project at {target} (template: {template})")
    click.echo("")
    click.echo("Next:")
    click.echo(f"  cd {name}")
    if template == "json":
        click.echo("  uv run dbt-ml seed --count 20")
    else:
        click.echo(
            f"  # drop your {template} files into ./data/, "
            "or `dbt-ml seed --count 20` for synthetic data"
        )
    click.echo("  uv run dbt-ml run")
    click.echo("  uv run dbt-ml test")


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
            "`dbt-ml search`."
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
                    f"Run `dbt-ml run` first. Available: {tables or '(none)'}"
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
    help="Repeatable FIELD OP VALUE filter. Use a JSON array with the 'in' operator.",
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


def _parse_search_filter(field: str, operator: str, value: str) -> SearchFilter:
    resolved_operator = SearchFilterOperator(operator)
    if resolved_operator != SearchFilterOperator.IN:
        return SearchFilter(field, resolved_operator, value)
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        raise ValueError("Search filter 'in' values must be a JSON array") from None
    if not isinstance(decoded, list) or not decoded:
        raise ValueError("Search filter 'in' values must be a non-empty JSON array")
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
            f"`dbt-ml seed` only supports local source paths; source "
            f"'{source.name}' points to remote path '{source.path}'. "
            "Use a local target source_paths override, or seed remote storage "
            "outside dbt-ml."
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

    rows: list[dict[str, object]] = []
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
        tags = ",".join(row["tags"]) if row["tags"] else "-"  # type: ignore[arg-type]
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
    if model.search is not None:
        return "search"
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
    "--json",
    "json_output",
    is_flag=True,
    help="Print the run_results.json payload to stdout instead of the table.",
)
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
    json_output: bool,
) -> None:
    """Extract and materialize selected models into the configured warehouse."""
    project_dir: Path = ctx.obj["project_dir"]
    profiles_dir = ctx.obj["profiles_dir"]
    target = ctx.obj["target"]

    if watch:
        _run_watch(
            project_dir,
            profiles_dir=profiles_dir,
            target=target,
            full_refresh=full_refresh,
            select=select,
            exclude=exclude,
            threads=threads,
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
        click.echo(results_path.read_text())
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
    if "provider_calls" in m:
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
    help="Persist failing test rows to dbt_ml_test_failures__* tables.",
)
@click.option(
    "--state",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Previous manifest.json (or its directory) for state:modified selection.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Print the run_results.json payload to stdout instead of the table.",
)
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
    json_output: bool,
) -> None:
    """Run and test each model in dependency order; downstream models are skipped
    when an upstream model errors or fails a test."""
    project_dir: Path = ctx.obj["project_dir"]
    profiles_dir = ctx.obj["profiles_dir"]
    target = ctx.obj["target"]
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
        click.echo(results_path.read_text())
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
    help="Persist failing test rows to dbt_ml_test_failures__* tables.",
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
    """Write a dbt-compatible sources.yml declaring dbt_ml's materialized tables.

    Drop the output into a dbt project using the matching warehouse adapter so
    models can refer to dbt-ml tables via `{{ source(...) }}`. With
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


def _load(project_dir: Path) -> tuple[ProjectConfig, list[SourceConfig], list[ModelConfig]]:
    try:
        return load_project(project_dir)
    except ConfigError as e:
        raise ConfigClickError(str(e)) from e


def _build_dag(sources: list[SourceConfig], models: list[ModelConfig]) -> ProjectDAG:
    try:
        return ProjectDAG(sources, models)
    except DAGError as e:
        raise ConfigClickError(str(e)) from e


def _run_watch(
    project_dir: Path,
    *,
    profiles_dir: Path | None,
    target: str | None,
    full_refresh: bool,
    select: str | None,
    exclude: str | None,
    threads: int = 1,
) -> None:
    """Watch source paths and re-run on changes. Blocking; Ctrl-C to exit."""
    from watchfiles import watch

    project, sources, models = _load(project_dir)
    try:
        dag = validate_project_contract(project, sources, models, project_dir)
        selected = dag.select_models(select=select, exclude=exclude)
        if not selected:
            click.echo("No models selected.")
            return
        required_sources = set(dag.required_sources(selected))
        resolved = resolve_profile(
            project, project_dir, target=target, profiles_dir=profiles_dir
        )
        selected_names = set(selected)
        validate_warehouse_capabilities(
            [model for model in models if model.name in selected_names],
            resolved.warehouse.type,
        )
        sources = apply_source_path_overrides(sources, resolved)
    except _CONFIG_ERRORS as e:
        raise ConfigClickError(str(e)) from e
    watch_paths = []
    for s in sources:
        if s.name not in required_sources:
            continue
        try:
            candidate = resolve_within_project(
                s.path,
                project_dir,
                surface=f"Source '{s.name}' path",
                external=s.external,
                hint="Set `external: true` on the source to allow it.",
            )
        except ConfigError as e:
            raise ConfigClickError(str(e)) from e
        if candidate.exists():
            watch_paths.append(candidate)
    if not watch_paths:
        raise click.ClickException(
            "No source paths exist on disk yet. Create them (or run `dbt-ml seed`) "
            "and try `dbt-ml run --watch` again."
        )

    click.echo(f"watching {len(watch_paths)} source path(s); Ctrl-C to stop")

    def _do_run() -> None:
        try:
            results = run_project(
                project_dir,
                full_refresh=full_refresh,
                select=select,
                exclude=exclude,
                target=target,
                profiles_dir=profiles_dir,
                threads=threads,
            )
        except (*_CONFIG_ERRORS, RunError) as e:
            click.echo(f"error: {e}", err=True)
            return
        write_manifest(project_dir, target=target, profiles_dir=profiles_dir)
        write_run_results(
            project_dir, results, target=target, profiles_dir=profiles_dir
        )
        for r in results:
            click.echo(
                f"  {r.model_name:<22} {r.kind:<12} "
                f"processed={r.documents_processed:<5} skipped={r.documents_skipped:<5} "
                f"rows={r.rows_written}"
            )

    _do_run()
    try:
        for _ in watch(*watch_paths, debounce=500, recursive=True):
            click.echo("change detected, re-running...")
            _do_run()
    except KeyboardInterrupt:
        click.echo("watch stopped.")


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
