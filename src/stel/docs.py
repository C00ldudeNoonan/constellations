"""Render a static docs site from manifest.json + run_results.json."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .config import load_project
from .manifest import (
    MANIFEST_FILENAME,
    RUN_RESULTS_FILENAME,
    write_manifest,
)


class DocsError(Exception):
    pass


@dataclass
class DocsResult:
    output_dir: Path
    pages_written: int


def generate_docs(
    project_dir: Path,
    *,
    target: str | None = None,
    profiles_dir: Path | None = None,
    output_dir: Path | None = None,
) -> DocsResult:
    """Generate target/docs/ (or `output_dir`) from the project's manifest."""
    project, _, _ = load_project(project_dir)
    target_dir = (project_dir / project.target_path).resolve()
    out_dir = output_dir or (target_dir / "docs")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = target_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        # Generate the manifest on the fly if missing.
        write_manifest(project_dir, target=target, profiles_dir=profiles_dir)

    manifest = json.loads((target_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    run_results = _read_run_results(target_dir)
    last_run_by_model = {r["model_name"]: r for r in run_results}

    env = _jinja_env()
    pages = 0

    pages += _write(out_dir / "index.html", env.get_template("index.html"), {
        **manifest,
        "generated_at": manifest["generated_at"],
    })
    pages += _write(out_dir / "lineage.html", env.get_template("lineage.html"), {
        **manifest,
        "generated_at": manifest["generated_at"],
    })
    for model in manifest["models"]:
        pages += _write(
            out_dir / f"model_{model['name']}.html",
            env.get_template("model.html"),
            {
                **manifest,
                "generated_at": manifest["generated_at"],
                "model": model,
                "last_run": last_run_by_model.get(model["name"]),
            },
        )

    return DocsResult(output_dir=out_dir, pages_written=pages)


def serve_docs(project_dir: Path, *, port: int = 8080) -> None:
    """Serve the generated docs over HTTP. Blocking; Ctrl-C to stop."""
    import http.server
    import socketserver

    project, _, _ = load_project(project_dir)
    out_dir = (project_dir / project.target_path / "docs").resolve()
    if not out_dir.exists():
        raise DocsError(
            f"No docs at {out_dir}. Run `stel docs generate` first."
        )

    handler_cls = http.server.SimpleHTTPRequestHandler

    class _Handler(handler_cls):  # type: ignore[valid-type, misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(out_dir), **kwargs)

    with socketserver.TCPServer(("127.0.0.1", port), _Handler) as httpd:
        print(f"Serving {out_dir} at http://127.0.0.1:{port}/  (Ctrl-C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")


def _jinja_env() -> Environment:
    template_dir = Path(__file__).parent / "templates" / "docs"
    return Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html"]),
    )


def _write(path: Path, template: Any, context: dict[str, Any]) -> int:
    path.write_text(template.render(**context), encoding="utf-8")
    return 1


def _read_run_results(target_dir: Path) -> list[dict[str, Any]]:
    p = target_dir / RUN_RESULTS_FILENAME
    if not p.exists():
        return []
    return cast(list[dict[str, Any]], json.loads(p.read_text(encoding="utf-8")).get("results", []))
