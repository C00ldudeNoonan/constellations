"""Scale benchmark: how fast does stel actually run?

Generates a configurable number of synthetic invoices into a temp project,
then times compile / first run / incremental run / single-doc change.

Run:  uv run python scripts/benchmark.py [--count 5000]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from stel.config import load_project
from stel.runner import run_project
from stel.synth import generate_invoices


@contextmanager
def timed(label: str, samples: list[tuple[str, float]]) -> Iterator[None]:
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    samples.append((label, elapsed))
    print(f"  {label:40s} {elapsed:8.3f}s")


def setup_project(src: Path, dst: Path) -> None:
    shutil.copytree(
        src, dst, ignore=shutil.ignore_patterns("data", "target", "__pycache__")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    example = repo / "examples" / "invoice_pipeline"

    print(f"== stel scale benchmark - {args.count} documents ==\n")
    samples: list[tuple[str, float]] = []

    with TemporaryDirectory() as tmp:
        project = Path(tmp) / "bench"
        setup_project(example, project)
        invoices_dir = project / "data" / "invoices"

        with timed(f"seed {args.count} invoices", samples):
            generate_invoices(args.count, invoices_dir, seed=args.seed)

        with timed("load_project (config + DAG)", samples):
            load_project(project)

        with timed("first run (cold)", samples):
            results = run_project(project)
        print(
            "    ->",
            ", ".join(
                f"{r.model_name}: processed={r.documents_processed}, "
                f"rows={r.rows_written}"
                for r in results
            ),
        )

        with timed("second run (all skipped)", samples):
            results = run_project(project)
        skipped_total = sum(r.documents_skipped for r in results)
        print(f"    -> skipped {skipped_total} docs")

        target = invoices_dir / f"invoice_{0:05d}.json"
        data = json.loads(target.read_text())
        data["vendor"] = "MUTATED"
        target.write_text(json.dumps(data))

        with timed("third run (1 changed)", samples):
            results = run_project(project)
        raw = next(r for r in results if r.model_name == "raw_invoices")
        print(f"    -> processed={raw.documents_processed}, skipped={raw.documents_skipped}")

        with timed("full-refresh", samples):
            run_project(project, full_refresh=True)

    print()
    print(f"{'step':40s} {'time':>9s}  {'docs/sec':>10s}")
    print("-" * 65)
    for label, t in samples:
        per_sec = args.count / t if t > 0 else 0
        print(f"{label:40s} {t:8.3f}s  {per_sec:10.1f}")


if __name__ == "__main__":
    main()
