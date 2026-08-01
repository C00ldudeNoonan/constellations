from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from ..adapters import WarehouseAdapter
from ..test_specs import (
    SUPPORTED_TESTS,
    TestSpecError,
    parse_test_spec,
)
from .python import CustomTestError, run_python_test

UnknownTestError = TestSpecError


@dataclass
class TestResult:
    __test__ = False  # tell pytest not to collect this dataclass as a test class

    test_name: str
    model_name: str
    column: str | None
    status: str  # "pass" | "warn" | "fail"
    message: str = ""
    severity: str = "error"
    failures_table: str | None = None
    failure_count: int | None = None

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    @property
    def is_hard_failure(self) -> bool:
        """True when this result should cause the run to exit non-zero."""
        return self.status == "fail"


def evaluate_test_spec(
    spec: Any,
    *,
    model_name: str,
    table_ref: str,
    adapter: WarehouseAdapter,
    project_dir: Path | None = None,
    store_failures: bool = False,
) -> list[TestResult]:
    """Parse one test spec and run it.

    Accepted forms:
        "not_empty"                                  -> bare test name
        {not_null: [a, b]}                           -> single-key mapping
        {not_null: [a, b], severity: warn}           -> with severity sibling key
        {python: my.module.path}                     -> custom Python test

    When `store_failures` is set, supporting tests persist their failing rows to
    a `dbt_ml_test_failures__…` table and record it on the result.
    """
    parsed = parse_test_spec(spec)
    return _apply_severity(
        _run_named_test(
            parsed.name,
            parsed.argument,
            model_name,
            table_ref,
            adapter,
            project_dir,
            store_failures,
        ),
        parsed.severity,
    )


def _apply_severity(results: list[TestResult], severity: str) -> list[TestResult]:
    out: list[TestResult] = []
    for r in results:
        new_status = r.status
        if r.status == "fail" and severity == "warn":
            new_status = "warn"
        out.append(
            TestResult(
                test_name=r.test_name,
                model_name=r.model_name,
                column=r.column,
                status=new_status,
                message=r.message,
                severity=severity,
                failures_table=r.failures_table,
                failure_count=r.failure_count,
            )
        )
    return out


def _run_named_test(
    test_name: str,
    arg: Any,
    model_name: str,
    table_ref: str,
    adapter: WarehouseAdapter,
    project_dir: Path | None,
    store_failures: bool = False,
) -> list[TestResult]:
    if test_name == "not_null":
        return _not_null(model_name, table_ref, adapter, arg, store_failures)
    if test_name == "unique":
        return [_unique(model_name, table_ref, adapter, arg, store_failures)]
    if test_name == "min_rows":
        return [_min_rows(model_name, table_ref, adapter, int(arg))]
    if test_name == "not_empty":
        return [_min_rows(model_name, table_ref, adapter, 1, display_as="not_empty")]
    if test_name == "python":
        if project_dir is None:
            raise UnknownTestError(
                "Custom python test requires the test runner to know project_dir; "
                "this usually means you're calling evaluate_test_spec directly without it."
            )
        return [_python(model_name, table_ref, adapter, str(arg), project_dir)]
    if test_name == "matches_regex":
        return [_matches_regex(model_name, table_ref, adapter, arg, store_failures)]
    if test_name == "accepted_values":
        return [_accepted_values(model_name, table_ref, adapter, arg, store_failures)]
    if test_name == "accepted_range":
        return [_accepted_range(model_name, table_ref, adapter, arg, store_failures)]
    if test_name == "null_rate":
        return [_null_rate(model_name, table_ref, adapter, arg, store_failures)]
    if test_name == "grounded_in":
        return [_grounded_in(model_name, table_ref, adapter, arg, store_failures)]
    if test_name == "relationships":
        return [_relationships(model_name, table_ref, adapter, arg, store_failures)]
    if test_name == "embedding_valid":
        return [_embedding_valid(model_name, table_ref, adapter, arg)]
    if test_name == "embedding_variance":
        return [_embedding_variance(model_name, table_ref, adapter, arg)]
    if test_name == "embedding_duplicates":
        return [_embedding_duplicates(model_name, table_ref, adapter, arg)]
    if test_name == "embedding_outliers":
        return [_embedding_outliers(model_name, table_ref, adapter, arg)]
    raise UnknownTestError(
        f"Unknown test '{test_name}'. Supported: {sorted(SUPPORTED_TESTS)}"
    )


def _slug(value: str) -> str:
    return re.sub(r"\W+", "_", value).strip("_")


def _failures_table_name(model_name: str, test_name: str, column: str | None) -> str:
    parts = ["dbt_ml_test_failures", _slug(model_name), _slug(test_name)]
    if column:
        parts.append(_slug(str(column)))
    return "__".join(parts)


def _store(
    adapter: WarehouseAdapter,
    model_name: str,
    test_name: str,
    column: str | None,
    select_sql: str,
    params: list[Any] | None,
    result: TestResult,
) -> None:
    """Materialize the rows selected by `select_sql` into a failures table and
    annotate `result` with the table name + row count."""
    table = _failures_table_name(model_name, test_name, column)
    df = adapter.query_df(select_sql, params)
    adapter.materialize_full(table, df)
    result.failures_table = table
    result.failure_count = df.height


def _store_df(
    adapter: WarehouseAdapter,
    model_name: str,
    test_name: str,
    column: str | None,
    df: pl.DataFrame,
    result: TestResult,
) -> None:
    table = _failures_table_name(model_name, test_name, column)
    adapter.materialize_full(table, df)
    result.failures_table = table
    result.failure_count = df.height


def _not_null(
    model_name: str,
    table_ref: str,
    adapter: WarehouseAdapter,
    arg: Any,
    store_failures: bool = False,
) -> list[TestResult]:
    cols = arg if isinstance(arg, list) else [arg]
    results: list[TestResult] = []
    for col in cols:
        where = f"{adapter.quote_ident(col)} IS NULL"
        count = adapter.scalar(f"SELECT COUNT(*) FROM {table_ref} WHERE {where}") or 0
        result = TestResult(
            test_name="not_null",
            model_name=model_name,
            column=col,
            status="pass" if count == 0 else "fail",
            message="" if count == 0 else f"{count} rows have NULL {col}",
        )
        if count and store_failures:
            _store(
                adapter, model_name, "not_null", col,
                f"SELECT * FROM {table_ref} WHERE {where}", None, result,
            )
        results.append(result)
    return results


def _unique(
    model_name: str,
    table_ref: str,
    adapter: WarehouseAdapter,
    arg: Any,
    store_failures: bool = False,
) -> TestResult:
    cols = arg if isinstance(arg, list) else [arg]
    col_list = ", ".join(adapter.quote_ident(c) for c in cols)
    count = adapter.scalar(
        f"SELECT COUNT(*) FROM ("
        f"  SELECT {col_list} FROM {table_ref}"
        f"  GROUP BY {col_list} HAVING COUNT(*) > 1"
        f")"
    ) or 0
    result = TestResult(
        test_name="unique",
        model_name=model_name,
        column=",".join(cols),
        status="pass" if count == 0 else "fail",
        message="" if count == 0 else f"{count} duplicate groups on {cols}",
    )
    if count and store_failures:
        _store(
            adapter, model_name, "unique", ",".join(cols),
            f"SELECT * FROM {table_ref} WHERE ({col_list}) IN ("
            f"  SELECT {col_list} FROM {table_ref}"
            f"  GROUP BY {col_list} HAVING COUNT(*) > 1)",
            None, result,
        )
    return result


def _python(
    model_name: str,
    table_ref: str,
    adapter: WarehouseAdapter,
    module_path: str,
    project_dir: Path,
) -> TestResult:
    try:
        message = run_python_test(module_path, project_dir, adapter, table_ref)
    except CustomTestError as e:
        return TestResult(
            test_name=f"python:{module_path}",
            model_name=model_name,
            column=None,
            status="fail",
            message=str(e),
        )
    return TestResult(
        test_name=f"python:{module_path}",
        model_name=model_name,
        column=None,
        status="pass" if message is None else "fail",
        message=message or "",
    )


def _min_rows(
    model_name: str,
    table_ref: str,
    adapter: WarehouseAdapter,
    n: int,
    *,
    display_as: str = "min_rows",
) -> TestResult:
    actual = adapter.scalar(f"SELECT COUNT(*) FROM {table_ref}") or 0
    return TestResult(
        test_name=display_as,
        model_name=model_name,
        column=None,
        status="pass" if actual >= n else "fail",
        message=f"actual={actual}, required>={n}",
    )


# ─── deterministic ML / statistical quality checks (issue #10) ─────────────


def _require_dict(test_name: str, arg: Any) -> dict[str, Any]:
    if not isinstance(arg, dict):
        raise UnknownTestError(
            f"Test '{test_name}' expects a mapping of options, got: {arg!r}"
        )
    return arg


def _matches_regex(
    model_name: str, table_ref: str, adapter: WarehouseAdapter, arg: Any,
    store_failures: bool = False,
) -> TestResult:
    """Every non-null value of `column` matches `pattern`. Deterministic."""
    opts = _require_dict("matches_regex", arg)
    column = opts["column"]
    pattern = re.compile(opts["pattern"])

    df = adapter.query_df(f"SELECT * FROM {table_ref}")
    col_values = df[column].to_list() if column in df.columns else []
    mask = [v is not None and not pattern.search(str(v)) for v in col_values]
    misses = [str(v) for v, m in zip(col_values, mask, strict=True) if m]
    n = len(misses)
    sample = ", ".join(repr(m) for m in misses[:3])
    result = TestResult(
        test_name="matches_regex",
        model_name=model_name,
        column=column,
        status="pass" if n == 0 else "fail",
        message="" if n == 0 else f"{n} values don't match {opts['pattern']!r} (e.g. {sample})",
    )
    if n and store_failures:
        _store_df(
            adapter, model_name, "matches_regex", column,
            df.filter(pl.Series(mask)), result,
        )
    return result


def _accepted_values(
    model_name: str, table_ref: str, adapter: WarehouseAdapter, arg: Any,
    store_failures: bool = False,
) -> TestResult:
    """Every non-null value of `column` is in `values`. Deterministic, SQL aggregate."""
    opts = _require_dict("accepted_values", arg)
    column = opts["column"]
    allowed = opts["values"]
    placeholders = ", ".join(["?"] * len(allowed))
    col = adapter.quote_ident(column)
    where = f"{col} IS NOT NULL AND {col} NOT IN ({placeholders})"
    bad = (
        adapter.scalar(f"SELECT COUNT(*) FROM {table_ref} WHERE {where}", list(allowed))
        or 0
    )
    result = TestResult(
        test_name="accepted_values",
        model_name=model_name,
        column=column,
        status="pass" if bad == 0 else "fail",
        message="" if bad == 0 else f"{bad} values outside {allowed}",
    )
    if bad and store_failures:
        _store(
            adapter, model_name, "accepted_values", column,
            f"SELECT * FROM {table_ref} WHERE {where}", list(allowed), result,
        )
    return result


def _accepted_range(
    model_name: str, table_ref: str, adapter: WarehouseAdapter, arg: Any,
    store_failures: bool = False,
) -> TestResult:
    """Numeric `column` within [min, max] (either bound optional). SQL aggregate."""
    opts = _require_dict("accepted_range", arg)
    column = opts["column"]
    col = adapter.quote_ident(column)
    conds = []
    params: list[Any] = []
    if "min" in opts:
        conds.append(f"{col} < ?")
        params.append(opts["min"])
    if "max" in opts:
        conds.append(f"{col} > ?")
        params.append(opts["max"])
    if not conds:
        raise UnknownTestError("accepted_range requires at least one of: min, max")
    where = f'{col} IS NOT NULL AND ({" OR ".join(conds)})'
    bad = adapter.scalar(f"SELECT COUNT(*) FROM {table_ref} WHERE {where}", params) or 0
    bounds = f"[{opts.get('min', '-inf')}, {opts.get('max', 'inf')}]"
    result = TestResult(
        test_name="accepted_range",
        model_name=model_name,
        column=column,
        status="pass" if bad == 0 else "fail",
        message="" if bad == 0 else f"{bad} values outside {bounds}",
    )
    if bad and store_failures:
        _store(
            adapter, model_name, "accepted_range", column,
            f"SELECT * FROM {table_ref} WHERE {where}", params, result,
        )
    return result


def _null_rate(
    model_name: str, table_ref: str, adapter: WarehouseAdapter, arg: Any,
    store_failures: bool = False,
) -> TestResult:
    """Fraction of NULLs in `column` is <= `max`. The #1 silent extraction failure."""
    opts = _require_dict("null_rate", arg)
    column = opts["column"]
    max_rate = float(opts.get("max", 0.0))
    total = adapter.scalar(f"SELECT COUNT(*) FROM {table_ref}") or 0
    if total == 0:
        return TestResult(
            test_name="null_rate", model_name=model_name, column=column,
            status="pass", message="empty table",
        )
    where = f"{adapter.quote_ident(column)} IS NULL"
    nulls = adapter.scalar(f"SELECT COUNT(*) FROM {table_ref} WHERE {where}") or 0
    rate = nulls / total
    result = TestResult(
        test_name="null_rate",
        model_name=model_name,
        column=column,
        status="pass" if rate <= max_rate else "fail",
        message=f"null_rate={rate:.3f} (max {max_rate:.3f}, {nulls}/{total})",
    )
    if result.status == "fail" and store_failures:
        _store(
            adapter, model_name, "null_rate", column,
            f"SELECT * FROM {table_ref} WHERE {where}", None, result,
        )
    return result


def _grounded_in(
    model_name: str, table_ref: str, adapter: WarehouseAdapter, arg: Any,
    store_failures: bool = False,
) -> TestResult:
    """Deterministic faithfulness proxy: the `value` column's text must appear in
    (or fuzzy-match) the `source` column's text — catching hallucinated values
    with zero LLM calls.

    options: value, source, method ("exact" | "fuzzy"), min_score (fuzzy, default 0.8).
    """
    opts = _require_dict("grounded_in", arg)
    value_col = opts["value"]
    source_col = opts["source"]
    method = opts.get("method", "exact")
    min_score = float(opts.get("min_score", 0.8))

    df = adapter.query_df(f"SELECT * FROM {table_ref}")
    mask: list[bool] = []  # True marks an ungrounded (failing) row
    checked = 0
    for row in df.iter_rows(named=True):
        val, src = row.get(value_col), row.get(source_col)
        if val is None or src is None or str(val).strip() == "":
            mask.append(False)
            continue
        checked += 1
        v = str(val).lower().strip()
        s = str(src).lower()
        if method == "exact":
            grounded = v in s
        else:  # fuzzy: best partial ratio of val against any window is approximated
            grounded = v in s or _fuzzy_contains(v, s, min_score)
        mask.append(not grounded)

    ungrounded = sum(mask)
    result = TestResult(
        test_name="grounded_in",
        model_name=model_name,
        column=value_col,
        status="pass" if ungrounded == 0 else "fail",
        message=(
            ""
            if ungrounded == 0
            else (
                f"{ungrounded}/{checked} '{value_col}' values not grounded "
                f"in '{source_col}' ({method})"
            )
        ),
    )
    if ungrounded and store_failures:
        _store_df(
            adapter, model_name, "grounded_in", value_col,
            df.filter(pl.Series(mask)), result,
        )
    return result


def _relationships(
    model_name: str, table_ref: str, adapter: WarehouseAdapter, arg: Any,
    store_failures: bool = False,
) -> TestResult:
    """Referential integrity: every non-null `column` value exists in the parent
    model's `field` column.

    options: column, to (ref of the parent model), field (parent column).
    """
    from ..dag import parse_ref

    opts = _require_dict("relationships", arg)
    column = opts["column"]
    field = opts.get("field") or opts.get("to_field")
    if not opts.get("to") or not field:
        raise UnknownTestError(
            "relationships requires 'column', 'to' (parent ref), and 'field' (parent column)"
        )
    parent_name = parse_ref(str(opts["to"]))
    parent_ref = adapter.table_ref(parent_name)
    col = adapter.quote_ident(column)
    parent_col = adapter.quote_ident(str(field))
    where = (
        f"{col} IS NOT NULL AND {col} NOT IN "
        f"(SELECT {parent_col} FROM {parent_ref} WHERE {parent_col} IS NOT NULL)"
    )
    bad = adapter.scalar(f"SELECT COUNT(*) FROM {table_ref} WHERE {where}") or 0
    result = TestResult(
        test_name="relationships",
        model_name=model_name,
        column=column,
        status="pass" if bad == 0 else "fail",
        message=(
            ""
            if bad == 0
            else f"{bad} '{column}' values missing from {parent_name}.{field}"
        ),
    )
    if bad and store_failures:
        _store(
            adapter, model_name, "relationships", column,
            f"SELECT * FROM {table_ref} WHERE {where}", None, result,
        )
    return result


# ─── embedding-quality checks (issue #10) ──────────────────────────────────


def _load_vectors(
    adapter: WarehouseAdapter, table_ref: str, column: str
) -> tuple[list[list[float]], int, int]:
    """Read only the vector `column` (not whole rows) and return the non-null
    vectors, the null count, and the total row count.

    Reading a single column keeps memory proportional to the embedding data
    rather than the full relation. Non-numeric or missing elements become NaN so
    `embedding_valid` can flag them; other checks assume finite input and rely on
    `embedding_valid` as the upstream gate.
    """
    schema = adapter.query_df(f"SELECT * FROM {table_ref} LIMIT 0")
    if column not in schema.columns:
        raise UnknownTestError(
            f"embedding check column '{column}' not found in the relation; "
            f"got: {sorted(schema.columns)}"
        )
    df = adapter.query_df(f"SELECT {adapter.quote_ident(column)} AS v FROM {table_ref}")
    vectors: list[list[float]] = []
    null_count = 0
    for item in df["v"].to_list():
        if item is None:
            null_count += 1
            continue
        vectors.append([_as_float(x) for x in item])
    return vectors, null_count, len(df)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _consistent_dimensions(vectors: list[list[float]]) -> int | None:
    """The shared dimensionality, or None if the vectors disagree."""
    dims = {len(v) for v in vectors}
    return next(iter(dims)) if len(dims) == 1 else None


def _norm(vector: list[float]) -> float:
    return math.sqrt(math.fsum(x * x for x in vector))


def _embedding_valid(
    model_name: str, table_ref: str, adapter: WarehouseAdapter, arg: Any
) -> TestResult:
    """Per-vector integrity: consistent `dimensions`, finite entries, L2 norm
    within [`min_norm`, `max_norm`], and zero-vector rate <= `max_zero_rate`
    (default 0). A zero or NaN embedding is a common silent provider failure."""
    opts = _require_dict("embedding_valid", arg)
    column = opts["column"]
    dimensions = opts.get("dimensions")
    min_norm = opts.get("min_norm")
    max_norm = opts.get("max_norm")
    max_zero_rate = float(opts.get("max_zero_rate", 0.0))

    vectors, _nulls, total = _load_vectors(adapter, table_ref, column)
    if not vectors:
        return _pass("embedding_valid", model_name, column, "no non-null vectors")

    dim_bad = nonfinite = norm_bad = zero = 0
    for vector in vectors:
        if dimensions is not None and len(vector) != dimensions:
            dim_bad += 1
            continue
        if any(not math.isfinite(x) for x in vector):
            nonfinite += 1
            continue
        norm = _norm(vector)
        if norm == 0.0:
            zero += 1
        if (min_norm is not None and norm < min_norm) or (
            max_norm is not None and norm > max_norm
        ):
            norm_bad += 1
    zero_rate = zero / len(vectors)

    problems = []
    if dim_bad:
        problems.append(f"{dim_bad} wrong-dimension")
    if nonfinite:
        problems.append(f"{nonfinite} non-finite")
    if norm_bad:
        problems.append(f"{norm_bad} out-of-norm-range")
    if zero_rate > max_zero_rate:
        problems.append(f"zero-vector rate {zero_rate:.3f} > {max_zero_rate:.3f}")
    ok = not problems
    return TestResult(
        test_name="embedding_valid",
        model_name=model_name,
        column=column,
        status="pass" if ok else "fail",
        message="" if ok else f"{', '.join(problems)} of {total} vectors",
    )


def _embedding_variance(
    model_name: str, table_ref: str, adapter: WarehouseAdapter, arg: Any
) -> TestResult:
    """Collapse guard: the mean per-dimension (population) variance across the
    embedding set must be >= `min_variance`. Near-zero variance means the
    provider is emitting almost-identical vectors."""
    opts = _require_dict("embedding_variance", arg)
    column = opts["column"]
    min_variance = float(opts["min_variance"])

    vectors, _nulls, _total = _load_vectors(adapter, table_ref, column)
    if len(vectors) < 2:
        return _pass("embedding_variance", model_name, column, "need >= 2 vectors")
    dims = _consistent_dimensions(vectors)
    if dims is None:
        return _inconsistent("embedding_variance", model_name, column)

    n = len(vectors)
    mean = [math.fsum(v[i] for v in vectors) / n for i in range(dims)]
    variance = [
        math.fsum((v[i] - mean[i]) ** 2 for v in vectors) / n for i in range(dims)
    ]
    mean_var = math.fsum(variance) / dims if dims else 0.0
    ok = mean_var >= min_variance
    return TestResult(
        test_name="embedding_variance",
        model_name=model_name,
        column=column,
        status="pass" if ok else "fail",
        message=f"mean per-dimension variance={mean_var:.6g} (min {min_variance:g})",
    )


def _embedding_duplicates(
    model_name: str, table_ref: str, adapter: WarehouseAdapter, arg: Any
) -> TestResult:
    """Exact-duplicate-vector rate (redundant copies / total) must be
    <= `max_rate` (default 0). Duplicates usually mean a cache or join bug rather
    than genuinely identical documents."""
    opts = _require_dict("embedding_duplicates", arg)
    column = opts["column"]
    max_rate = float(opts.get("max_rate", 0.0))

    vectors, _nulls, _total = _load_vectors(adapter, table_ref, column)
    if not vectors:
        return _pass("embedding_duplicates", model_name, column, "no non-null vectors")

    counts = Counter(tuple(v) for v in vectors)
    duplicate_copies = sum(count - 1 for count in counts.values() if count > 1)
    rate = duplicate_copies / len(vectors)
    ok = rate <= max_rate
    return TestResult(
        test_name="embedding_duplicates",
        model_name=model_name,
        column=column,
        status="pass" if ok else "fail",
        message=(
            f"{duplicate_copies}/{len(vectors)} duplicate vector copies "
            f"(rate {rate:.3f}, max {max_rate:.3f})"
        ),
    )


def _embedding_outliers(
    model_name: str, table_ref: str, adapter: WarehouseAdapter, arg: Any
) -> TestResult:
    """Fraction of vectors whose L2 distance from the centroid exceeds
    `z` standard deviations (default 3) must be <= `max_rate` (default 0)."""
    opts = _require_dict("embedding_outliers", arg)
    column = opts["column"]
    max_rate = float(opts.get("max_rate", 0.0))
    z = float(opts.get("z", 3.0))

    vectors, _nulls, _total = _load_vectors(adapter, table_ref, column)
    if len(vectors) < 3:
        return _pass("embedding_outliers", model_name, column, "need >= 3 vectors")
    dims = _consistent_dimensions(vectors)
    if dims is None:
        return _inconsistent("embedding_outliers", model_name, column)

    n = len(vectors)
    centroid = [math.fsum(v[i] for v in vectors) / n for i in range(dims)]
    distances = [
        math.sqrt(math.fsum((v[i] - centroid[i]) ** 2 for i in range(dims)))
        for v in vectors
    ]
    mean_d = math.fsum(distances) / n
    std_d = math.sqrt(math.fsum((d - mean_d) ** 2 for d in distances) / n)
    cutoff = mean_d + z * std_d
    outliers = 0 if std_d == 0.0 else sum(1 for d in distances if d > cutoff)
    rate = outliers / n
    ok = rate <= max_rate
    return TestResult(
        test_name="embedding_outliers",
        model_name=model_name,
        column=column,
        status="pass" if ok else "fail",
        message=(
            f"{outliers}/{n} vectors beyond {z:g}σ of the centroid "
            f"(rate {rate:.3f}, max {max_rate:.3f})"
        ),
    )


def _pass(test_name: str, model_name: str, column: str, message: str) -> TestResult:
    return TestResult(
        test_name=test_name,
        model_name=model_name,
        column=column,
        status="pass",
        message=message,
    )


def _inconsistent(test_name: str, model_name: str, column: str) -> TestResult:
    return TestResult(
        test_name=test_name,
        model_name=model_name,
        column=column,
        status="fail",
        message=(
            "vectors have inconsistent dimensionality; add an `embedding_valid` "
            "check with `dimensions` to locate the offending rows"
        ),
    )


def _fuzzy_contains(needle: str, haystack: str, min_score: float) -> bool:
    """True if `needle` approximately appears in `haystack` (stdlib difflib).

    Slides difflib's ratio over haystack windows the size of needle; cheap enough
    for demo/real use, upgradeable to rapidfuzz later.
    """
    import difflib

    if not needle:
        return True
    window = len(needle)
    step = max(1, window // 4)
    best = 0.0
    for i in range(0, max(1, len(haystack) - window + 1), step):
        chunk = haystack[i : i + window]
        score = difflib.SequenceMatcher(None, needle, chunk).ratio()
        if score >= min_score:
            return True
        best = max(best, score)
    return best >= min_score
