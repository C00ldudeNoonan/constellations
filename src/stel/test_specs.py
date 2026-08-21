from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

SUPPORTED_TESTS = {
    "not_null",
    "unique",
    "min_rows",
    "not_empty",
    "python",
    "matches_regex",
    "accepted_values",
    "accepted_range",
    "null_rate",
    "grounded_in",
    "relationships",
    "embedding_valid",
    "embedding_variance",
    "embedding_duplicates",
    "embedding_outliers",
    "column_stat",
    "cardinality",
    "outlier_rate",
    "drift",
    "golden",
    "llm_judge",
}

_COLUMN_STATS = {"mean", "min", "max", "sum", "stddev", "median", "quantile"}
_DRIFT_METRICS = {"psi", "ks", "jensen_shannon", "chi_squared"}
SUPPORTED_SEVERITIES = {"error", "warn"}

_REF_PATTERN = re.compile(r"^\s*ref\(\s*['\"]([^'\"]+)['\"]\s*\)\s*$")


class TestSpecError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedTestSpec:
    name: str
    argument: Any
    severity: str

    @property
    def relationship_target(self) -> str | None:
        return self.ref_target

    @property
    def ref_target(self) -> str | None:
        """The model this test depends on via a `to:` ref (relationships parent,
        or the drift baseline), so the DAG builds it first."""
        if self.name not in {"relationships", "drift", "golden"}:
            return None
        target = self.argument["to"]
        match = _REF_PATTERN.match(target)
        return match.group(1) if match else target.strip()


def parse_test_spec(spec: Any) -> ParsedTestSpec:
    if isinstance(spec, str):
        if spec not in SUPPORTED_TESTS:
            raise TestSpecError(
                f"Unknown test '{spec}'. Supported: {sorted(SUPPORTED_TESTS)}"
            )
        if spec != "not_empty":
            raise TestSpecError(f"Test '{spec}' requires an argument")
        return ParsedTestSpec(name=spec, argument=None, severity="error")

    if not isinstance(spec, dict):
        raise TestSpecError(f"Unsupported test spec: {spec!r}")

    body = dict(spec)
    severity = body.pop("severity", "error")
    if not isinstance(severity, str) or severity not in SUPPORTED_SEVERITIES:
        raise TestSpecError(
            f"Unknown severity '{severity}'. Allowed: {sorted(SUPPORTED_SEVERITIES)}"
        )
    if len(body) != 1:
        raise TestSpecError(
            "Test spec must have exactly one test key (plus optional severity), "
            f"got: {spec!r}"
        )
    ((name, argument),) = body.items()
    if not isinstance(name, str) or name not in SUPPORTED_TESTS:
        raise TestSpecError(
            f"Unknown test '{name}'. Supported: {sorted(SUPPORTED_TESTS)}"
        )

    _validate_argument(name, argument)
    return ParsedTestSpec(name=name, argument=argument, severity=severity)


def uses_llm_judge(specs: list[Any]) -> bool:
    """Whether any test in `specs` is an `llm_judge`, which needs an `llm:`
    profile — used to preflight that requirement before discovery/build."""
    return any(parse_test_spec(spec).name == "llm_judge" for spec in specs)


def relationship_test_targets(specs: list[Any]) -> set[str]:
    """Models referenced by a test's `to:` (relationships parent or drift
    baseline), which must be built before the test runs."""
    targets: set[str] = set()
    for spec in specs:
        target = parse_test_spec(spec).ref_target
        if target is not None:
            targets.add(target)
    return targets


def _validate_argument(name: str, argument: Any) -> None:
    if name in {"not_null", "unique"}:
        _validate_columns(name, argument)
        return
    if name == "min_rows":
        if isinstance(argument, bool) or not isinstance(argument, int) or argument < 0:
            raise TestSpecError("Test 'min_rows' expects a non-negative integer")
        return
    if name == "not_empty":
        if argument is not None:
            raise TestSpecError("Test 'not_empty' does not accept an argument")
        return
    if name == "python":
        _require_nonempty_string(name, argument, "module path")
        return
    if name == "matches_regex":
        options = _options(name, argument, required={"column", "pattern"})
        _require_nonempty_string(name, options["column"], "column")
        pattern = _require_nonempty_string(name, options["pattern"], "pattern")
        try:
            re.compile(pattern)
        except re.error as e:
            raise TestSpecError(f"Test 'matches_regex' has invalid pattern: {e}") from e
        return
    if name == "accepted_values":
        options = _options(name, argument, required={"column", "values"})
        _require_nonempty_string(name, options["column"], "column")
        values = options["values"]
        if not isinstance(values, list) or not values:
            raise TestSpecError("Test 'accepted_values' expects a non-empty values list")
        return
    if name == "accepted_range":
        options = _options(
            name, argument, required={"column"}, optional={"min", "max"}
        )
        _require_nonempty_string(name, options["column"], "column")
        if "min" not in options and "max" not in options:
            raise TestSpecError("accepted_range requires at least one of: min, max")
        minimum = _optional_finite_number(name, options, "min")
        maximum = _optional_finite_number(name, options, "max")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise TestSpecError("Test 'accepted_range' requires min <= max")
        return
    if name == "null_rate":
        options = _options(name, argument, required={"column"}, optional={"max"})
        _require_nonempty_string(name, options["column"], "column")
        maximum = options.get("max", 0.0)
        rate = _finite_number(name, maximum, "max")
        if not 0.0 <= rate <= 1.0:
            raise TestSpecError("Test 'null_rate' max must be between 0 and 1")
        return
    if name == "grounded_in":
        options = _options(
            name,
            argument,
            required={"value", "source"},
            optional={"method", "min_score"},
        )
        _require_nonempty_string(name, options["value"], "value")
        _require_nonempty_string(name, options["source"], "source")
        method = options.get("method", "exact")
        if not isinstance(method, str) or method not in {"exact", "fuzzy"}:
            raise TestSpecError("Test 'grounded_in' method must be 'exact' or 'fuzzy'")
        score = _finite_number(name, options.get("min_score", 0.8), "min_score")
        if not 0.0 <= score <= 1.0:
            raise TestSpecError("Test 'grounded_in' min_score must be between 0 and 1")
        return
    if name == "embedding_valid":
        options = _options(
            name,
            argument,
            required={"column"},
            optional={"dimensions", "min_norm", "max_norm", "max_zero_rate"},
        )
        _require_nonempty_string(name, options["column"], "column")
        if "dimensions" in options:
            _positive_int(name, options["dimensions"], "dimensions")
        min_norm = _optional_finite_number(name, options, "min_norm")
        max_norm = _optional_finite_number(name, options, "max_norm")
        for label, value in (("min_norm", min_norm), ("max_norm", max_norm)):
            if value is not None and value < 0:
                raise TestSpecError(f"Test '{name}' {label} must be non-negative")
        if min_norm is not None and max_norm is not None and min_norm > max_norm:
            raise TestSpecError(f"Test '{name}' requires min_norm <= max_norm")
        _rate(name, options, "max_zero_rate")
        return
    if name == "embedding_variance":
        options = _options(name, argument, required={"column", "min_variance"})
        _require_nonempty_string(name, options["column"], "column")
        if _finite_number(name, options["min_variance"], "min_variance") < 0:
            raise TestSpecError(f"Test '{name}' min_variance must be non-negative")
        return
    if name == "embedding_duplicates":
        options = _options(name, argument, required={"column"}, optional={"max_rate"})
        _require_nonempty_string(name, options["column"], "column")
        _rate(name, options, "max_rate")
        return
    if name == "embedding_outliers":
        options = _options(
            name, argument, required={"column"}, optional={"max_rate", "z"}
        )
        _require_nonempty_string(name, options["column"], "column")
        _rate(name, options, "max_rate")
        if "z" in options and _finite_number(name, options["z"], "z") <= 0:
            raise TestSpecError(f"Test '{name}' z must be a positive number")
        return
    if name == "column_stat":
        options = _options(
            name, argument, required={"column", "stat"}, optional={"min", "max", "quantile"}
        )
        _require_nonempty_string(name, options["column"], "column")
        stat = options["stat"]
        if not isinstance(stat, str) or stat not in _COLUMN_STATS:
            raise TestSpecError(
                f"Test '{name}' stat must be one of {sorted(_COLUMN_STATS)}"
            )
        minimum = _optional_finite_number(name, options, "min")
        maximum = _optional_finite_number(name, options, "max")
        if minimum is None and maximum is None:
            raise TestSpecError(f"Test '{name}' requires at least one of: min, max")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise TestSpecError(f"Test '{name}' requires min <= max")
        if stat == "quantile":
            if "quantile" not in options:
                raise TestSpecError(f"Test '{name}' stat 'quantile' requires a quantile")
            q = _finite_number(name, options["quantile"], "quantile")
            if not 0.0 <= q <= 1.0:
                raise TestSpecError(f"Test '{name}' quantile must be between 0 and 1")
        elif "quantile" in options:
            raise TestSpecError(
                f"Test '{name}' quantile only applies when stat is 'quantile'"
            )
        return
    if name == "cardinality":
        options = _options(
            name,
            argument,
            required={"column"},
            optional={"min", "max", "min_ratio", "max_ratio"},
        )
        _require_nonempty_string(name, options["column"], "column")
        for key in ("min", "max"):
            if key in options:
                value = options[key]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise TestSpecError(
                        f"Test '{name}' {key} must be a non-negative integer"
                    )
        min_ratio = _rate(name, options, "min_ratio") if "min_ratio" in options else None
        max_ratio = _rate(name, options, "max_ratio") if "max_ratio" in options else None
        if not any(key in options for key in ("min", "max", "min_ratio", "max_ratio")):
            raise TestSpecError(
                f"Test '{name}' requires at least one of: min, max, min_ratio, max_ratio"
            )
        if "min" in options and "max" in options and options["min"] > options["max"]:
            raise TestSpecError(f"Test '{name}' requires min <= max")
        if min_ratio is not None and max_ratio is not None and min_ratio > max_ratio:
            raise TestSpecError(f"Test '{name}' requires min_ratio <= max_ratio")
        return
    if name == "outlier_rate":
        options = _options(
            name, argument, required={"column"}, optional={"method", "k", "max_rate"}
        )
        _require_nonempty_string(name, options["column"], "column")
        method = options.get("method", "iqr")
        if not isinstance(method, str) or method not in {"iqr", "zscore"}:
            raise TestSpecError(f"Test '{name}' method must be 'iqr' or 'zscore'")
        if "k" in options and _finite_number(name, options["k"], "k") <= 0:
            raise TestSpecError(f"Test '{name}' k must be a positive number")
        _rate(name, options, "max_rate")
        return
    if name == "drift":
        options = _options(
            name,
            argument,
            required={"column", "to", "max"},
            optional={"field", "metric", "bins"},
        )
        _require_nonempty_string(name, options["column"], "column")
        _require_nonempty_string(name, options["to"], "to")
        if "field" in options:
            _require_nonempty_string(name, options["field"], "field")
        metric = options.get("metric", "psi")
        if not isinstance(metric, str) or metric not in _DRIFT_METRICS:
            raise TestSpecError(
                f"Test '{name}' metric must be one of {sorted(_DRIFT_METRICS)}"
            )
        max_value = _finite_number(name, options["max"], "max")
        if max_value < 0:
            raise TestSpecError(f"Test '{name}' max must be non-negative")
        if "bins" in options:
            bins = options["bins"]
            if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
                raise TestSpecError(f"Test '{name}' bins must be an integer >= 2")
        return
    if name == "golden":
        options = _options(
            name,
            argument,
            required={"to", "key"},
            optional={"columns", "tolerance", "exhaustive"},
        )
        _require_nonempty_string(name, options["to"], "to")
        _require_nonempty_string(name, options["key"], "key")
        if "columns" in options:
            columns = options["columns"]
            if not isinstance(columns, list) or not columns or any(
                not isinstance(c, str) or not c.strip() for c in columns
            ):
                raise TestSpecError(
                    f"Test '{name}' columns must be a non-empty list of column names"
                )
        if "tolerance" in options:
            tolerance = options["tolerance"]
            if not isinstance(tolerance, dict) or not tolerance:
                raise TestSpecError(f"Test '{name}' tolerance must be a non-empty mapping")
            for col, tol in tolerance.items():
                if not isinstance(col, str) or not col.strip():
                    raise TestSpecError(f"Test '{name}' tolerance keys must be column names")
                if _finite_number(name, tol, f"tolerance[{col}]") < 0:
                    raise TestSpecError(f"Test '{name}' tolerance values must be non-negative")
        if "exhaustive" in options and not isinstance(options["exhaustive"], bool):
            raise TestSpecError(f"Test '{name}' exhaustive must be a boolean")
        return
    if name == "llm_judge":
        options = _options(
            name,
            argument,
            required={"column", "criterion"},
            optional={"sample_size", "min_pass_rate", "seed", "max_output_tokens"},
        )
        _require_nonempty_string(name, options["column"], "column")
        _require_nonempty_string(name, options["criterion"], "criterion")
        if "sample_size" in options:
            size = options["sample_size"]
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise TestSpecError(f"Test '{name}' sample_size must be a positive integer")
        if "min_pass_rate" in options:
            rate = _finite_number(name, options["min_pass_rate"], "min_pass_rate")
            if not 0.0 <= rate <= 1.0:
                raise TestSpecError(f"Test '{name}' min_pass_rate must be between 0 and 1")
        if "seed" in options and (
            isinstance(options["seed"], bool) or not isinstance(options["seed"], int)
        ):
            raise TestSpecError(f"Test '{name}' seed must be an integer")
        if "max_output_tokens" in options:
            tokens = options["max_output_tokens"]
            if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
                raise TestSpecError(
                    f"Test '{name}' max_output_tokens must be a positive integer"
                )
        return
    if name == "relationships":
        if not isinstance(argument, dict) or not all(
            key in argument for key in ("column", "to")
        ):
            raise TestSpecError(
                "relationships requires 'column', 'to' (parent ref), and exactly "
                "one of 'field' or 'to_field' (parent column)"
            )
        options = _options(
            name,
            argument,
            required={"column", "to"},
            optional={"field", "to_field"},
        )
        _require_nonempty_string(name, options["column"], "column")
        _require_nonempty_string(name, options["to"], "to")
        fields = [key for key in ("field", "to_field") if key in options]
        if len(fields) != 1:
            raise TestSpecError(
                "relationships requires 'column', 'to' (parent ref), and exactly "
                "one of 'field' or 'to_field' (parent column)"
            )
        _require_nonempty_string(name, options[fields[0]], fields[0])


def _validate_columns(name: str, value: Any) -> None:
    if isinstance(value, str):
        _require_nonempty_string(name, value, "column")
        return
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(column, str) or not column.strip() for column in value)
    ):
        raise TestSpecError(
            f"Test '{name}' expects a column name or non-empty list of column names"
        )


def _options(
    name: str,
    value: Any,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TestSpecError(f"Test '{name}' expects a mapping of options")
    non_string_keys = [key for key in value if not isinstance(key, str)]
    if non_string_keys:
        raise TestSpecError(f"Test '{name}' option names must be strings")
    typed_value = {
        key: item for key, item in value.items() if isinstance(key, str)
    }
    allowed = required | (optional or set())
    missing = sorted(required - set(typed_value))
    unknown = sorted(set(typed_value) - allowed)
    if missing:
        raise TestSpecError(f"Test '{name}' is missing required options: {missing}")
    if unknown:
        raise TestSpecError(f"Test '{name}' has unknown options: {unknown}")
    return typed_value


def _require_nonempty_string(name: str, value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TestSpecError(f"Test '{name}' {label} must be a non-empty string")
    return value


def _optional_finite_number(
    name: str, options: dict[str, Any], key: str
) -> float | None:
    if key not in options:
        return None
    return _finite_number(name, options[key], key)


def _finite_number(name: str, value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TestSpecError(f"Test '{name}' {label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise TestSpecError(f"Test '{name}' {label} must be a finite number")
    return number


def _positive_int(name: str, value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TestSpecError(f"Test '{name}' {label} must be a positive integer")
    return value


def _rate(name: str, options: dict[str, Any], key: str) -> float:
    """A fraction option in [0, 1], defaulting to 0.0 when absent."""
    if key not in options:
        return 0.0
    rate = _finite_number(name, options[key], key)
    if not 0.0 <= rate <= 1.0:
        raise TestSpecError(f"Test '{name}' {key} must be between 0 and 1")
    return rate


def declared_accepted_values(specs: Sequence[Any]) -> dict[str, list[list[Any]]]:
    """Every explicit `accepted_values` list, grouped by column.

    Used to keep a derived enum check from duplicating a hand-written one, and
    to notice when the two disagree (issue #304). A column may carry more than
    one such test, and each of them runs — so each is collected. Keeping only
    the last would hide a conflicting earlier declaration behind a matching
    later one while the conflicting check still executed.
    """
    out: dict[str, list[list[Any]]] = {}
    for spec in specs:
        try:
            parsed = parse_test_spec(spec)
        except TestSpecError:
            # Invalid specs are reported by the compiler's own validation; this
            # helper must not raise a second, less specific error.
            continue
        if parsed.name != "accepted_values" or not isinstance(parsed.argument, dict):
            continue
        column = parsed.argument.get("column")
        values = parsed.argument.get("values")
        if isinstance(column, str) and isinstance(values, list):
            out.setdefault(column, []).append(values)
    return out


def has_model_tests(model: Any) -> bool:
    """Whether a model has any check to run — declared or derived.

    An enum field carries an `accepted_values` check with no `tests:` entry to
    see (issue #304), so anything deciding "does this model get tested" has to
    ask here rather than looking at `model.tests` directly. Two things do: the
    `stel test` selection loop, and the warehouse-capability preflight, which
    must know a schema test is coming *before* the warehouse is mutated.
    """
    if model.tests:
        return True
    return any(getattr(field, "values", None) for field in model.fields)


def enum_test_specs(
    fields: Sequence[Any], declared: Mapping[str, list[list[Any]]]
) -> list[dict[str, Any]]:
    """Derive an `accepted_values` check for every field declaring a value set.

    The declaration on the field is the single source of truth (issue #304), so
    there is no hand-typed list here to drift from the prompt or the provider
    schema. A column a user already wrote an explicit check for is skipped —
    theirs runs, not two of them.
    """
    return [
        {"accepted_values": {"column": field.name, "values": list(field.values)}}
        for field in fields
        if getattr(field, "values", None) and field.name not in declared
    ]


def enum_test_drift(
    fields: Sequence[Any], declared: Mapping[str, list[list[Any]]]
) -> list[tuple[str, list[Any], list[Any]]]:
    """Explicit accepted_values lists that disagree with a field's declared set.

    Returns (field name, declared values, the disagreeing explicit list) per
    offending declaration — every one of them, since every one of them runs.
    """
    out: list[tuple[str, list[Any], list[Any]]] = []
    for field in fields:
        values = getattr(field, "values", None)
        if not values:
            continue
        for explicit in declared.get(field.name, []):
            if set(explicit) != set(values):
                out.append((field.name, list(values), explicit))
    return out
