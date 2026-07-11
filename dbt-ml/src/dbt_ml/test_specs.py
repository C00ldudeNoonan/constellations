from __future__ import annotations

import math
import re
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
}
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
        if self.name != "relationships":
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


def relationship_test_targets(specs: list[Any]) -> set[str]:
    targets: set[str] = set()
    for spec in specs:
        target = parse_test_spec(spec).relationship_target
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
    allowed = required | (optional or set())
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - allowed)
    if missing:
        raise TestSpecError(f"Test '{name}' is missing required options: {missing}")
    if unknown:
        raise TestSpecError(f"Test '{name}' has unknown options: {unknown}")
    return value


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
