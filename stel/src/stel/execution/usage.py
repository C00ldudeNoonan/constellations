from __future__ import annotations


def add_provider_usage(
    totals: dict[str, int | float],
    usage: dict[str, int | float],
) -> None:
    for key, value in usage.items():
        totals[key] = totals.get(key, 0) + value
