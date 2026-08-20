from __future__ import annotations

import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from faker import Faker

_PRODUCTS = ["acme-cloud", "acme-mobile", "acme-cli", "acme-billing", "acme-ide"]
_PRIORITIES = ["low", "medium", "high", "urgent"]
_STATUSES = ["open", "in_progress", "waiting_customer", "resolved", "closed"]
_TEAMS = ["frontend", "platform", "billing", "integrations", "data"]
_CUSTOMER_TIERS = ["free", "team", "business", "enterprise"]

# SLA targets in hours per priority
_SLA_HOURS = {"urgent": 1, "high": 8, "medium": 24, "low": 72}


def generate_support_tickets(count: int, output_dir: Path, seed: int = 42) -> list[Path]:
    """Generate `count` synthetic support tickets as JSON files into `output_dir`.

    Each ticket has the shape a typical B2B SaaS would actually produce:
    id, product, priority, status, customer_tier, assigned_team,
    created_at / first_response_at / resolved_at timestamps, summary.
    """
    Faker.seed(seed)
    rng = random.Random(seed)
    fake = Faker()

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    now = datetime.now(UTC)

    for i in range(count):
        ticket = _make_ticket(fake, rng, i, now)
        path = output_dir / f"ticket_{i:05d}.json"
        path.write_text(json.dumps(ticket, indent=2, default=str))
        paths.append(path)
    return paths


def _make_ticket(
    fake: Faker, rng: random.Random, index: int, now: datetime
) -> dict[str, Any]:
    priority = rng.choices(_PRIORITIES, weights=[5, 10, 4, 1])[0]
    status = rng.choices(_STATUSES, weights=[3, 4, 2, 4, 7])[0]
    age_hours = rng.uniform(0.1, 200)
    created_at = now - timedelta(hours=age_hours)

    first_response_at = None
    resolved_at = None
    if status != "open":
        # First response somewhere between 0.1h and 2x the SLA
        sla = _SLA_HOURS[priority]
        first_response_hours = rng.uniform(0.1, sla * 2)
        first_response_at = created_at + timedelta(hours=first_response_hours)

    if status in {"resolved", "closed"}:
        # Resolved between first_response and now
        baseline = first_response_at or created_at
        resolved_at = baseline + timedelta(hours=rng.uniform(0.5, max(1.0, age_hours / 2)))
        if resolved_at > now:
            resolved_at = now

    return {
        "ticket_id": f"TIC-{index:05d}",
        "product": rng.choice(_PRODUCTS),
        "priority": priority,
        "status": status,
        "customer_tier": rng.choice(_CUSTOMER_TIERS),
        "assigned_team": rng.choice(_TEAMS),
        "customer_name": fake.name(),
        "customer_email": fake.email(),
        "summary": fake.sentence(nb_words=8).rstrip("."),
        "created_at": created_at.isoformat(),
        "first_response_at": first_response_at.isoformat() if first_response_at else None,
        "resolved_at": resolved_at.isoformat() if resolved_at else None,
        "sla_target_hours": _SLA_HOURS[priority],
    }
