from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from email.message import EmailMessage
from email.utils import format_datetime, make_msgid
from pathlib import Path

from faker import Faker

_SUBJECTS = [
    "Can't log in",
    "Billing question",
    "Feature request: export to CSV",
    "Integration with Slack failing",
    "API rate limit hit",
    "Need to add a seat",
    "How do I export data?",
    "Dashboard not loading",
    "Password reset email didn't arrive",
    "Pricing for 50+ users",
]


def generate_support_emails(count: int, output_dir: Path, seed: int = 42) -> list[Path]:
    """Generate `count` synthetic support emails as .eml files."""
    Faker.seed(seed)
    rng = random.Random(seed)
    fake = Faker()

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    now = datetime.now(UTC)

    for i in range(count):
        sent_at = now - timedelta(hours=rng.uniform(0.1, 240))
        msg = EmailMessage()
        msg["From"] = f"{fake.name()} <{fake.email()}>"
        msg["To"] = "support@acme.example"
        msg["Subject"] = rng.choice(_SUBJECTS)
        msg["Date"] = format_datetime(sent_at)
        msg["Message-ID"] = make_msgid(domain="acme.example")
        body = "\n\n".join(fake.paragraphs(nb=rng.randint(1, 3)))
        msg.set_content(
            f"Hi team,\n\n{body}\n\nThanks,\n{msg['From'].split('<')[0].strip()}\n"
        )
        path = output_dir / f"email_{i:05d}.eml"
        path.write_bytes(bytes(msg))
        paths.append(path)
    return paths
