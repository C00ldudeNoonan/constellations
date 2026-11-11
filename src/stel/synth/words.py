"""Small, deterministic vocabulary for synthetic example documents."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

# A fixed epoch makes the same seed reproducible on different days and machines.
SYNTHETIC_NOW = datetime(2025, 1, 1, tzinfo=UTC)
_FIRST = ("Avery", "Jordan", "Morgan", "Riley", "Sam", "Taylor", "Casey", "Alex")
_LAST = ("Bennett", "Chen", "Diaz", "Foster", "Kim", "Patel", "Reed", "Singh")
_COMPANY = ("Harbor", "Maple", "Northstar", "Orchard", "Summit", "Willow")
_WORDS = (
    "analysis", "project", "system", "team", "customer", "service", "report", "model",
    "design", "data", "review", "process", "support", "change", "workflow", "result",
    "clear", "reliable", "practical", "shared", "improve", "build", "deliver", "plan",
)


class SyntheticText:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def name(self) -> str:
        return f"{self.rng.choice(_FIRST)} {self.rng.choice(_LAST)}"

    def email(self) -> str:
        return f"{self.rng.choice(_FIRST).lower()}.{self.rng.choice(_LAST).lower()}@example.com"

    def company(self) -> str:
        return f"{self.rng.choice(_COMPANY)} {self.rng.choice(('Labs', 'Group', 'Works'))}"

    def sentence(self, nb_words: int) -> str:
        words = self.rng.choices(_WORDS, k=nb_words)
        return " ".join(words).capitalize() + "."

    def paragraphs(self, nb: int) -> list[str]:
        return [
            " ".join(self.sentence(self.rng.randint(6, 12)) for _ in range(3))
            for _ in range(nb)
        ]

    def catch_phrase(self) -> str:
        return self.sentence(4).rstrip(".")

    def bs(self) -> str:
        return self.sentence(3).rstrip(".")

    def recent_date(self) -> str:
        return (SYNTHETIC_NOW - timedelta(days=self.rng.randint(0, 365))).date().isoformat()
