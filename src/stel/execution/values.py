from __future__ import annotations

import json
from typing import Any


def scalarize(value: Any) -> Any:
    if isinstance(value, dict | list):
        return json.dumps(value, default=str)
    return value
