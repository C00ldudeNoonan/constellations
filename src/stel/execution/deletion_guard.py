"""Snapshot-bound authorization for guarded extraction deletions (#741)."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Mapping, Sequence

from ..adapters.base import StateValue
from ..sources import DocumentRef


def deletion_approval_token(
    *,
    project: str,
    target: str,
    model: str,
    state: Mapping[str, StateValue],
    documents: Sequence[DocumentRef],
) -> str:
    """Bind approval to the complete discovered snapshot and prior state.

    Only the digest is returned; raw keys and content are never emitted.
    """
    payload = {
        "domain": "stel-extraction-deletions/v1",
        "project": project,
        "target": target,
        "model": model,
        "state": sorted(
            (key, value.input_fingerprint, value.code_version) for key, value in state.items()
        ),
        "source": sorted((doc.document_id, doc.content_hash) for doc in documents),
    }
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def is_valid_approval(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def approval_matches(expected: str, supplied: str | None) -> bool:
    return supplied is not None and hmac.compare_digest(expected, supplied)
