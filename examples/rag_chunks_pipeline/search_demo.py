from __future__ import annotations

import json
from pathlib import Path

from stel.config import load_project
from stel.profile import resolve_profile
from stel.retrieval import StoreRole, create_store

PROJECT_DIR = Path(__file__).parent


def main() -> None:
    project, _, _ = load_project(PROJECT_DIR)
    profile = resolve_profile(project, PROJECT_DIR)
    if profile.retrieval is None:
        raise RuntimeError("retrieval profile is not configured")
    store = create_store(
        profile.retrieval.stores["local"],
        project_name=project.name,
        target_name=profile.target_name,
        alias="local",
        # This demo queries, so it wants the serving budget.
        role=StoreRole.SERVE,
    )
    collection = store.physical_collection("document_chunks")
    with store:
        rows = store.text_search(
            collection,
            "leave policy",
            text_field="text",
            limit=3,
            columns=["chunk_id", "source_uri", "text", "_score"],
        )
    print(json.dumps(rows.to_pylist(), indent=2))


if __name__ == "__main__":
    main()
