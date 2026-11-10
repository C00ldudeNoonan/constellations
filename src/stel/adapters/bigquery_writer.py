"""Non-expiring, cooperative BigQuery writer exclusion (#741).

The tables.insert API arbitrates first acquisition, including an empty dataset.
No participating process can steal or expire a claim. Release requires every
submitted job to be terminal; an ambiguous submission strands the claim. Manual
recovery must stop the old process and finish its jobs before deleting the table.
This protocol cannot exclude older binaries or direct warehouse clients.
"""

from __future__ import annotations

import os
import re
import socket
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, PositiveInt, ValidationError

from ..interrupts import check_interruption, defer_interrupts
from .base import WRITER_CLAIM_TABLE, AdapterError

_OWNER_LABEL = "stel_writer"
_RECOVERY_LABEL = "stel_recovery"
_SETTLE_SECONDS = 10.0


class _ClaimDetails(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    version: Literal[1]
    hostname: str
    pid: PositiveInt
    started_at: AwareDatetime
    execution_project: str


def _owner(table: Any) -> str | None:
    label = table.labels.get(_OWNER_LABEL)
    return label if isinstance(label, str) and re.fullmatch(r"[0-9a-f]{32}", label) else None


def _details(table: Any) -> _ClaimDetails | None:
    if not table.description:
        return None
    try:
        return _ClaimDetails.model_validate_json(table.description)
    except ValidationError:
        return None  # Legacy, malformed or unknown metadata cannot authorize recovery.


def _created_at(table: Any) -> datetime | None:
    created = getattr(table, "created", None)
    return created if isinstance(created, datetime) else None


def _job_id_prefix(owner: str) -> str:
    return f"stel_{owner}_"


def _read_claim(client: Any, table_id: str) -> Any | None:
    from google.api_core.exceptions import NotFound

    try:
        return client.get_table(table_id)
    except NotFound:
        return None
    except Exception:
        raise AdapterError("BigQuery writer claim could not be read.") from None


def writer_claim_status(client: Any, table_id: str) -> dict[str, str | int | None]:
    table = _read_claim(client, table_id)
    if table is None:
        return {"status": "absent"}
    details = _details(table)
    created_at = _created_at(table)
    return {
        "status": "recovery_started" if table.labels.get(_RECOVERY_LABEL) else "claimed",
        "run_id": _owner(table),
        "created_at": created_at.isoformat() if created_at is not None else None,
        "hostname": details.hostname if details else None,
        "pid": details.pid if details else None,
        "started_at": details.started_at.isoformat() if details else None,
        "execution_project": details.execution_project if details else None,
    }


def _check_job_page(payload: dict[str, Any], job_prefix: str) -> None:
    if payload.get("unreachable"):
        raise AdapterError("BigQuery could not list jobs in every location.")
    for item in payload.get("jobs", ()):
        job_id = item.get("jobReference", {}).get("jobId")
        if not job_id or not isinstance(job_id, str):
            raise AdapterError("BigQuery job listing did not include job identities.")
        if job_id.startswith(job_prefix) and item.get("state") != "DONE":
            raise AdapterError("A BigQuery job for this writer claim is still active.")


def _require_terminal_jobs(
    client: Any, *, project: str, owner: str, created_at: datetime
) -> None:
    job_prefix = _job_id_prefix(owner)
    try:
        pages = client.list_jobs(
            project=project, all_users=True, min_creation_time=created_at
        ).pages
        for page in pages:
            _check_job_page(page.raw_page, job_prefix)
    except AdapterError:
        raise
    except Exception:
        raise AdapterError("BigQuery could not verify the writer's jobs are terminal.") from None


def release_writer_claim(client: Any, table_id: str, *, run_id: str) -> None:
    """Release after the operator establishes process and API-request quiescence."""
    table = _read_claim(client, table_id)
    if table is None or _owner(table) != run_id:
        raise AdapterError("BigQuery writer claim changed or is absent. Check writer-claim status.")
    if table.labels.get(_RECOVERY_LABEL):
        raise AdapterError(
            "BigQuery writer claim already has a recovery attempt. Inspect it first."
        )
    details = _details(table)
    created_at = _created_at(table)
    if details is None or not details.execution_project or created_at is None:
        raise AdapterError("BigQuery writer claim lacks recovery metadata. Follow manual recovery.")
    if not table.etag or table.expires is not None:
        raise AdapterError("BigQuery writer claim cannot be fenced for recovery. Inspect it first.")
    _require_terminal_jobs(
        client, project=details.execution_project, owner=run_id, created_at=created_at
    )
    # Claim recovery with an ETag precondition. A losing recovery must never
    # delete by table name, which a subsequent writer could already own.
    recovery = uuid4().hex
    table.labels = {**table.labels, _RECOVERY_LABEL: recovery}
    try:
        marked = client.update_table(table, ["labels"], retry=None)
        if _owner(marked) != run_id or marked.labels.get(_RECOVERY_LABEL) != recovery:
            raise AdapterError("BigQuery did not confirm the writer claim recovery marker.")
        client.delete_table(table_id, not_found_ok=False, retry=None)
    except Exception:
        raise AdapterError(
            "BigQuery writer claim recovery could not be confirmed. Check writer-claim status "
            "and account for the recovery request before retrying."
        ) from None


class BigQueryWriterClaim:
    def __init__(
        self,
        client: Any,
        bigquery: Any,
        project: str,
        dataset: str,
        location: str | None,
        *,
        execution_project: str,
    ) -> None:
        self.client = client
        self.table_id = f"{project}.{dataset}.{WRITER_CLAIM_TABLE}"
        self.owner = uuid4().hex
        self._bigquery = bigquery
        self._location = location
        self._details = _ClaimDetails(
            version=1,
            hostname=socket.gethostname(),
            pid=os.getpid(),
            started_at=datetime.now(UTC),
            execution_project=execution_project,
        )
        self._jobs: dict[int, Any] = {}
        self._ambiguous_submissions = 0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        try:
            dataset = self._bigquery.Dataset(self.table_id.rsplit(".", 1)[0])
            dataset.location = self._location
            self.client.create_dataset(dataset, exists_ok=True)
            table = self._bigquery.Table(self.table_id)
            table.labels = {_OWNER_LABEL: self.owner}
            table.description = self._details.model_dump_json()
            # A dataset's default expiration applies at creation. Remove it
            # before permitting any work; failed acquisition never deletes an
            # existing claim, including one whose API response was lost.
            created = self.client.create_table(table, exists_ok=False)
            created.expires = None
            updated = self.client.update_table(created, ["expires"])
            if updated.expires is not None:
                raise AdapterError("BigQuery writer claim must not expire")
        except Exception:
            created = self._existing_creation_time()
            when = f" Claim created at {created.isoformat()}." if created is not None else ""
            raise AdapterError(
                f"BigQuery writer claim unavailable.{when} Another writer may be active. "
                "Use stel writer-claim status and inspect outstanding jobs before recovery."
            ) from None

    def _existing_creation_time(self) -> datetime | None:
        try:
            return _created_at(_read_claim(self.client, self.table_id))
        except AdapterError:
            return None  # The original acquisition error remains the diagnosis.

    def verify(self) -> None:
        try:
            table = self.client.get_table(self.table_id)
            if (
                _owner(table) != self.owner
                or table.expires is not None
                or table.labels.get(_RECOVERY_LABEL)
            ):
                raise AdapterError("BigQuery writer claim changed")
        except Exception:
            raise AdapterError(
                "BigQuery writer claim could not be verified; writes refused"
            ) from None

    def _begin_operation(self) -> None:
        with self._lock:
            check_interruption()
            self.verify()
            check_interruption()
            if self._ambiguous_submissions:
                raise AdapterError("BigQuery job submission is uncertain; further writes refused")
            self._ambiguous_submissions += 1

    def mutate_metadata[T](self, operation: Callable[[], T]) -> T:
        """A failed table API response can also hide an outstanding mutation."""
        with defer_interrupts():
            self._begin_operation()
            result = operation()
            with self._lock:
                self._ambiguous_submissions -= 1
        check_interruption()
        return result

    def submit(self, operation: Callable[[], Any]) -> Any:
        with defer_interrupts():
            self._begin_operation()
            # A lost response can hide a job accepted by the server.
            job = operation()
            with self._lock:
                self._jobs[id(job)] = job
                self._ambiguous_submissions -= 1
        check_interruption()
        return job

    @property
    def job_id_prefix(self) -> str:
        return _job_id_prefix(self.owner)

    def completed(self, job: Any) -> None:
        """Forget a job only after its terminal result has been confirmed."""
        with self._lock:
            self._jobs.pop(id(job), None)

    def _cancel_and_confirm(self) -> bool:
        deadline = time.monotonic() + _SETTLE_SECONDS
        pending = list(self._jobs.values())
        cancel_requested: set[int] = set()
        while pending:
            for job in pending.copy():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                if job.state != "DONE":
                    job.reload(client=self.client, retry=None, timeout=remaining)
                if job.state == "DONE":
                    pending.remove(job)
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                if id(job) not in cancel_requested:
                    job.cancel(client=self.client, retry=None, timeout=remaining)
                    cancel_requested.add(id(job))
            if pending:
                time.sleep(min(0.25, max(0, deadline - time.monotonic())))
        return True

    def release(self, *, interrupted: bool) -> None:
        try:
            with self._lock:
                self.verify()
                terminal = (
                    self._cancel_and_confirm()
                    if interrupted
                    else all(job.done() for job in self._jobs.values())
                )
                if self._ambiguous_submissions or not terminal:
                    raise AdapterError("BigQuery writer still has unconfirmed jobs")
                # A retry by name could delete a new holder's table after a lost response.
                self.client.delete_table(self.table_id, not_found_ok=False, retry=None)
        except Exception:
            raise AdapterError(
                "BigQuery writer claim retained because completion or ownership is uncertain. "
                "Stop the old writer and account for all outstanding API requests before "
                "using stel writer-claim release."
            ) from None
