"""Non-expiring, cooperative BigQuery writer exclusion (#741).

The tables.insert API arbitrates first acquisition, including an empty dataset.
No participating process can steal or expire a claim. Release requires every
submitted job to be terminal; an ambiguous submission strands the claim. Manual
recovery must stop the old process and finish its jobs before deleting the table.
This protocol cannot exclude older binaries or direct warehouse clients.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from .base import WRITER_CLAIM_TABLE, AdapterError


class BigQueryWriterClaim:
    def __init__(
        self, client: Any, bigquery: Any, project: str, dataset: str, location: str | None
    ) -> None:
        self.client = client
        self.table_id = f"{project}.{dataset}.{WRITER_CLAIM_TABLE}"
        self.owner = uuid4().hex
        self._bigquery = bigquery
        self._location = location
        self._jobs: dict[int, Any] = {}
        self._ambiguous_submissions = 0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        try:
            dataset = self._bigquery.Dataset(self.table_id.rsplit(".", 1)[0])
            dataset.location = self._location
            self.client.create_dataset(dataset, exists_ok=True)
            table = self._bigquery.Table(self.table_id)
            table.labels = {"stel_writer": self.owner}
            # A dataset's default expiration applies at creation. Remove it
            # before permitting any work; failed acquisition never deletes an
            # existing claim, including one whose API response was lost.
            created = self.client.create_table(table, exists_ok=False)
            created.expires = None
            updated = self.client.update_table(created, ["expires"])
            if updated.expires is not None:
                raise AdapterError("BigQuery writer claim must not expire")
        except Exception:
            raise AdapterError(
                "BigQuery writer claim unavailable. Another writer may be active; "
                "inspect stel_writer_claim and outstanding jobs before operator recovery."
            ) from None

    def verify(self) -> None:
        try:
            table = self.client.get_table(self.table_id)
            if table.labels.get("stel_writer") != self.owner or table.expires is not None:
                raise AdapterError("BigQuery writer claim changed")
        except Exception:
            raise AdapterError(
                "BigQuery writer claim could not be verified; writes refused"
            ) from None

    def _begin_operation(self) -> None:
        with self._lock:
            self.verify()
            if self._ambiguous_submissions:
                raise AdapterError("BigQuery job submission is uncertain; further writes refused")
            self._ambiguous_submissions += 1

    def mutate_metadata[T](self, operation: Callable[[], T]) -> T:
        """A failed table API response can also hide an outstanding mutation."""
        self._begin_operation()
        result = operation()
        with self._lock:
            self._ambiguous_submissions -= 1
        return result

    def submit(self, operation: Callable[[], Any]) -> Any:
        self._begin_operation()
        # Keep the counter incremented if submission raises: the server may
        # still have accepted it. A timer or cancellation request is no proof.
        job = operation()
        with self._lock:
            self._jobs[id(job)] = job
            self._ambiguous_submissions -= 1
        return job

    def completed(self, job: Any) -> None:
        """Forget a job only after its terminal result has been confirmed."""
        with self._lock:
            self._jobs.pop(id(job), None)

    def release(self) -> None:
        try:
            with self._lock:
                self.verify()
                if self._ambiguous_submissions or any(
                    not job.done() for job in self._jobs.values()
                ):
                    raise AdapterError("BigQuery writer still has unconfirmed jobs")
                self.client.delete_table(self.table_id, not_found_ok=False)
        except Exception:
            raise AdapterError(
                "BigQuery writer claim retained because completion or ownership is uncertain. "
                "Stop the old writer and verify all its jobs are terminal before deleting "
                "stel_writer_claim; never clear a claim based on age alone."
            ) from None
