# ADR-0058 — BigQuery writer recovery requires quiescence

- **Status** Accepted. Amends the recovery procedure in ADR-0048.
- **Prompted by** [#759](https://github.com/C00ldudeNoonan/constellations/issues/759) (ALE-620).

Two interrupted runs stranded dataset writer claims in astrolabe. SIGTERM can
unwind through the writer context while a process is still alive. An OOM kill
leaves only warehouse evidence and whatever the operator can establish about
the holder and its requests.

For SIGTERM and SIGINT, stel records the outcome of any in-flight submission,
refuses new BigQuery submissions, and tries to cancel outstanding jobs. It
releases the claim only after confirming terminal jobs and unambiguous API
outcomes.

For recovery after process death, the claim records the holder and execution
project at acquisition. The owner label also serves as the run ID and is part
of every claimed job ID, which identifies matching jobs even when BigQuery
redacts job configuration. The operator must establish that the old writer
cannot resume and every API request is accounted for. A job listing cannot
establish the outcome of a submission that has yet to become visible, or of a
table metadata operation. The CLI independently checks the recorded project's
job listing before using the claim table's ETag to elect one recovery attempt.
The elected attempt deletes without retrying the deletion by table name.

A per-job journal in the claim description would add a BigQuery metadata
update to every submission. BigQuery limits these updates to five operations
per table per ten seconds, and the journal would still leave open the outcome
of a table API request interrupted in flight. Claims keep the non-expiring
exclusion of ADR-0048. [BigQuery quotas](https://docs.cloud.google.com/bigquery/quotas#standard_tables)
