-- Closes the round trip (#177): dbt-ml built `raw_vendors` (extraction), this
-- mart is native dbt SQL, `flagged_invoices` is a dbt-ml transform that read
-- this mart back via `dbt_ref('invoice_facts')`, and this final mart is native
-- dbt SQL again — dbt-ml -> dbt -> dbt-ml -> dbt, all in one `dbt build`.
select
    vendor,
    total_spend
from {{ ref('flagged_invoices') }}
where flagged
order by total_spend desc
