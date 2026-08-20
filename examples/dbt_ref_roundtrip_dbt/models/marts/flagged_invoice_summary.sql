-- Closes the round trip (#177): stel built `raw_vendors` (extraction), this
-- mart is native dbt SQL, `flagged_invoices` is a stel transform that read
-- this mart back via `dbt_ref('invoice_facts')`, and this final mart is native
-- dbt SQL again — stel -> dbt -> stel -> dbt, all in one `dbt build`.
select
    vendor,
    total_spend
from {{ ref('flagged_invoices') }}
where flagged
order by total_spend desc
