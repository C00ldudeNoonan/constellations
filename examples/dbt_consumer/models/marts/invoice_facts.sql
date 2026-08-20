{{ config(materialized='table') }}

SELECT
    invoice_id,
    vendor,
    CAST(issue_date AS DATE) AS issue_date,
    DATE_TRUNC('month', CAST(issue_date AS DATE)) AS issue_month,
    currency,
    total,
    CASE
        WHEN total > 5000 THEN 'large'
        WHEN total > 1000 THEN 'medium'
        ELSE 'small'
    END AS size_bucket
FROM {{ source('dbt_ml_invoice_pipeline', 'raw_invoices') }}
