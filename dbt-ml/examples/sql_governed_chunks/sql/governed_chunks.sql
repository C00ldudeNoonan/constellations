-- Warehouse-native SQL transform (#143): joins chunk rows to tenant/access
-- metadata entirely inside the warehouse (no rows move through dbt-ml). The
-- result is governed *build-time* metadata; runtime retrieval must still apply
-- mandatory policy filters — this projection does not enforce authorization.
select
    c.chunk_id,
    c.doc_ref,
    c.text,
    p.tenant_id,
    p.access_groups,
    coalesce(p.is_public, false) as is_public
from {{ ref('document_chunks') }} as c
left join {{ ref('document_permissions') }} as p
  on c.doc_ref = p.doc_ref
