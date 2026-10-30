SELECT r.document_id AS evaluation_document, s.canonical_id AS subject_canonical_id,
       r.relation_type, o.canonical_id AS object_canonical_id
FROM {{ ref('relation_eval_assertions') }} r
LEFT JOIN {{ ref('relation_eval_mentions') }} s ON r.subject_mention_id = s.entity_id
LEFT JOIN {{ ref('relation_eval_mentions') }} o ON r.object_mention_id = o.entity_id
WHERE r.status = 'asserted'
