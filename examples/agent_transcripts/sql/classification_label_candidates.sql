-- Candidate expected labels (#456, #329 phase 3). Never read by `eval:`; a
-- human promotes these, and nothing here promotes itself at any confidence.
--
-- Joined back to `correction_inputs` because an `llm:` model emits its id and
-- its declared fields only -- the provenance and the candidate ids live on
-- the input row, which is where they belong.
SELECT
  i.input_id,
  i.session_id,
  i.harness,
  i.exchange_ordinal,
  i.id_space,
  -- The first candidate id is the cited one where there was a citation, so
  -- it is the agent's own best claim about what it was answering from. A
  -- reviewer confirms it; nothing here decides.
  json_extract_string(i.candidate_context_ids, '$[0]') AS context_id,
  d.corrected_label AS expected_label,
  i.observed_at
FROM {{ ref('drafted_corrections') }} AS d
JOIN {{ ref('correction_inputs') }} AS i USING (input_id)
WHERE d.corrected_label IS NOT NULL
  AND trim(d.corrected_label) <> ''
  AND json_extract_string(i.candidate_context_ids, '$[0]') IS NOT NULL
