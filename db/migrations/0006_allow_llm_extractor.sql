-- Allow model-backed LLM extractors in extraction_attempts audit table
ALTER TABLE extraction_attempts DROP CONSTRAINT IF EXISTS extraction_attempts_extractor_name_check;
ALTER TABLE extraction_attempts DROP CONSTRAINT IF EXISTS extraction_attempts_extractor_kind_check;
ALTER TABLE extraction_attempts DROP CONSTRAINT IF EXISTS extraction_attempts_quality_status_check;
