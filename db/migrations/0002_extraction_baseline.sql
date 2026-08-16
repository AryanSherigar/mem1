CREATE TABLE extraction_attempts (
    attempt_id TEXT PRIMARY KEY,
    chunk_id TEXT NOT NULL REFERENCES evidence_chunks(chunk_id) ON DELETE RESTRICT,
    context_id TEXT NOT NULL,
    extractor_name TEXT NOT NULL CHECK (extractor_name = 'deterministic-fixture'),
    extractor_version TEXT NOT NULL CHECK (extractor_version = 'v1'),
    extractor_kind TEXT NOT NULL CHECK (extractor_kind = 'deterministic_fixture'),
    quality_status TEXT NOT NULL CHECK (quality_status = 'baseline_only'),
    input_content_hash TEXT NOT NULL,
    accepted_count INTEGER NOT NULL CHECK (accepted_count >= 0),
    rejected_count INTEGER NOT NULL CHECK (rejected_count >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE extracted_memory_candidates (
    attempt_id TEXT NOT NULL REFERENCES extraction_attempts(attempt_id) ON DELETE RESTRICT,
    candidate_id TEXT NOT NULL,
    memory_text TEXT NOT NULL,
    memory_type TEXT NOT NULL CHECK (memory_type IN ('semantic', 'procedural')),
    scope_type TEXT NOT NULL CHECK (scope_type IN ('chat', 'session', 'user')),
    scope_id TEXT NOT NULL,
    source_record_id TEXT NOT NULL,
    source_start INTEGER NOT NULL CHECK (source_start >= 0),
    source_end INTEGER NOT NULL CHECK (source_end > source_start),
    confidence DOUBLE PRECISION NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    observed_at TIMESTAMPTZ NOT NULL,
    valid_from TIMESTAMPTZ,
    valid_to TIMESTAMPTZ,
    entities JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (attempt_id, candidate_id),
    CHECK (valid_from IS NULL OR valid_to IS NULL OR valid_from <= valid_to)
);

CREATE TABLE rejected_extraction_candidates (
    attempt_id TEXT NOT NULL REFERENCES extraction_attempts(attempt_id) ON DELETE RESTRICT,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    candidate_id TEXT NOT NULL,
    rejection_reason TEXT NOT NULL,
    draft JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (attempt_id, ordinal)
);

CREATE INDEX extraction_attempts_context_chunk_idx ON extraction_attempts (context_id, chunk_id);
