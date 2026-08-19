CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE evidence_chunks (
    chunk_id TEXT PRIMARY KEY,
    context_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_external_id TEXT NOT NULL,
    source_record_id TEXT NOT NULL,
    session_id TEXT,
    actor_id TEXT,
    actor_role TEXT,
    raw_text TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (context_id, source_record_id)
);

CREATE TABLE graph_id_registry (
    graph_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    node_kind TEXT NOT NULL,
    context_id TEXT NOT NULL,
    logical_key TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (graph_id >= 0),
    UNIQUE (node_kind, context_id, logical_key)
);

CREATE TABLE ingestion_jobs (
    job_id TEXT PRIMARY KEY,
    chunk_id TEXT NOT NULL UNIQUE REFERENCES evidence_chunks(chunk_id) ON DELETE RESTRICT,
    context_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN (
        'pending_graph', 'pending_embeddings', 'verifying', 'completed',
        'retryable_failed', 'terminal_failed', 'manual_repair'
    )),
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    last_verified_state TEXT,
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE memory_embeddings (
    embedding_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    context_id TEXT NOT NULL,
    subject_kind TEXT NOT NULL CHECK (subject_kind IN ('fact', 'chunk')),
    subject_id TEXT NOT NULL,
    source_chunk_id TEXT NOT NULL REFERENCES evidence_chunks(chunk_id) ON DELETE RESTRICT,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK (dimensions > 0),
    embedding vector NOT NULL,
    embedded_content_hash TEXT NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (context_id, subject_kind, subject_id, model_name, model_version)
);

CREATE INDEX evidence_chunks_context_record_idx
    ON evidence_chunks (context_id, source_record_id);

CREATE INDEX ingestion_jobs_context_state_idx
    ON ingestion_jobs (context_id, state);

CREATE INDEX memory_embeddings_context_subject_idx
    ON memory_embeddings (context_id, subject_kind, is_active);
