CREATE TABLE conversation_buffer (
    buffer_id BIGSERIAL PRIMARY KEY,
    context_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX conv_buffer_lookup_idx ON conversation_buffer (context_id, session_id, turn_index DESC);

CREATE TABLE fact_search_index (
    fact_id INTEGER PRIMARY KEY,
    context_id TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    text_tsvector tsvector GENERATED ALWAYS AS (to_tsvector('english', raw_text)) STORED,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX fact_tsvector_idx ON fact_search_index USING GIN (text_tsvector);
