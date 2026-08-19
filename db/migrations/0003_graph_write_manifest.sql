CREATE TABLE graph_write_manifests (
    record_kind TEXT NOT NULL CHECK (record_kind IN ('node', 'relationship')),
    context_id TEXT NOT NULL,
    logical_key TEXT NOT NULL,
    graph_id BIGINT NOT NULL CHECK (graph_id >= 0),
    payload_hash TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (record_kind, context_id, logical_key),
    UNIQUE (graph_id)
);

CREATE INDEX graph_write_manifests_context_idx ON graph_write_manifests (context_id, record_kind);
