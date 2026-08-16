"""PostgreSQL persistence adapter for immutable chunks and graph ID allocation."""

from __future__ import annotations

import json

from context_memory.core.errors import GraphPayloadConflictError, ImmutableRecordConflictError
from context_memory.core.graph import GraphNode, GraphRelationship, GraphWritePlan
from context_memory.core.models import Chunk, ExtractedMemoryCandidate, SourceDescriptor


class PostgresChunkStore:
    def __init__(self, connection: object) -> None:
        self._connection = connection

    def put(self, chunk: Chunk) -> Chunk:
        with self._connection.transaction():
            with self._connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT context_id, source_type, source_external_id, source_record_id,
                           session_id, actor_id, actor_role, raw_text, content_hash,
                           occurred_at, metadata
                    FROM evidence_chunks WHERE chunk_id = %s AND context_id = %s FOR UPDATE
                    """,
                    (chunk.chunk_id, chunk.context_id),
                )
                existing = cursor.fetchone()
                if existing is not None:
                    expected = (
                        chunk.context_id,
                        chunk.source.source_type,
                        chunk.source.source_external_id,
                        chunk.source_record_id,
                        chunk.session_id,
                        chunk.actor_id,
                        chunk.actor_role,
                        chunk.raw_text,
                        chunk.content_hash,
                        chunk.occurred_at,
                        dict(chunk.metadata),
                    )
                    if existing != expected:
                        raise ImmutableRecordConflictError(
                            f"chunk_id {chunk.chunk_id} has different immutable content"
                        )
                    return chunk
                cursor.execute(
                    """
                    INSERT INTO evidence_chunks (
                        chunk_id, context_id, source_type, source_external_id,
                        source_record_id, session_id, actor_id, actor_role,
                        raw_text, content_hash, occurred_at, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.context_id,
                        chunk.source.source_type,
                        chunk.source.source_external_id,
                        chunk.source_record_id,
                        chunk.session_id,
                        chunk.actor_id,
                        chunk.actor_role,
                        chunk.raw_text,
                        chunk.content_hash,
                        chunk.occurred_at,
                        json.dumps(dict(chunk.metadata), sort_keys=True),
                    ),
                )
                cursor.execute(
                    """
                    INSERT INTO ingestion_jobs (job_id, chunk_id, context_id, state)
                    VALUES (%s, %s, %s, 'pending_graph')
                    """,
                    (f"job:{chunk.chunk_id}", chunk.chunk_id, chunk.context_id),
                )
        return chunk

    def get(self, context_id: str, chunk_id: str) -> Chunk | None:
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT context_id, source_type, source_external_id, source_record_id,
                       session_id, actor_id, actor_role, raw_text, content_hash,
                       occurred_at, metadata
                FROM evidence_chunks WHERE chunk_id = %s AND context_id = %s
                """,
                (chunk_id, context_id),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return Chunk(
            chunk_id=chunk_id,
            context_id=row[0],
            source=SourceDescriptor(source_type=row[1], source_external_id=row[2]),
            source_record_id=row[3],
            session_id=row[4],
            actor_id=row[5],
            actor_role=row[6],
            raw_text=row[7],
            content_hash=row[8],
            occurred_at=row[9],
            metadata=row[10],
        )

    def allocate_graph_id(self, node_kind: str, context_id: str, logical_key: str) -> int:
        with self._connection.transaction():
            with self._connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO graph_id_registry (node_kind, context_id, logical_key)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (node_kind, context_id, logical_key)
                    DO UPDATE SET logical_key = EXCLUDED.logical_key
                    RETURNING graph_id
                    """,
                    (node_kind, context_id, logical_key),
                )
                row = cursor.fetchone()
        if row is None:
            raise RuntimeError("graph ID allocation returned no row")
        return int(row[0])


class PostgresExtractionStore:
    """Append-only SQL audit trail for the deterministic M4 baseline."""

    def __init__(self, connection: object) -> None:
        self._connection = connection

    def record(
        self,
        *,
        attempt_id: str,
        chunk: Chunk,
        extractor_name: str,
        extractor_version: str,
        accepted: object,
        rejected: object,
    ) -> None:
        accepted_items = tuple(accepted)  # type: ignore[arg-type]
        rejected_items = tuple(rejected)  # type: ignore[arg-type]
        with self._connection.transaction():
            with self._connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO extraction_attempts (
                        attempt_id, chunk_id, context_id, extractor_name, extractor_version,
                        extractor_kind, quality_status, input_content_hash, accepted_count, rejected_count
                    ) VALUES (%s, %s, %s, %s, %s, 'deterministic_fixture', 'baseline_only', %s, %s, %s)
                    ON CONFLICT (attempt_id) DO NOTHING
                    """,
                    (attempt_id, chunk.chunk_id, chunk.context_id, extractor_name, extractor_version,
                     chunk.content_hash, len(accepted_items), len(rejected_items)),
                )
                for candidate in accepted_items:
                    assert isinstance(candidate, ExtractedMemoryCandidate)
                    cursor.execute(
                        """
                        INSERT INTO extracted_memory_candidates (
                            attempt_id, candidate_id, memory_text, memory_type, scope_type, scope_id,
                            source_record_id, source_start, source_end, confidence, observed_at,
                            valid_from, valid_to, entities
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                        ON CONFLICT (attempt_id, candidate_id) DO NOTHING
                        """,
                        (
                            attempt_id, candidate.candidate_id, candidate.text, candidate.memory_type.value,
                            candidate.scope_type.value, candidate.scope_id, candidate.source_span.source_record_id,
                            candidate.source_span.source_start, candidate.source_span.source_end, candidate.confidence,
                            candidate.temporal.observed_at, candidate.temporal.valid_from, candidate.temporal.valid_to,
                            json.dumps([
                                {"surface": entity.surface, "entity_type": entity.entity_type}
                                for entity in candidate.entities
                            ], sort_keys=True),
                        ),
                    )

                for ordinal, item in enumerate(rejected_items):
                    draft = item.draft
                    cursor.execute(
                        """
                        INSERT INTO rejected_extraction_candidates (
                            attempt_id, ordinal, candidate_id, rejection_reason, draft
                        ) VALUES (%s, %s, %s, %s, %s::jsonb)
                        ON CONFLICT (attempt_id, ordinal) DO NOTHING
                        """,
                        (
                            attempt_id, ordinal, item.candidate_id, item.reason,
                            json.dumps({
                                "candidate_id": draft.candidate_id, "text": draft.text,
                                "source_start": draft.source_start, "source_end": draft.source_end,
                                "confidence": draft.confidence,
                                "memory_type": draft.memory_type.value if draft.memory_type else None,
                                "scope_type": draft.scope_type.value if draft.scope_type else None,
                                "scope_id": draft.scope_id,
                            }, sort_keys=True),
                        ),
                    )


class PostgresGraphManifestStore:
    """Reject changed payload before a non-atomic local HydraDB write begins."""

    def __init__(self, connection: object) -> None:
        self._connection = connection

    def register(self, plan: GraphWritePlan) -> None:
        with self._connection.transaction():
            with self._connection.cursor() as cursor:
                for record in plan.records():
                    kind = "node" if isinstance(record, GraphNode) else "relationship"
                    payload_hash = plan.payload_hash(record)
                    cursor.execute(
                        "SELECT graph_id, payload_hash FROM graph_write_manifests WHERE record_kind = %s AND context_id = %s AND logical_key = %s FOR UPDATE",
                        (kind, plan.context_id, record.logical_key),
                    )
                    existing = cursor.fetchone()
                    if existing is not None:
                        if existing != (record.graph_id, payload_hash):
                            raise GraphPayloadConflictError(f"{kind} {record.logical_key} has a different immutable graph payload")
                        continue
                    cursor.execute(
                        "INSERT INTO graph_write_manifests (record_kind, context_id, logical_key, graph_id, payload_hash, payload) VALUES (%s, %s, %s, %s, %s, %s::jsonb)",
                        (kind, plan.context_id, record.logical_key, record.graph_id, payload_hash, json.dumps(self._payload(record), sort_keys=True)),
                    )

    @staticmethod
    def _payload(record: GraphNode | GraphRelationship) -> dict[str, object]:
        if isinstance(record, GraphNode):
            return {"label": record.label, "id": record.graph_id, "properties": dict(record.properties)}
        return {"type": record.relationship_type, "id": record.graph_id, "source": record.source_id, "destination": record.destination_id, "source_label": record.source_label, "destination_label": record.destination_label, "properties": dict(record.properties)}
