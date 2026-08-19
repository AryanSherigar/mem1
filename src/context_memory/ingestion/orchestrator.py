"""Milestone 8: chains extraction -> resolution -> graph write -> embeddings per chunk,
driving the job state machine from `docs/ingestion_contract_v1.md` §5.

Scope, stated plainly rather than overclaimed: this gives *safe replay of the
whole chunk pipeline*, not fine-grained per-stage resumption. Calling
`run_chunk` again after a `retryable_failed` result re-runs extraction,
resolution, graph-plan construction, the graph write, and embedding
persistence from the top — every one of those stages was already built to be
an idempotent no-op on unchanged replay (`ExtractionService`'s stable
`attempt_id`, `PostgresGraphManifestStore`, `PostgresEmbeddingStore`,
`GraphWriter`'s `MERGE` identities), so redoing the whole chunk is correct,
just not free. There is no port to re-derive "what candidates did we already
accept" from storage alone (`ExtractionStore`/`EmbeddingStore` are write/audit
paths, not read-back paths) — building that is future work if per-stage
resumption is ever needed. This matches ADR-007's framing ("replay instead of
claimed atomicity"), not a stronger guarantee than that.

Verification before `completed` is same-process evidence (every write in this
call raised nothing, plus a real `chunk_store.get` re-read) — not an
independent re-read of HydraDB or pgvector, since `EmbeddingStore` has no
`get`/`exists` port yet. Documented, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from context_memory.core.enums import IngestionJobState
from context_memory.core.errors import ContractValidationError, GraphPayloadConflictError, ImmutableRecordConflictError
from context_memory.core.models import Chunk, ContextBatch, ContextRecord, Embedding
from context_memory.core.validation import chunk_from_record
from context_memory.ingestion.extraction import ExtractionService
from context_memory.ingestion.graph_plan_builder import GraphPlanBuilder, ResolveEntity
from context_memory.ingestion.graph_writer import GraphWriter
from context_memory.ingestion.ports import ChunkStore, Embedder, EmbeddingStore, JobStore

# Errors that mean the payload/policy itself is invalid; retrying unchanged input cannot help.
_TERMINAL_ERROR_TYPES = (ContractValidationError, ImmutableRecordConflictError, GraphPayloadConflictError)


@dataclass(frozen=True)
class ChunkRunResult:
    chunk_id: str
    state: IngestionJobState
    accepted_fact_count: int = 0
    error: str | None = None


@dataclass(frozen=True)
class BatchRunResult:
    context_id: str
    results: tuple[ChunkRunResult, ...]

    @property
    def completed_count(self) -> int:
        return sum(1 for r in self.results if r.state == IngestionJobState.COMPLETED)


class IngestionOrchestrator:
    def __init__(
        self,
        *,
        chunk_store: ChunkStore,
        job_store: JobStore,
        extraction_service: ExtractionService,
        graph_plan_builder: GraphPlanBuilder,
        graph_writer: GraphWriter,
        resolve_entity: ResolveEntity,
        embedder: Embedder,
        embedding_store: EmbeddingStore,
    ) -> None:
        self._chunk_store = chunk_store
        self._job_store = job_store
        self._extraction_service = extraction_service
        self._graph_plan_builder = graph_plan_builder
        self._graph_writer = graph_writer
        self._resolve_entity = resolve_entity
        self._embedder = embedder
        self._embedding_store = embedding_store

    def run_batch(self, batch: ContextBatch) -> BatchRunResult:
        results = [self.run_record(batch, record) for record in batch.records]
        return BatchRunResult(context_id=batch.context_id, results=tuple(results))

    def run_record(self, batch: ContextBatch, record: ContextRecord) -> ChunkRunResult:
        chunk = chunk_from_record(batch, record)
        self._chunk_store.put(chunk)
        job = self._job_store.get(chunk.chunk_id)
        if job is None:
            job = self._job_store.seed(chunk.chunk_id, chunk.context_id)
        return self.run_chunk(batch, record, chunk, job.state)

    def run_chunk(
        self, batch: ContextBatch, record: ContextRecord, chunk: Chunk, current_state: IngestionJobState
    ) -> ChunkRunResult:
        if current_state == IngestionJobState.COMPLETED:
            return ChunkRunResult(chunk.chunk_id, IngestionJobState.COMPLETED)
        if current_state in (IngestionJobState.TERMINAL_FAILED, IngestionJobState.MANUAL_REPAIR):
            return ChunkRunResult(chunk.chunk_id, current_state, error="blocked: requires manual repair, not auto-retried")

        try:
            extraction = self._extraction_service.extract(batch, record, chunk)
            plan = self._graph_plan_builder.build(chunk, extraction, self._resolve_entity)
            self._graph_writer.write(plan)
            self._job_store.transition(chunk.chunk_id, IngestionJobState.PENDING_EMBEDDINGS)

            for candidate in extraction.accepted:
                vector = self._embedder.embed(candidate.text)
                self._embedding_store.put(
                    Embedding(
                        context_id=chunk.context_id, subject_kind="fact", subject_id=candidate.candidate_id,
                        source_chunk_id=chunk.chunk_id, model_name=getattr(self._embedder, "model_name", "unknown"),
                        model_version=getattr(self._embedder, "model_version", "1"), values=vector,
                        embedded_content_hash=f"sha256:{sha256(candidate.text.encode('utf-8')).hexdigest()}",
                    )
                )
            self._job_store.transition(chunk.chunk_id, IngestionJobState.VERIFYING)

            # Same-process verification, not an independent store re-read (see module docstring).
            verified_chunk = self._chunk_store.get(chunk.context_id, chunk.chunk_id)
            if verified_chunk is None or verified_chunk.content_hash != chunk.content_hash:
                raise RuntimeError(f"post-write verification failed for chunk {chunk.chunk_id}")

            self._job_store.transition(chunk.chunk_id, IngestionJobState.COMPLETED)
            return ChunkRunResult(chunk.chunk_id, IngestionJobState.COMPLETED, accepted_fact_count=len(extraction.accepted))
        except _TERMINAL_ERROR_TYPES as error:
            self._safe_transition(chunk.chunk_id, IngestionJobState.TERMINAL_FAILED, str(error))
            return ChunkRunResult(chunk.chunk_id, IngestionJobState.TERMINAL_FAILED, error=str(error))
        except Exception as error:  # transient/unclassified: safe to retry (idempotent stages, see docstring)
            self._safe_transition(chunk.chunk_id, IngestionJobState.RETRYABLE_FAILED, str(error))
            return ChunkRunResult(chunk.chunk_id, IngestionJobState.RETRYABLE_FAILED, error=str(error))

    def _safe_transition(self, chunk_id: str, state: IngestionJobState, error: str) -> None:
        try:
            self._job_store.transition(chunk_id, state, error=error)
        except Exception:
            pass  # job row may already be in a state that makes this a no-op; original error is what matters
