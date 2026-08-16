"""Generic record-to-chunk ingestion boundary; no source-specific behavior."""

from __future__ import annotations

from dataclasses import dataclass

from context_memory.ingestion.ports import ChunkStore
from context_memory.core.models import ContextBatch
from context_memory.core.validation import chunk_from_record


@dataclass(frozen=True)
class IngestionResult:
    context_id: str
    chunk_ids: tuple[str, ...]

    @property
    def accepted_record_count(self) -> int:
        return len(self.chunk_ids)


class IngestionPartialFailure(RuntimeError):
    """Some chunks are durable; caller must replay the complete batch safely."""

    def __init__(self, persisted_chunk_ids: tuple[str, ...], cause: Exception) -> None:
        self.persisted_chunk_ids = persisted_chunk_ids
        self.cause = cause
        super().__init__(f"ingestion stopped after {len(persisted_chunk_ids)} durable chunk(s): {cause}")


class IngestionService:
    """Shared source-neutral ingestion entrypoint."""

    def __init__(self, chunk_store: ChunkStore) -> None:
        self._chunk_store = chunk_store

    def ingest(self, batch: ContextBatch) -> IngestionResult:
        persisted: list[str] = []
        try:
            for record in batch.records:
                chunk = chunk_from_record(batch, record)
                self._chunk_store.put(chunk)
                persisted.append(chunk.chunk_id)
        except Exception as error:
            raise IngestionPartialFailure(tuple(persisted), error) from error
        return IngestionResult(context_id=batch.context_id, chunk_ids=tuple(persisted))
