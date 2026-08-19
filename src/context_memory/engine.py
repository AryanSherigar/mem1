from __future__ import annotations

from datetime import datetime, timezone
import uuid
import json

from context_memory.core.logging import get_logger, timed_operation
from context_memory.core.models import ContextBatch, ContextRecord, SourceDescriptor
from context_memory.ingestion.orchestrator import IngestionOrchestrator
from context_memory.ingestion.sources.chat import adapt_chat_turn
from context_memory.retrieval import HybridRetrievalEngine
from context_memory.core.llm_client import LLMClient
from context_memory.hydration import HydrationManager
from concurrent.futures import ThreadPoolExecutor

logger = get_logger(__name__)

class MemoryEngine:
    def __init__(
        self,
        orchestrator: IngestionOrchestrator,
        retrieval_engine: HybridRetrievalEngine,
        llm_client: LLMClient,
        pg_connection: object,
        hydration_manager: HydrationManager | None = None
    ):
        self._orchestrator = orchestrator
        self._retrieval_engine = retrieval_engine
        self._llm = llm_client
        self._pg = pg_connection
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._hydration_manager = hydration_manager

    def add_turn_async(self, context_id: str, session_id: str, role: str, content: str, timestamp: datetime) -> None:
        with timed_operation(logger, "memory_engine.add_turn_async", {"context_id": context_id, "session_id": session_id, "role": role}) as ctx:
            # 1. Record to conversation buffer
            with self._pg.cursor() as cursor:
                cursor.execute(
                    "SELECT COALESCE(MAX(turn_index), -1) + 1 FROM conversation_buffer WHERE context_id = %s AND session_id = %s",
                    (context_id, session_id)
                )
                turn_index = cursor.fetchone()[0]
                
                cursor.execute(
                    """
                    INSERT INTO conversation_buffer (context_id, session_id, turn_index, role, content, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (context_id, session_id, turn_index, role, content, timestamp)
                )
                try:
                    self._pg.commit()
                except AttributeError:
                    pass
                ctx["turn_index"] = turn_index

            # 2. Ingest via orchestrator
            payload = {
                "context_id": context_id,
                "session_id": session_id,
                "role": role,
                "content": content,
                "timestamp": timestamp,
            }
            batch = adapt_chat_turn(payload, f"ingest-{uuid.uuid4()}")
            # Async execution off the critical path
            self._executor.submit(self._run_orchestrator_safe, batch)

    def _run_orchestrator_safe(self, batch: ContextBatch) -> None:
        try:
            with timed_operation(logger, "memory_engine.bg_ingest", {"batch_id": batch.ingestion_id, "context_id": batch.context_id}):
                self._orchestrator.run_batch(batch)
        except Exception as e:
            logger.error("Background ingestion failed for batch %s: %s", batch.ingestion_id, e, exc_info=True)

    def search_memories(self, context_id: str, query: str, question_date: datetime) -> str:
        with timed_operation(logger, "memory_engine.search_memories", {"context_id": context_id, "query_len": len(query)}):
            return self._retrieval_engine.retrieve_and_answer(context_id, query, question_date)

    def generate_reply(self, context_id: str, session_id: str, user_message: str) -> str:
        with timed_operation(logger, "memory_engine.generate_reply", {"context_id": context_id, "session_id": session_id}) as ctx:
            now = datetime.now(timezone.utc)
            
            # Ingest user turn
            self.add_turn_async(context_id, session_id, "user", user_message, now)
            
            # Retrieve context and answer
            reply = self.search_memories(context_id, user_message, now)
            
            # Ingest assistant turn
            self.add_turn_async(context_id, session_id, "assistant", reply, now)
            ctx["reply_len"] = len(reply)
            return reply

    def hydrate(self, context_id: str) -> None:
        """Pre-populates in-memory caches from Postgres and HydraDB."""
        if self._hydration_manager:
            with timed_operation(logger, "memory_engine.hydrate", {"context_id": context_id}):
                self._hydration_manager.hydrate_context(context_id)
