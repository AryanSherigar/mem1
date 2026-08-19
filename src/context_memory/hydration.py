"""Startup hydration logic to prepopulate active indices."""

from __future__ import annotations

import numpy as np

from context_memory.client.hydradb_http import HydraHttpTransport
# Use absolute imports correctly from src
import sys
import os
# Adjust as necessary; we assume db.embedding_index is accessible 
# wait, it is in src/db/embedding_index.py, so `from db.embedding_index` should work if PYTHONPATH=src
from db.embedding_index import EmbeddingIndex, EntityNameIndex


class HydrationManager:
    """Queries Postgres and HydraDB on startup to populate NumPy indexes."""
    
    def __init__(
        self,
        pg_connection: object,
        hydra_client: HydraHttpTransport,
        embedding_index: EmbeddingIndex | None = None,
        entity_name_index: EntityNameIndex | None = None,
        embedder: object | None = None,
    ) -> None:
        self._pg = pg_connection
        self._hydra = hydra_client
        self._embedding_index = embedding_index if embedding_index is not None else EmbeddingIndex()
        self._entity_name_index = entity_name_index if entity_name_index is not None else EntityNameIndex()
        self._embedder = embedder

    def hydrate_context(self, context_id: str) -> None:
        """Loads all active facts and entities for a given context into the indexes."""
        
        # 1. Hydrate EmbeddingIndex from pgvector
        with self._pg.cursor() as cursor:
            cursor.execute(
                """
                SELECT subject_id, values 
                FROM memory_embeddings 
                WHERE context_id = %s AND subject_kind = 'fact' AND is_active = true
                """,
                (context_id,)
            )
            for row in cursor.fetchall():
                fact_id = row[0]
                vector_data = row[1]
                # Convert pgvector string format to numpy array if it comes as string, 
                # but psycopg usually handles it if properly registered, or it comes as a list of floats.
                if isinstance(vector_data, str):
                    vector = np.array([float(x) for x in vector_data.strip("[]").split(",")], dtype=np.float32)
                else:
                    vector = np.array(vector_data, dtype=np.float32)
                
                self._embedding_index.add_vector(fact_id=str(fact_id), vector=vector, haystack_id=context_id)
        
        # 2. Hydrate EntityNameIndex from HydraDB
        cypher = """
        MATCH (e:Entity {context_id: $context_id}) 
        RETURN e.logical_key as id, e.canonical_name as name, e.entity_type as type
        """
        try:
            res = self._hydra.read(cypher, {"context_id": context_id}, None)
            for row in res:
                entity_id = row.get("id")
                # Remove "entity:" prefix if present for clean ID
                if entity_id and entity_id.startswith("entity:"):
                    entity_id = entity_id[7:]
                
                name = row.get("name")
                entity_type = row.get("type", "other")
                
                if entity_id and name:
                    self._entity_name_index.add(
                        entity_id=entity_id, 
                        name=name, 
                        entity_type=entity_type, 
                        haystack_id=context_id
                    )
        except Exception as e:
            # If hydra query fails (e.g., db offline), we just have an empty cache for entities.
            print(f"Failed to hydrate entities from HydraDB for {context_id}: {e}")
