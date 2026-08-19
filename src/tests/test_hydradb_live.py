"""Local graph-node smoke. Requires explicit local URI/token, never hosted credentials."""

from __future__ import annotations

import os
import unittest
from uuid import uuid4

from context_memory.ingestion.fakes import InMemoryGraphManifestStore
from context_memory.client.hydradb_http import HydraHttpTransport
from context_memory.ingestion.graph_writer import GraphWriter
from context_memory.core.graph import GraphNode, GraphRelationship, GraphWritePlan

URI = os.environ.get("CONTEXT_MEMORY_HYDRADB_URL")
TOKEN = os.environ.get("CONTEXT_MEMORY_HYDRADB_TOKEN")
DATABASE = os.environ.get("CONTEXT_MEMORY_HYDRADB_DATABASE", "default")


@unittest.skipUnless(URI and TOKEN, "requires local CONTEXT_MEMORY_HYDRADB_URL and CONTEXT_MEMORY_HYDRADB_TOKEN")
class HydraDbLiveTests(unittest.TestCase):
    def test_graph_write_and_causal_read(self) -> None:
        seed, context = int(uuid4().int % 1_000_000_000), f"live-context-{uuid4().hex}"
        plan = GraphWritePlan(context, f"live-plan-{seed}", (GraphNode(seed, "Session", f"session-{seed}", {"context_id": context, "session_id": "live"}), GraphNode(seed + 1, "Turn", f"turn-{seed}", {"context_id": context, "source_chunk_id": f"chunk-{seed}"})), (GraphRelationship(seed + 2, "HAS_TURN", f"has-turn-{seed}", seed, seed + 1, "Session", "Turn", {"context_id": context, "turn_index": 0}),))
        transport = HydraHttpTransport(URI or "", TOKEN or "", graph_id=DATABASE)
        bookmarks = GraphWriter(InMemoryGraphManifestStore(), transport).write(plan)
        rows = transport.read("MATCH (s:Session {id: $id})-[:HAS_TURN]->(t:Turn) RETURN t.source_chunk_id AS chunk_id", {"id": seed}, bookmarks[-1] if bookmarks else None)
        self.assertEqual(rows, [{"chunk_id": f"chunk-{seed}"}])
