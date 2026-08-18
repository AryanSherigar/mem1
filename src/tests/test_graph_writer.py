from __future__ import annotations

import unittest

from context_memory.ingestion.fakes import InMemoryGraphManifestStore, RecordingGraphTransport
from context_memory.core.errors import ContractValidationError, GraphPayloadConflictError
from context_memory.ingestion.graph_writer import GraphWriter
from context_memory.core.graph import GraphNode, GraphRelationship, GraphWritePlan


class GraphWriterTests(unittest.TestCase):
    def plan(self) -> GraphWritePlan:
        context = "context-001"
        return GraphWritePlan(
            context,
            "plan-001",
            (
                GraphNode(1, "Session", "session-001", {"context_id": context, "session_id": "session-001"}),
                GraphNode(2, "Turn", "turn-001", {"context_id": context, "source_chunk_id": "chunk-001"}),
            ),
            (GraphRelationship(3, "HAS_TURN", "has-turn-001", 1, 2, "Session", "Turn", {"context_id": context, "turn_index": 0}),),
        )

    def test_writes_supported_unwind_batches_with_stable_keys(self) -> None:
        transport = RecordingGraphTransport()
        bookmarks = GraphWriter(InMemoryGraphManifestStore(), transport).write(self.plan())
        self.assertEqual(bookmarks, ("bookmark-1", "bookmark-2", "bookmark-3"))
        self.assertEqual(len(transport.writes), 3)
        self.assertIn("UNWIND $rows AS row MERGE", transport.writes[0][0])
        self.assertIn("SET n:Session", transport.writes[0][0])
        self.assertIn("MATCH (s:Session {id: row.source_id}), (d:Turn {id: row.destination_id})", transport.writes[2][0])
        self.assertRegex(transport.writes[0][2], r"^context-memory-[0-9a-f]{64}$")

    def test_manifest_rejects_changed_replay_before_transport(self) -> None:
        manifest, transport = InMemoryGraphManifestStore(), RecordingGraphTransport()
        writer, plan = GraphWriter(manifest, transport), self.plan()
        writer.write(plan)
        changed = GraphWritePlan(plan.context_id, plan.plan_key, (GraphNode(1, "Session", "session-001", {"context_id": plan.context_id, "session_id": "changed"}),), ())
        with self.assertRaises(GraphPayloadConflictError):
            writer.write(changed)
        self.assertEqual(len(transport.writes), 3)

    def test_rejects_unsafe_property_and_id_collision(self) -> None:
        with self.assertRaises(ContractValidationError):
            GraphNode(1, "Fact", "fact-001", {"context_id": "context-001", "x) SET n.pwned": "bad"})
        with self.assertRaises(ContractValidationError):
            GraphWritePlan("context-001", "plan", (GraphNode(1, "Fact", "fact-001", {"context_id": "context-001"}),), (GraphRelationship(1, "ABOUT", "about-001", 1, 1, "Fact", "Entity", {"context_id": "context-001"}),))
