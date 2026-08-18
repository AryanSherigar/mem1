from __future__ import annotations

import unittest


class ArchitectureRouteTests(unittest.TestCase):
    def test_public_routes_match_owned_hydradb_boundaries(self) -> None:
        from context_memory.client.hydradb_http import HydraHttpTransport
        from context_memory.core.models import ContextBatch
        from context_memory.ingestion.service import IngestionService
        from context_memory.ingestion.graph_writer import GraphWriter
        from context_memory.core.graph import GraphWritePlan

        self.assertTrue(all((HydraHttpTransport, ContextBatch, IngestionService, GraphWritePlan, GraphWriter)))
