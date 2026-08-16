from __future__ import annotations

import json
import unittest

from context_memory.client.hydradb_http import HydraHttpError, HydraHttpTransport


class HydraHttpTransportTests(unittest.TestCase):
    def test_write_uses_local_query_endpoint_and_query_id(self) -> None:
        calls: list[tuple[str, str, dict[str, str], bytes]] = []

        def requester(method: str, url: str, headers: dict[str, str], body: bytes) -> dict[str, object]:
            calls.append((method, url, headers, body))
            return {"bookmark": "bookmark-1", "columns": [], "rows": []}

        transport = HydraHttpTransport("http://127.0.0.1:8443", "local-token", requester=requester)
        self.assertEqual(transport.write("UNWIND $rows AS row RETURN row.id", [{"id": 1}], "stable-key"), "bookmark-1")
        method, url, headers, body = calls[0]
        self.assertEqual((method, url), ("POST", "http://127.0.0.1:8443/v1/graphs/default/query"))
        self.assertEqual(headers["X-Graph-Namespace"], "default")
        self.assertEqual(headers["Authorization"], "Bearer local-token")
        self.assertEqual(json.loads(body), {"cell_id": "cell-0", "parameters": {"rows": [{"id": 1}]}, "query": "UNWIND $rows AS row RETURN row.id", "query_id": "stable-key"})

    def test_read_decodes_typed_values_and_forwards_bookmark(self) -> None:
        seen: list[dict[str, object]] = []

        def requester(_: str, __: str, ___: dict[str, str], body: bytes) -> dict[str, object]:
            seen.append(json.loads(body))
            return {"columns": ["name", "count"], "rows": [[{"type": "string", "value": "Max"}, {"type": "integer", "value": 2}]]}

        transport = HydraHttpTransport("http://local", "token", requester=requester)
        self.assertEqual(transport.read("MATCH", {"id": 1}, "bookmark-1"), [{"name": "Max", "count": 2}])
        self.assertEqual(seen[0]["bookmark"], "bookmark-1")

    def test_rejects_malformed_response(self) -> None:
        transport = HydraHttpTransport("http://local", "token", requester=lambda *_: {"columns": ["x"], "rows": [[{"type": "path", "value": {}}]]})
        with self.assertRaises(HydraHttpError):
            transport.read("MATCH", {}, None)
