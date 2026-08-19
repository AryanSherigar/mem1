"""Disposable local Docker smoke for the checked-in HydraDB HTTP integration."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_memory.ingestion.fakes import InMemoryGraphManifestStore
from context_memory.client.hydradb_http import HydraHttpTransport
from context_memory.ingestion.graph_writer import GraphWriter
from context_memory.core.graph import GraphNode, GraphRelationship, GraphWritePlan

IMAGE = "ghcr.io/hydra-db/hydradb:latest"
TOKEN = "context-memory-local-smoke-token-32b"
PORTS = (17687, 18443, 19090)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="context-memory-hydradb-") as directory:
        root = Path(directory)
        (root / "store").mkdir()
        (root / "cache").mkdir()
        (root / "auth-token").write_text(f"{TOKEN}\n", encoding="utf-8")
        command = [
            "docker", "run", "--rm", "-d", "--user", f"{os.getuid()}:{os.getgid()}",
            "-p", f"{PORTS[0]}:7687", "-p", f"{PORTS[1]}:8443", "-p", f"{PORTS[2]}:9090",
            "-v", f"{root}:/data", "-e", "CLOUD_PROVIDER=local", "-e", "LOCAL_PATH=/data/store",
            "-e", "GRAPH_NAMESPACE=default", "-e", "GRAPH_ID=default", "-e", "GRAPH_CELL_ID=cell-0",
            "-e", "GRAPH_CELLS=cell-0", "-e", "GRAPH_NODE_ID=node-0",
            "-e", "GRAPH_BOLT_NODE_ADDRESSES=node-0=127.0.0.1:7687",
            "-e", "GRAPH_ADVERTISED_BOLT_ADDR=127.0.0.1:7687", "-e", "GRAPH_DATA_CACHE_DIR=/data/cache",
            "-e", "GRAPH_AUTH_TOKEN_FILE=/data/auth-token", "-e", "GRAPH_ALLOW_PLAINTEXT=true",
            "-e", "RUST_MIN_STACK=33554432", IMAGE,
        ]
        container = subprocess.check_output(command, text=True).strip()
        try:
            _wait_ready(container)
            _round_trip()
            print("hydradb-http-smoke-ok")
            return 0
        finally:
            subprocess.run(["docker", "stop", container], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _wait_ready(container: str) -> None:
    for _ in range(120):
        probe = subprocess.run(["curl", "-fsS", f"http://127.0.0.1:{PORTS[2]}/readyz"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if probe.returncode == 0:
            return
        if subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", container], capture_output=True, text=True).stdout.strip() != "true":
            logs = subprocess.run(["docker", "logs", container], capture_output=True, text=True).stderr
            raise RuntimeError(f"graph-node stopped before ready: {logs}")
        time.sleep(0.25)
    raise TimeoutError("graph-node did not become ready")


def _round_trip() -> None:
    context = "local-smoke"
    plan = GraphWritePlan(
        context, "local-smoke-plan",
        (GraphNode(1, "Session", "session:smoke", {"context_id": context, "session_id": "smoke"}), GraphNode(2, "Turn", "turn:smoke", {"context_id": context, "source_chunk_id": "chunk:smoke"})),
        (GraphRelationship(3, "HAS_TURN", "has-turn:smoke", 1, 2, "Session", "Turn", {"context_id": context, "turn_index": 0}),),
    )
    transport = HydraHttpTransport(f"http://127.0.0.1:{PORTS[1]}", TOKEN)
    bookmarks = GraphWriter(InMemoryGraphManifestStore(), transport).write(plan)
    rows = transport.read("MATCH (s:Session {id: $id})-[:HAS_TURN]->(t:Turn) RETURN t.source_chunk_id AS chunk_id", {"id": 1}, bookmarks[-1] if bookmarks else None)
    if rows != [{"chunk_id": "chunk:smoke"}]:
        raise AssertionError(rows)


if __name__ == "__main__":
    sys.exit(main())
