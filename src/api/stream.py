import asyncio
import json
import logging
from context_memory.core.graph import GraphWritePlan

logger = logging.getLogger(__name__)

class GraphStreamer:
    def __init__(self):
        self.queues = set()
        self.loop = None

    def add_queue(self) -> asyncio.Queue:
        q = asyncio.Queue()
        self.queues.add(q)
        return q

    def remove_queue(self, q: asyncio.Queue):
        if q in self.queues:
            self.queues.remove(q)

    def push_event(self, data: dict):
        if not self.queues:
            return
            
        if self.loop is None:
            try:
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                return
                
        try:
            self.loop.call_soon_threadsafe(self._push_all, data)
        except Exception as e:
            logger.error(f"Error pushing event: {e}")

    def broadcast_plan(self, plan: GraphWritePlan):
        if not self.queues:
            return
            
        nodes_data = []
        for n in plan.nodes:
            nodes_data.append({
                "id": str(n.graph_id),
                "label": n.label,
                "properties": dict(n.properties)
            })
            
        edges_data = []
        for r in plan.relationships:
            edges_data.append({
                "id": str(r.graph_id),
                "source_id": str(r.source_id),
                "target_id": str(r.destination_id),
                "type": r.relationship_type
            })
            
        data = {
            "type": "graph_update",
            "nodes": nodes_data,
            "edges": edges_data
        }
        
        self.push_event(data)

    def _push_all(self, data):
        for q in list(self.queues):
            try:
                q.put_nowait(data)
            except Exception:
                pass

streamer = GraphStreamer()

