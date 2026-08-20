import time
import asyncio
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from context_memory.core.logging import get_logger, setup_logging
from context_memory.ingestion.graph_writer import GraphWriter
from api.stream import streamer

setup_logging()
logger = get_logger("api.server")

# Monkey Patch GraphWriter to intercept writes
original_write = GraphWriter.write
def patched_write(self, plan):
    res = original_write(self, plan)
    try:
        streamer.broadcast_plan(plan)
    except Exception as e:
        logger.error(f"Error in broadcast: {e}")
    return res
GraphWriter.write = patched_write

app = FastAPI(title="Context Memory API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000.0
    logger.info(
        "HTTP %s %s -> status=%d duration=%.2fms",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response

app.include_router(router)

@app.get("/health")
def read_health():
    return {"status": "ok"}

@app.get("/v1/memory/stream")
async def stream_graph(request: Request):
    if streamer.loop is None:
        streamer.loop = asyncio.get_running_loop()
        
    q = streamer.add_queue()
    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(q.get(), timeout=1.0)
                    yield f"data: {json.dumps(data)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            streamer.remove_queue(q)
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")
