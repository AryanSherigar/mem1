import time
from fastapi import FastAPI, Request
from api.routes import router
from context_memory.core.logging import get_logger, setup_logging

setup_logging()
logger = get_logger("api.server")

app = FastAPI(title="Context Memory API")

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
