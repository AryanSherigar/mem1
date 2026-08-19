"""Structured logging and latency measurement utilities for HydraDB Context Memory Engine."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Generator


def setup_logging(
    level: int = logging.INFO,
    format_str: str = "%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
) -> None:
    """Configures application-wide structured logging."""
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Returns a logger for the given module name."""
    return logging.getLogger(name)


@contextmanager
def timed_operation(
    logger: logging.Logger,
    operation_name: str,
    extra: dict[str, Any] | None = None,
    log_level: int = logging.INFO,
) -> Generator[dict[str, Any], None, None]:
    """Context manager measuring execution latency and logging outcome with diagnostics.

    Usage:
        with timed_operation(logger, "llm_fact_extraction", {"chunk_id": chunk.chunk_id}) as ctx:
            result = do_work()
            ctx["facts_extracted"] = len(result)
    """
    context: dict[str, Any] = extra.copy() if extra else {}
    start_time = time.perf_counter()
    logger.log(
        log_level,
        "[START] %s | %s",
        operation_name,
        " ".join(f"{k}={v}" for k, v in context.items()) if context else "no extra metadata",
    )
    try:
        yield context
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        meta_str = f" | {' '.join(f'{k}={v}' for k, v in context.items())}" if context else ""
        logger.log(log_level, "[DONE] %s in %.2f ms%s", operation_name, elapsed_ms, meta_str)
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        meta_str = f" | {' '.join(f'{k}={v}' for k, v in context.items())}" if context else ""
        logger.exception(
            "[FAIL] %s FAILED after %.2f ms%s | error=%s: %s",
            operation_name,
            elapsed_ms,
            meta_str,
            type(e).__name__,
            str(e),
        )
        raise
