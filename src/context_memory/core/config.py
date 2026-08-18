"""Environment-driven configuration for the M5/M7 provider-neutral adapters (ADR-027).

Deliberately narrow: only the two bounded-LLM roles (`ingestion.ports.
EntityResolutionModel`, `TemporalUpdateModel`) and the embedding model. Graph
transport configuration (`CONTEXT_MEMORY_HYDRADB_URL`/`_TOKEN`) and the SQL
connection string stay where they already are, read directly at call sites
(`tests/test_hydradb_live.py`, `tests/test_postgres_persistence.py`) — this
module does not centralize them, to avoid moving working, tested behavior.

Deliberately does NOT auto-load `.env` on import. `main` branch's version of
this module did (`load_dotenv()` at module scope) — that is an import-time
side effect: merely importing this module (e.g. `unittest discover` importing
every test file to enumerate cases, including a live-gated one it never runs)
silently repopulates `FIREWORKS_API_KEY` into the process environment from
disk, defeating any `skipUnless(os.environ.get(...))` gate and risking a real
billed call from a plain test run. Same posture `test_hydradb_live.py` already
takes for `CONTEXT_MEMORY_HYDRADB_TOKEN`: credentials come from the runtime
environment the caller explicitly set up (`source src/.env`, an exported var,
or a deploy-time secret), never from an implicit file read triggered by import.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from context_memory.core.llm_client import LLMClient


@dataclass(frozen=True)
class Config:
    """Model allocation for the bounded entity-resolution and temporal-update ports."""

    # Default provider/model: Fireworks + deepseek-v4-flash. Both roles fall back to
    # the shared FIREWORKS_API_KEY when a role-specific key isn't set, so one key in
    # .env is enough; set the role-specific vars only to split roles across models.
    entity_resolution_base_url: str = field(
        default_factory=lambda: os.getenv("ENTITY_RESOLUTION_BASE_URL", "https://api.fireworks.ai/inference/v1")
    )
    entity_resolution_api_key: str = field(
        default_factory=lambda: os.getenv("ENTITY_RESOLUTION_API_KEY", os.getenv("FIREWORKS_API_KEY", ""))
    )
    entity_resolution_model: str = field(
        default_factory=lambda: os.getenv(
            "ENTITY_RESOLUTION_MODEL", "accounts/fireworks/models/deepseek-v4-flash-0731"
        )
    )

    temporal_update_base_url: str = field(
        default_factory=lambda: os.getenv("TEMPORAL_UPDATE_BASE_URL", "https://api.fireworks.ai/inference/v1")
    )
    temporal_update_api_key: str = field(
        default_factory=lambda: os.getenv("TEMPORAL_UPDATE_API_KEY", os.getenv("FIREWORKS_API_KEY", ""))
    )
    temporal_update_model: str = field(
        default_factory=lambda: os.getenv(
            "TEMPORAL_UPDATE_MODEL", "accounts/fireworks/models/deepseek-v4-flash-0731"
        )
    )

    embedding_model_name: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    )
    embedding_model_version: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL_VERSION", "1")
    )

    def get_entity_resolution_client(self) -> LLMClient:
        """LLMClient for `LLMEntityResolutionModel`. Requires a real API key to call."""
        return LLMClient(
            base_url=self.entity_resolution_base_url,
            api_key=self.entity_resolution_api_key,
            model_name=self.entity_resolution_model,
        )

    def get_temporal_update_client(self) -> LLMClient:
        """LLMClient for `LLMTemporalUpdateModel`. Requires a real API key to call."""
        return LLMClient(
            base_url=self.temporal_update_base_url,
            api_key=self.temporal_update_api_key,
            model_name=self.temporal_update_model,
        )
