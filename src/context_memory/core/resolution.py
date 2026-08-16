"""Entity and temporal decisions kept independent from graph transport."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from context_memory.core.errors import ContractValidationError


def canonicalize_entity_surface(surface: str) -> str:
    if not isinstance(surface, str) or not surface.strip():
        raise ContractValidationError("entity.surface", "must be a non-empty string")
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", surface).strip()).casefold()


class ResolutionStatus(StrEnum):
    EXACT_CANONICAL = "exact_canonical"
    EXACT_ALIAS = "exact_alias"
    MODEL_RESOLVED = "model_resolved"
    NEW_ENTITY = "new_entity"
    UNRESOLVED = "unresolved"


class TemporalRelation(StrEnum):
    CORRECTION = "correction"
    STATE_CHANGE = "state_change"
    NO_UPDATE = "no_update"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class EntityProfile:
    graph_id: int
    context_id: str
    canonical_name: str
    entity_type: str = "other"
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.graph_id, int) or self.graph_id < 0:
            raise ContractValidationError("entity.graph_id", "must be a non-negative integer")
        if not self.context_id:
            raise ContractValidationError("entity.context_id", "must be non-empty")
        canonicalize_entity_surface(self.canonical_name)
        if not self.entity_type:
            raise ContractValidationError("entity.entity_type", "must be non-empty")
        for alias in self.aliases:
            canonicalize_entity_surface(alias)


@dataclass(frozen=True)
class EntityResolution:
    status: ResolutionStatus
    entity: EntityProfile | None
    reason: str


@dataclass(frozen=True)
class FactState:
    fact_id: str
    subject_entity_id: int
    predicate_key: str
    text: str
    observed_at: datetime
    valid_from: datetime | None = None
    valid_to: datetime | None = None

    def __post_init__(self) -> None:
        if not self.fact_id or not self.predicate_key or not self.text:
            raise ContractValidationError("fact_state", "fact_id, predicate_key, and text must be non-empty")
        if not isinstance(self.subject_entity_id, int) or self.subject_entity_id < 0:
            raise ContractValidationError("fact_state.subject_entity_id", "must be non-negative")
        if self.observed_at.tzinfo is None:
            raise ContractValidationError("fact_state.observed_at", "must include a UTC offset")
        if self.valid_from is not None and self.valid_from.tzinfo is None:
            raise ContractValidationError("fact_state.valid_from", "must include a UTC offset")
        if self.valid_to is not None and self.valid_to.tzinfo is None:
            raise ContractValidationError("fact_state.valid_to", "must include a UTC offset")
        if self.valid_from and self.valid_to and self.valid_from > self.valid_to:
            raise ContractValidationError("fact_state.valid_from", "must not be later than valid_to")


@dataclass(frozen=True)
class TemporalUpdateDecision:
    relation: TemporalRelation
    reason: str
    prior_superseded_at: datetime | None = None
    prior_valid_to: datetime | None = None
