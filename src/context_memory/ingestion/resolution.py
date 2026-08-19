"""Bounded LLM-assisted entity resolution and temporal update decisions."""

from __future__ import annotations

from collections.abc import Iterable

from context_memory.ingestion.ports import EntityResolutionModel, GraphIdAllocator, TemporalUpdateModel
from context_memory.core.resolution import (
    EntityProfile,
    EntityResolution,
    FactState,
    ResolutionStatus,
    TemporalRelation,
    TemporalUpdateDecision,
    canonicalize_entity_surface,
)
from db.embedding_index import EntityNameIndex

class EntityRegistry:
    def __init__(self, allocator: GraphIdAllocator, name_index: EntityNameIndex | None = None, model: EntityResolutionModel | None = None) -> None:
        self._allocator = allocator
        self._name_index = name_index
        self._model = model
        self._profiles: dict[int, EntityProfile] = {}

    def register(self, profile: EntityProfile) -> None:
        existing = self._profiles.get(profile.graph_id)
        if existing is not None and existing != profile:
            raise ValueError(f"graph ID {profile.graph_id} has conflicting entity content")
        self._profiles[profile.graph_id] = profile

    def resolve_entity(self, context_id: str, surface: str, entity_type: str = "other") -> EntityProfile | None:
        """Helper adapter matching GraphPlanBuilder.ResolveEntity callable."""
        res = self.resolve(context_id=context_id, surface=surface, entity_type=entity_type)
        return res.entity

    def resolve(
        self,
        *,
        context_id: str,
        surface: str,
        entity_type: str = "other",
        candidate_ids: Iterable[int] = (),
        model: EntityResolutionModel | None = None,
    ) -> EntityResolution:
        effective_model = model if model is not None else self._model
        # Fetch dynamic candidates from Tier-2 Cache if available
        candidate_ids = list(candidate_ids)
        if self._name_index and not candidate_ids:
            candidates = self._name_index.find_candidates(query_name=surface, entity_type=entity_type, haystack_id=context_id)
            for c in candidates:
                # EntityNameIndex returns string IDs. In the real system they might need mapping back to graph IDs,
                # but if they match graph IDs (which are int), we map them back.
                try:
                    candidate_ids.append(int(c["entity_id"]))
                except ValueError:
                    pass
        
        canonical = canonicalize_entity_surface(surface)
        in_context = [profile for profile in self._profiles.values() if profile.context_id == context_id]
        canonicals = [profile for profile in in_context if canonicalize_entity_surface(profile.canonical_name) == canonical]
        if len(canonicals) == 1:
            return EntityResolution(ResolutionStatus.EXACT_CANONICAL, canonicals[0], "exact canonical match")
        aliases = [profile for profile in in_context if canonical in {canonicalize_entity_surface(a) for a in profile.aliases}]
        if len(aliases) == 1:
            return EntityResolution(ResolutionStatus.EXACT_ALIAS, aliases[0], "exact alias match")
        shortlist = {profile.graph_id: profile for profile in [*canonicals, *aliases]}
        for graph_id in candidate_ids:
            profile = self._profiles.get(graph_id)
            if profile is not None and profile.context_id == context_id:
                shortlist[graph_id] = profile
        if shortlist:
            if effective_model is None:
                return EntityResolution(ResolutionStatus.UNRESOLVED, None, "ambiguous candidate set requires model")
            selected_id = effective_model.resolve_entity(context_id=context_id, surface=surface, candidates=tuple(shortlist.values()))
            selected = shortlist.get(selected_id) if selected_id is not None else None
            if selected is None:
                return EntityResolution(ResolutionStatus.UNRESOLVED, None, "model abstained or selected outside bounded candidates")
            return EntityResolution(ResolutionStatus.MODEL_RESOLVED, selected, "model selected bounded candidate")
        graph_id = self._allocator.allocate_graph_id("entity", context_id, f"entity:{canonical}")
        profile = EntityProfile(graph_id, context_id, canonical, entity_type)
        self.register(profile)
        return EntityResolution(ResolutionStatus.NEW_ENTITY, profile, "no candidate in active context")


class TemporalUpdateClassifier:
    def __init__(self, model: TemporalUpdateModel) -> None:
        self._model = model

    def classify(self, *, new_fact: FactState, prior_fact: FactState) -> TemporalUpdateDecision:
        if new_fact.observed_at <= prior_fact.observed_at:
            return TemporalUpdateDecision(TemporalRelation.NO_UPDATE, "out-of-order or equal observation time")
        if new_fact.subject_entity_id != prior_fact.subject_entity_id:
            return TemporalUpdateDecision(TemporalRelation.NO_UPDATE, "different subject entity")
        if new_fact.predicate_key != prior_fact.predicate_key:
            return TemporalUpdateDecision(TemporalRelation.NO_UPDATE, "different predicate key")
        relation = self._model.classify_update(new_fact=new_fact, prior_fact=prior_fact)
        if relation is TemporalRelation.CORRECTION:
            return TemporalUpdateDecision(relation, "model-confirmed correction", prior_superseded_at=new_fact.observed_at)
        if relation is TemporalRelation.STATE_CHANGE:
            effective_at = new_fact.valid_from or new_fact.observed_at
            return TemporalUpdateDecision(
                relation, "model-confirmed state change", prior_superseded_at=new_fact.observed_at,
                prior_valid_to=effective_at,
            )
        if relation is TemporalRelation.NO_UPDATE:
            return TemporalUpdateDecision(relation, "model found no update")
        return TemporalUpdateDecision(TemporalRelation.UNRESOLVED, "model abstained")
