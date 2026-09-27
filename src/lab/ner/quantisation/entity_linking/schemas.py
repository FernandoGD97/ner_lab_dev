"""Entity-linking benchmark schemas, explicitly separating offline and online cost."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any

EL_COLUMNS = [
    "experiment_id","run_id","model_id","compression","mention_count","concept_count",
    "embedding_dimension","index_type","distance_metric","index_size_bytes","candidate_k",
    "embedding_memory_bytes",
    "context_time_s","entity_encoding_time_s","index_build_time_s","index_serialization_time_s",
    "mention_encoding_time_s","retrieval_time_s","candidate_preparation_time_s",
    "biencoder_time_s","index_time_s","crossencoder_time_s","postprocessing_time_s","el_total_time_s","candidate_pairs",
    "pairs_per_s","accuracy_at_1","recall_at_1","recall_at_5","recall_at_25",
    "recall_at_50","recall_at_100","recall_at_200","mrr","candidate_recall",
    "seen_accuracy","unseen_accuracy","status",
    "el_total_energy_kwh","el_co2_kg","joules_per_1000_mentions",
    "joules_per_1000_candidate_pairs","joules_per_1000_queries",
]

@dataclass(frozen=True)
class OfflineCosts:
    entity_encoding_time_s: float | None = None
    index_build_time_s: float | None = None
    index_serialization_time_s: float | None = None
    index_size_bytes: int | None = None

@dataclass(frozen=True)
class OnlineCosts:
    context_time_s: float | None = None
    mention_encoding_time_s: float | None = None
    retrieval_time_s: float | None = None
    candidate_preparation_time_s: float | None = None
    crossencoder_time_s: float | None = None
    postprocessing_time_s: float | None = None
    el_total_time_s: float | None = None


def energy_normalization(total_kwh: float | None, mentions: int, pairs: int, queries: int) -> dict[str, Any]:
    factor = None if total_kwh is None else total_kwh * 3_600_000 * 1000
    return {
        "joules_per_1000_mentions": None if factor is None or not mentions else factor / mentions,
        "joules_per_1000_candidate_pairs": None if factor is None or not pairs else factor / pairs,
        "joules_per_1000_queries": None if factor is None or not queries else factor / queries,
    }
