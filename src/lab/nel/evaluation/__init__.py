"""Ranking and ontology-aware evaluation, imported lazily by dependency family."""
from __future__ import annotations
import importlib

_HIERARCHY = {
    "build_snomed_graph", "calculate_recall_at_k", "graph_metrics_for_dataframe",
    "load_ontology_graph_pickle", "load_snomed_graph", "load_snomed_graph_pickle",
    "ontology_distance_and_direction", "ontology_summary_from_codes",
    "snomed_graph_distance_and_direction",
}
_METRICS = {
    "evaluate_candidate_dataframe", "evaluate_candidate_rows", "evaluate_predictions",
    "parse_code_list", "retrieval_metrics_from_codes", "unique_preserve_order",
}
__all__ = sorted(_HIERARCHY | _METRICS)

def __getattr__(name):
    if name in _HIERARCHY:
        return getattr(importlib.import_module("lab.nel.evaluation.hierarchy"), name)
    if name in _METRICS:
        return getattr(importlib.import_module("lab.nel.evaluation.metrics"), name)
    raise AttributeError(name)
