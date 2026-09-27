"""EL retrieval metrics and training-derived seen/unseen breakdowns."""
from __future__ import annotations
from collections import Counter
from typing import Any, Sequence

DEFAULT_K = (1,5,25,50,100,200)

def el_metrics(gold: Sequence[Any], predictions: Sequence[Sequence[Any]], k_values=DEFAULT_K) -> dict[str, Any]:
    from importlib import import_module
    retrieval_metrics_from_codes = import_module(
        "lab.nel.evaluation.metrics"
    ).retrieval_metrics_from_codes
    metrics = retrieval_metrics_from_codes(gold,predictions,k_values)
    metrics["accuracy_at_1"] = metrics.get("recall@1")
    metrics["candidate_recall"] = metrics.get(f"recall@{max(k_values)}")
    return metrics

def concept_frequency_bucket(count: int) -> str:
    if count == 0:return "unseen"
    if count == 1:return "singleton"
    if count <= 5:return "2-5"
    if count <= 20:return "6-20"
    return ">20"

def seen_unseen_metrics(gold, predictions, training_counts: Counter[str] | None):
    if training_counts is None:
        return {"seen_accuracy":None,"unseen_accuracy":None,"definition":None}
    groups={"seen":[],"unseen":[]}
    for index,code in enumerate(gold): groups["seen" if training_counts[str(code)] else "unseen"].append(index)
    output={"definition":"seen iff gold concept ID occurs in supplied training annotations"}
    for name,indices in groups.items():
        output[f"{name}_accuracy"] = None if not indices else sum(bool(predictions[i]) and str(predictions[i][0])==str(gold[i]) for i in indices)/len(indices)
    return output
