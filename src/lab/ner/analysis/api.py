"""Public entry points for creating and reopening NER analysis artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from lab.ner.analysis.config import AnalysisConfig
from lab.ner.analysis.diagnostics import enrich_diagnostics
from lab.ner.analysis.evaluation import evaluate_to_events, read_span_input
from lab.ner.analysis.exposure import ExposureIndex
from lab.ner.analysis.features import audit_annotations, enrich_features, resolve_documents
from lab.ner.analysis.models import AnalysisResult
from lab.ner.analysis.parquet import SCHEMA_VERSION, read_evaluation_parquet, write_evaluation_parquet
from lab.ner.analysis.reporting import (
    concentration,
    confusion_tables,
    dataset_shift,
    document_analyses,
    error_intersections,
    factual_summary,
    label_analyses,
)
from lab.ner.analysis.subgroups import subgroup_analyses

EVALUATION_FILENAME = "evaluation.parquet"


def analyze_evaluation(
    predictions: pd.DataFrame | str | Path,
    evaluation_gold: pd.DataFrame | str | Path,
    training_gold: pd.DataFrame | str | Path,
    output_dir: str | Path,
    *,
    run_id: str,
    tags: list[str] | None = None,
    min_overlap_percentage: float = 40.0,
    config: AnalysisConfig | None = None,
    evaluation_documents: pd.DataFrame | dict[str, str] | str | Path | None = None,
    training_documents: pd.DataFrame | dict[str, str] | str | Path | None = None,
    tokenizer=None,
) -> AnalysisResult:
    """
    Create the reusable analytical foundation for one evaluated NER run.

    ``training_gold`` is deliberately a first-class input even though stage one
    only records its compact provenance and cardinalities. Later exposure and
    generalization analyses can extend this artifact without changing how a run
    is created. Full documents and training entities are not duplicated in the
    event table.
    """
    config = config or AnalysisConfig()
    training = read_span_input(training_gold)
    gold = read_span_input(evaluation_gold)
    predicted = read_span_input(predictions, scored=True)
    resolved_tags = tags if tags is not None else sorted(
        set(gold["label"].astype(str)) | set(predicted["label"].astype(str))
    )
    events, metrics = evaluate_to_events(
        gold,
        predicted,
        run_id=run_id,
        tags=resolved_tags,
        min_overlap_percentage=min_overlap_percentage,
    )
    events = enrich_diagnostics(events, gold, predicted)
    documents = resolve_documents(evaluation_documents)
    events = enrich_features(
        events, gold, predicted, evaluation_documents=documents, tokenizer=tokenizer,
        acronym_max_length=config.acronym_max_length,
        acronym_min_uppercase_ratio=config.acronym_min_uppercase_ratio,
        high_confidence_threshold=config.high_confidence_threshold,
        low_confidence_threshold=config.low_confidence_threshold,
    )
    exposure = ExposureIndex(training, config)
    events = _enrich_exposure(events, gold, predicted, exposure)
    subgroups = subgroup_analyses(
        gold, predicted, exposure, config, tags=resolved_tags,
        min_overlap_percentage=min_overlap_percentage, tokenizer=tokenizer,
    )
    confusion = confusion_tables(events)
    aggregates = {
        "subgroups": subgroups,
        "confusion": confusion,
        "documents": document_analyses(
            gold, predicted, events, tags=resolved_tags,
            min_overlap_percentage=min_overlap_percentage,
            documents=documents,
        ),
        "error_intersections": error_intersections(events, config.minimum_intersection_support),
        "error_concentration": concentration(events),
        "labels": label_analyses(events, training, subgroups),
        "dataset_shift": dataset_shift(training, gold),
    }
    aggregates["summary"] = factual_summary(metrics, events, subgroups, confusion)
    training_document_map = resolve_documents(training_documents)
    training_audit = audit_annotations(training, training_document_map)
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(run_id),
        "official_evaluator": "lab.ner.evaluation.span_metrics",
        "official_metrics": metrics,
        "strict_event_semantics": "TP rows contain both sides; FP prediction only; FN gold only",
        "character_metrics_level": "aggregate",
        "analysis_config": config.to_dict(),
        "similarity_backend": exposure.backend,
        "similarity_unique_training_mentions": exposure.unique_training_mentions,
        "similarity_cache_hits": exposure.cache_hits,
        "aggregates": aggregates,
        "training_gold": {
            "n_entities": int(len(training)),
            "n_documents": int(training["filename"].nunique()),
            "sha256": _span_digest(training),
            "annotation_valid": int(training_audit["annotation_valid"].sum()),
            "annotation_invalid": int((~training_audit["annotation_valid"]).sum()),
            "annotation_issue_counts": _issue_counts(training_audit),
        },
    }
    path = write_evaluation_parquet(
        events, Path(output_dir) / EVALUATION_FILENAME, metadata
    )

    return AnalysisResult(path=path, events=events, metrics=metrics, metadata=metadata)


def load_evaluation(path: str | Path) -> AnalysisResult:
    """Load a previously written ``evaluation.parquet`` without rerunning inference."""
    events, metadata = read_evaluation_parquet(path)

    return AnalysisResult(
        path=Path(path),
        events=events,
        metrics=metadata["official_metrics"],
        metadata=metadata,
    )


def _span_digest(spans: pd.DataFrame) -> str:
    rows = spans.sort_values(
        ["filename", "start_span", "end_span", "label", "text"], kind="stable"
    ).to_dict(orient="records")
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _enrich_exposure(events, gold, predicted, exposure):
    enriched = events.copy()
    descriptions: dict[tuple[str, int], dict] = {}
    for side, frame in (("gold", gold), ("pred", predicted)):
        for index, row in frame.iterrows():
            descriptions[(side, int(index))] = exposure.describe(str(row.text), str(row.label))
    columns = list(next(iter(descriptions.values()), exposure.describe("", "")))
    for column in columns:
        enriched[column] = None
    enriched["zero_shot_outcome"] = None
    enriched["train_exact_seen_with_pred_label"] = None
    enriched["train_normalized_seen_with_pred_label"] = None
    enriched["train_nearest_has_pred_label"] = None
    for event_index, event in enriched.iterrows():
        key = ("gold", int(event.gold_id)) if event.gold_exists else ("pred", int(event.pred_id))
        for column, value in descriptions[key].items():
            enriched.at[event_index, column] = value
        if event.gold_exists and bool(descriptions[key]["zero_shot_default"]):
            enriched.at[event_index, "zero_shot_outcome"] = f"ZERO_SHOT_{event.error_primary}"
        if event.gold_exists and pd.notna(event.diagnostic_pred_id):
            pred_label = str(predicted.iloc[int(event.diagnostic_pred_id)].label)
            exact_labels = descriptions[key]["train_exact_labels"]
            normalized_labels = descriptions[key]["train_normalized_labels"]
            enriched.at[event_index, "train_exact_seen_with_pred_label"] = pred_label in exact_labels
            enriched.at[event_index, "train_normalized_seen_with_pred_label"] = pred_label in normalized_labels
            enriched.at[event_index, "train_nearest_has_pred_label"] = descriptions[key]["train_nearest_label"] == pred_label
    return enriched


def _issue_counts(audit: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for issues in audit["annotation_issues"]:
        for issue in issues:
            counts[issue] = counts.get(issue, 0) + 1
    return dict(sorted(counts.items()))
