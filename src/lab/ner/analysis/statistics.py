"""Deterministic document-level bootstrap uncertainty for official metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from lab.core.scoring import safe_f1
from lab.ner.evaluation import span_metrics


@dataclass(frozen=True)
class BootstrapConfig:
    enabled: bool = True
    samples: int = 2000
    confidence_level: float = 0.95
    seed: int = 20250927
    minimum_documents: int = 2

    def __post_init__(self):
        if self.samples < 1:
            raise ValueError("bootstrap samples must be positive.")
        if not 0 < self.confidence_level < 1:
            raise ValueError("bootstrap confidence_level must be between 0 and 1.")

    def to_dict(self):
        return asdict(self)


def bootstrap_intervals(events: pd.DataFrame, config: BootstrapConfig) -> pd.DataFrame:
    """Bootstrap overall and configured analytical strata by whole document."""
    targets = _targets(events)
    documents = sorted(events["document_id"].astype(str).unique())
    columns = [
        "family", "value", "metric", "observed", "ci_low", "ci_high",
        "confidence_level", "samples", "n_documents", "support", "status",
    ]
    if not config.enabled:
        return pd.DataFrame(columns=columns)
    observed = {key: _score(events, selector) for key, selector in targets.items()}
    if len(documents) < config.minimum_documents:
        return pd.DataFrame([
            {
                "family": family, "value": value, "metric": metric,
                "observed": score[metric], "ci_low": None, "ci_high": None,
                "confidence_level": config.confidence_level, "samples": 0,
                "n_documents": len(documents), "support": score["support"],
                "status": "insufficient_documents",
            }
            for (family, value), score in observed.items()
            for metric in _metrics()
        ], columns=columns)

    per_document = {
        key: {
            document: _score(events[events["document_id"].astype(str) == document], selector)
            for document in documents
        }
        for key, selector in targets.items()
    }
    rng = np.random.default_rng(config.seed)
    draws = {key: {metric: [] for metric in _metrics()} for key in targets}
    for _ in range(config.samples):
        sampled = rng.choice(documents, size=len(documents), replace=True)
        for key in targets:
            score = _aggregate_scores([per_document[key][str(document)] for document in sampled])
            for metric in _metrics():
                draws[key][metric].append(score[metric])
    alpha = (1 - config.confidence_level) / 2
    rows = []
    for (family, value), score in observed.items():
        for metric in _metrics():
            values = draws[(family, value)][metric]
            rows.append({
                "family": family, "value": value, "metric": metric,
                "observed": score[metric],
                "ci_low": float(np.quantile(values, alpha)),
                "ci_high": float(np.quantile(values, 1 - alpha)),
                "confidence_level": config.confidence_level,
                "samples": config.samples, "n_documents": len(documents),
                "support": score["support"], "status": "ok",
            })
    return pd.DataFrame(rows, columns=columns)


def paired_seen_zero_shot_difference(events: pd.DataFrame, config: BootstrapConfig) -> pd.DataFrame:
    """Paired document bootstrap of seen-minus-zero-shot metric differences."""
    columns = ["metric", "observed_difference", "ci_low", "ci_high", "samples", "n_documents", "status"]
    documents = sorted(events["document_id"].astype(str).unique())
    seen = lambda frame: ~frame["zero_shot_default"].fillna(True).astype(bool)
    unseen = lambda frame: frame["zero_shot_default"].fillna(False).astype(bool)
    observed_seen, observed_unseen = _score(events, seen), _score(events, unseen)
    if not config.enabled or len(documents) < config.minimum_documents:
        return pd.DataFrame([
            {"metric": metric, "observed_difference": observed_seen[metric] - observed_unseen[metric],
             "ci_low": None, "ci_high": None, "samples": 0, "n_documents": len(documents),
             "status": "disabled" if not config.enabled else "insufficient_documents"}
            for metric in _metrics()
        ], columns=columns)
    per_document = {
        "seen": {document: _score(events[events["document_id"].astype(str) == document], seen) for document in documents},
        "unseen": {document: _score(events[events["document_id"].astype(str) == document], unseen) for document in documents},
    }
    rng = np.random.default_rng(config.seed)
    values = {metric: [] for metric in _metrics()}
    for _ in range(config.samples):
        sampled = rng.choice(documents, size=len(documents), replace=True)
        left = _aggregate_scores([per_document["seen"][str(document)] for document in sampled])
        right = _aggregate_scores([per_document["unseen"][str(document)] for document in sampled])
        for metric in _metrics():
            values[metric].append(left[metric] - right[metric])
    alpha = (1 - config.confidence_level) / 2
    return pd.DataFrame([
        {"metric": metric, "observed_difference": observed_seen[metric] - observed_unseen[metric],
         "ci_low": float(np.quantile(values[metric], alpha)),
         "ci_high": float(np.quantile(values[metric], 1 - alpha)),
         "samples": config.samples, "n_documents": len(documents), "status": "ok"}
        for metric in _metrics()
    ], columns=columns)


def _targets(events):
    targets = {("overall", "all"): lambda frame: pd.Series(True, index=frame.index)}
    for family, column in (
        ("zero_shot_default", "zero_shot_default"),
        ("similarity", "similarity_bin"),
        ("label", "gold_label"),
        ("training_frequency", "train_frequency_bin"),
        ("overlapping", "gold_overlaps_other_gold"),
        ("acronym", "entity_acronym_like"),
    ):
        values = sorted(str(value) for value in events[column].dropna().unique())
        for value in values:
            targets[(family, value)] = lambda frame, column=column, value=value: frame[column].astype(str) == value
    return targets


def _score(events, selector):
    gold_events = events[events["gold_exists"]].copy()
    pred_events = events[events["pred_exists"]].copy()
    gold_mask = selector(gold_events).fillna(False)
    pred_mask = _prediction_selector(selector, pred_events).fillna(False)
    gold = _span_frame(gold_events.loc[gold_mask], "gold")
    predicted = _span_frame(pred_events.loc[pred_mask], "pred")
    tags = sorted(set(gold["label"].astype(str)) | set(predicted["label"].astype(str)))
    metrics = span_metrics(gold, predicted, tags=tags)
    incorrect = int(metrics.get("span_strict_incorrect", 0)) + int(metrics.get("span_strict_partial", 0))
    return {
        "support": len(gold),
        "strict_tp": int(metrics.get("span_strict_correct", 0)),
        "strict_fp": int(metrics.get("span_strict_spurious", 0)) + incorrect,
        "strict_fn": int(metrics.get("span_strict_missed", 0)) + incorrect,
        "char_tp": int(metrics.get("char_correct", 0)),
        "char_fp": int(metrics.get("char_spurious", 0)),
        "char_fn": int(metrics.get("char_missed", 0)),
        "strict_precision": float(metrics.get("span_strict_precision", 0.0)),
        "strict_recall": float(metrics.get("span_strict_recall", 0.0)),
        "strict_f1": float(metrics.get("span_strict_f1", 0.0)),
        "char_precision": float(metrics.get("char_precision", 0.0)),
        "char_recall": float(metrics.get("char_recall", 0.0)),
        "char_f1": float(metrics.get("char_f1", 0.0)),
    }


def _prediction_selector(selector, frame):
    renamed = frame.copy()
    renamed["gold_label"] = renamed["pred_label"]
    return selector(renamed)


def _span_frame(events, prefix):
    frame = pd.DataFrame({
        "filename": events["document_id"].astype(str),
        "label": events[f"{prefix}_label"].astype(str),
        "start_span": events[f"{prefix}_start"].astype(int),
        "end_span": events[f"{prefix}_end"].astype(int),
        "text": events[f"{prefix}_text"].astype(str),
        "_order": events[f"{prefix}_id"].astype(int),
    })
    return frame.sort_values(["filename", "_order"], kind="stable").drop(columns="_order").reset_index(drop=True)


def _aggregate_scores(scores):
    totals = {
        key: sum(int(score[key]) for score in scores)
        for key in ("support", "strict_tp", "strict_fp", "strict_fn", "char_tp", "char_fp", "char_fn")
    }
    strict = safe_f1(totals["strict_tp"], totals["strict_fp"], totals["strict_fn"])
    character = safe_f1(totals["char_tp"], totals["char_fp"], totals["char_fn"])
    return {
        **totals,
        "strict_precision": strict["precision"], "strict_recall": strict["recall"], "strict_f1": strict["f1"],
        "char_precision": character["precision"], "char_recall": character["recall"], "char_f1": character["f1"],
    }


def _metrics():
    return ("strict_precision", "strict_recall", "strict_f1", "char_precision", "char_recall", "char_f1")
