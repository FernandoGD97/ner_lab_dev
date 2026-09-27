"""Methodologically complete subgroup scoring through the existing evaluator."""

from __future__ import annotations

import pandas as pd

from lab.ner.analysis.config import AnalysisConfig
from lab.ner.analysis.diagnostics import structural_gold_features
from lab.ner.analysis.exposure import ExposureIndex
from lab.ner.analysis.features import entity_features
from lab.ner.evaluation import span_metrics


def subgroup_analyses(
    gold: pd.DataFrame,
    predicted: pd.DataFrame,
    exposure: ExposureIndex,
    config: AnalysisConfig,
    *,
    tags: list[str],
    min_overlap_percentage: float,
    tokenizer=None,
) -> list[dict]:
    """Evaluate gold and prediction sides independently for every declared stratum."""
    gold_properties = _properties(gold, exposure, config, tokenizer)
    pred_properties = _properties(predicted, exposure, config, tokenizer)
    rows: list[dict] = []

    def add(family: str, value: str, gold_mask, pred_mask, label: str | None = None):
        result = score_subgroup(
            gold, predicted, gold_mask, pred_mask,
            tags=tags if label is None else [label],
            min_overlap_percentage=min_overlap_percentage,
        )
        rows.append({"family": family, "value": value, "label": label, **result})

    binary = {
        "exact_exposure": "train_exact_seen",
        "normalized_exposure": "train_normalized_seen",
        "fuzzy_exposure": "train_fuzzy_seen",
        "zero_shot_default": "zero_shot_default",
        "acronym": "entity_acronym_like",
        "overlapping": "gold_overlaps_other_gold",
        "nested": "gold_nested",
    }
    for family, column in binary.items():
        for value in (True, False):
            add(family, str(value).lower(), gold_properties[column] == value, pred_properties[column] == value)
    for column, family in (
        ("similarity_bin", "similarity"),
        ("train_frequency_bin", "training_frequency"),
        ("entity_word_length_bin", "word_length"),
        ("tokenization_fragmentation_category", "tokenization"),
    ):
        values = sorted(set(gold_properties[column].dropna()) | set(pred_properties[column].dropna()))
        for value in values:
            add(family, str(value), gold_properties[column] == value, pred_properties[column] == value)
    for label in tags:
        gold_label = gold["label"].astype(str) == label
        pred_label = predicted["label"].astype(str) == label
        add("label", label, gold_label, pred_label, label)
        for zero in (False, True):
            add(
                "label_zero_shot", f"{'zero_shot' if zero else 'seen'}",
                gold_label & (gold_properties["zero_shot_default"] == zero),
                pred_label & (pred_properties["zero_shot_default"] == zero), label,
            )
    if "score" in predicted.columns:
        for threshold in config.confidence_thresholds:
            add(
                "confidence_threshold", f"{threshold:.6g}",
                pd.Series(True, index=gold.index),
                predicted["score"].astype(float) >= threshold,
            )
            rows[-1]["retained_predictions"] = int((predicted["score"].astype(float) >= threshold).sum())
    return rows


def score_subgroup(
    gold: pd.DataFrame,
    predicted: pd.DataFrame,
    gold_mask,
    pred_mask,
    *,
    tags: list[str],
    min_overlap_percentage: float = 40.0,
) -> dict:
    """Score independently selected gold/prediction populations with the official scorer."""
    selected_gold = gold.loc[pd.Series(gold_mask, index=gold.index).fillna(False)].reset_index(drop=True)
    selected_pred = predicted.loc[pd.Series(pred_mask, index=predicted.index).fillna(False)].reset_index(drop=True)
    metrics = span_metrics(
        selected_gold, selected_pred, tags=tags, min_overlap_percentage=min_overlap_percentage
    )
    incorrect = int(metrics.get("span_strict_incorrect", 0)) + int(metrics.get("span_strict_partial", 0))
    return {
        "support": int(len(selected_gold)),
        "predicted_support": int(len(selected_pred)),
        "strict_tp": int(metrics.get("span_strict_correct", 0)),
        "strict_fp": int(metrics.get("span_strict_spurious", 0)) + incorrect,
        "strict_fn": int(metrics.get("span_strict_missed", 0)) + incorrect,
        "strict_precision": float(metrics.get("span_strict_precision", 0.0)),
        "strict_recall": float(metrics.get("span_strict_recall", 0.0)),
        "strict_f1": float(metrics.get("span_strict_f1", 0.0)),
        "char_tp": int(metrics.get("char_correct", 0)),
        "char_fp": int(metrics.get("char_spurious", 0)),
        "char_fn": int(metrics.get("char_missed", 0)),
        "char_precision": float(metrics.get("char_precision", 0.0)),
        "char_recall": float(metrics.get("char_recall", 0.0)),
        "char_f1": float(metrics.get("char_f1", 0.0)),
    }


def _properties(frame, exposure, config, tokenizer):
    records = []
    structures = structural_gold_features(frame)
    for index, row in frame.iterrows():
        described = exposure.describe(str(row.text), str(row.label))
        described["train_fuzzy_seen"] = not described["zero_shot_fuzzy"]
        described.update(entity_features(
            str(row.text), tokenizer, config.acronym_max_length, config.acronym_min_uppercase_ratio
        ))
        structure = structures[int(index)]
        described["gold_overlaps_other_gold"] = structure["gold_overlaps_other_gold"]
        described["gold_nested"] = structure["gold_nested"]
        records.append(described)
    if records:
        return pd.DataFrame(records, index=frame.index)
    template = exposure.describe("", "")
    template["train_fuzzy_seen"] = not template["zero_shot_fuzzy"]
    template.update(entity_features(
        "", tokenizer, config.acronym_max_length, config.acronym_min_uppercase_ratio
    ))
    template.update({"gold_overlaps_other_gold": False, "gold_nested": False})
    return pd.DataFrame(columns=template, index=frame.index)
