"""Diagnostic-only association, error taxonomy, and span structure features."""

from __future__ import annotations

import math
import string
from collections import Counter, defaultdict
from typing import Any

import pandas as pd


def enrich_diagnostics(
    events: pd.DataFrame,
    gold: pd.DataFrame,
    predicted: pd.DataFrame,
) -> pd.DataFrame:
    """Associate official FP/FN rows without changing their official contribution."""
    enriched = events.copy()
    defaults: dict[str, Any] = {
        "error_primary": None, "error_fragmentation": False, "error_merging": False,
        "error_duplicate_prediction": False, "error_nested_entity": False,
        "error_overlapping_entity": False, "error_pair_id": None,
        "error_pair_role": None, "error_label_agreement": None,
        "span_intersection": None, "span_iou": None, "span_start_delta": None,
        "span_end_delta": None, "span_length_delta": None,
        "span_abs_start_delta": None, "span_abs_end_delta": None,
        "span_boundary_error": None, "span_relation": None,
        "span_one_character_offset": False, "span_leading_whitespace": False,
        "span_trailing_whitespace": False, "span_punctuation_included": False,
        "span_punctuation_excluded": False,
        "diagnostic_gold_id": None, "diagnostic_pred_id": None,
        "gold_overlaps_other_gold": False, "gold_nested": False,
        "gold_contains_other_gold": False, "gold_contained_by_other_gold": False,
        "gold_overlap_depth": None,
    }
    for column, value in defaults.items():
        enriched[column] = value

    gold_structure = structural_gold_features(gold)
    duplicate_predictions = _duplicate_indices(predicted)
    unmatched_gold = set(enriched.loc[enriched["strict_is_fn"], "gold_id"].astype(int))
    unmatched_pred = set(enriched.loc[enriched["strict_is_fp"], "pred_id"].astype(int))
    overlaps_by_gold, overlaps_by_pred = _overlap_maps(gold, predicted, unmatched_gold, unmatched_pred)

    for index, row in enriched.iterrows():
        if row.strict_is_tp:
            enriched.at[index, "error_primary"] = "CORRECT"
        elif row.strict_is_fn:
            enriched.at[index, "error_primary"] = "MISSED"
        else:
            enriched.at[index, "error_primary"] = "SPURIOUS"

        if row.gold_exists:
            features = gold_structure[int(row.gold_id)]
            for name, value in features.items():
                enriched.at[index, name] = value
        if row.pred_exists and int(row.pred_id) in duplicate_predictions:
            enriched.at[index, "error_duplicate_prediction"] = True

    pairs = []
    for document in sorted(set(gold["filename"].astype(str)) | set(predicted["filename"].astype(str))):
        gold_ids = sorted(i for i in unmatched_gold if str(gold.iloc[i].filename) == document)
        pred_ids = sorted(i for i in unmatched_pred if str(predicted.iloc[i].filename) == document)
        pairs.extend(_assign(gold, predicted, gold_ids, pred_ids))

    gold_event = {
        int(row.gold_id): index for index, row in enriched.iterrows() if row.strict_is_fn
    }
    pred_event = {
        int(row.pred_id): index for index, row in enriched.iterrows() if row.strict_is_fp
    }
    for pair_id, (gold_id, pred_id) in enumerate(pairs):
        gold_row, pred_row = gold.iloc[gold_id], predicted.iloc[pred_id]
        details = _pair_details(gold_row, pred_row)
        primary = (
            "LABEL_ERROR" if details["span_boundary_error"] is None
            else "BOUNDARY_ERROR" if details["error_label_agreement"]
            else "BOUNDARY_AND_LABEL_ERROR"
        )
        for role, event_index in (("GOLD", gold_event[gold_id]), ("PRED", pred_event[pred_id])):
            enriched.at[event_index, "error_pair_id"] = pair_id
            enriched.at[event_index, "error_pair_role"] = role
            enriched.at[event_index, "error_primary"] = primary
            enriched.at[event_index, "diagnostic_gold_id"] = gold_id
            enriched.at[event_index, "diagnostic_pred_id"] = pred_id
            for name, value in details.items():
                enriched.at[event_index, name] = value

    for gold_id, predictions in overlaps_by_gold.items():
        if len(predictions) >= 2:
            enriched.loc[enriched["gold_id"] == gold_id, "error_fragmentation"] = True
            for pred_id in predictions:
                enriched.loc[enriched["pred_id"] == pred_id, "error_fragmentation"] = True
    for pred_id, golds in overlaps_by_pred.items():
        if len(golds) >= 2:
            enriched.loc[enriched["pred_id"] == pred_id, "error_merging"] = True
            for gold_id in golds:
                enriched.loc[enriched["gold_id"] == gold_id, "error_merging"] = True

    # Keep the Part-1 compatibility alias while making the taxonomy explicit.
    enriched["error_type"] = enriched["error_primary"]
    return enriched


def structural_gold_features(gold: pd.DataFrame) -> dict[int, dict[str, Any]]:
    result = {}
    for index, row in gold.iterrows():
        overlaps = []
        contains = contained = False
        for other_index, other in gold[gold["filename"].astype(str) == str(row.filename)].iterrows():
            if index == other_index:
                continue
            intersection = _intersection(row.start_span, row.end_span, other.start_span, other.end_span)
            if intersection:
                overlaps.append(other_index)
                contains |= row.start_span <= other.start_span and row.end_span >= other.end_span
                contained |= other.start_span <= row.start_span and other.end_span >= row.end_span
        result[int(index)] = {
            "gold_overlaps_other_gold": bool(overlaps),
            "gold_nested": contains or contained,
            "gold_contains_other_gold": contains,
            "gold_contained_by_other_gold": contained,
            "gold_overlap_depth": len(overlaps) + 1,
            "error_nested_entity": contains or contained,
            "error_overlapping_entity": bool(overlaps),
        }
    return result


def _assign(gold, predicted, gold_ids: list[int], pred_ids: list[int]) -> list[tuple[int, int]]:
    size = max(len(gold_ids), len(pred_ids))
    if not size:
        return []
    weights = [[0.0] * size for _ in range(size)]
    for i, gold_id in enumerate(gold_ids):
        for j, pred_id in enumerate(pred_ids):
            weights[i][j] = _association_weight(gold.iloc[gold_id], predicted.iloc[pred_id])
    assignment = _hungarian_max(weights)
    return [
        (gold_ids[i], pred_ids[j])
        for i, j in enumerate(assignment[: len(gold_ids)])
        if j < len(pred_ids) and weights[i][j] > 0
    ]


def _association_weight(gold, pred) -> float:
    intersection = _intersection(gold.start_span, gold.end_span, pred.start_span, pred.end_span)
    same_boundary = gold.start_span == pred.start_span and gold.end_span == pred.end_span
    if not intersection and not same_boundary:
        return 0.0
    union = max(gold.end_span, pred.end_span) - min(gold.start_span, pred.start_span)
    iou = intersection / union if union else 0.0
    distance = abs(pred.start_span - gold.start_span) + abs(pred.end_span - gold.end_span)
    confidence = float(pred.score) if "score" in pred.index and pd.notna(pred.score) else 0.0
    return (
        1_000_000 * same_boundary + 100_000 * iou + 10_000 * (str(gold.label) == str(pred.label))
        + 1_000 * (gold.start_span <= pred.start_span and gold.end_span >= pred.end_span)
        + 1_000 * (pred.start_span <= gold.start_span and pred.end_span >= gold.end_span)
        + 100 / (1 + distance) + confidence
    )


def _hungarian_max(weights: list[list[float]]) -> list[int]:
    """Deterministic O(n^3) maximum-weight assignment for a square matrix."""
    n = len(weights)
    maximum = max(max(row) for row in weights) if n else 0.0
    costs = [[maximum - value for value in row] for row in weights]
    u, v = [0.0] * (n + 1), [0.0] * (n + 1)
    match, way = [0] * (n + 1), [0] * (n + 1)
    for i in range(1, n + 1):
        match[0], j0 = i, 0
        minimum, used = [math.inf] * (n + 1), [False] * (n + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = match[j0], math.inf, 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                current = costs[i0 - 1][j - 1] - u[i0] - v[j]
                if current < minimum[j]:
                    minimum[j], way[j] = current, j0
                if minimum[j] < delta:
                    delta, j1 = minimum[j], j
            for j in range(n + 1):
                if used[j]:
                    u[match[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if match[j0] == 0:
                break
        while True:
            j1 = way[j0]
            match[j0] = match[j1]
            j0 = j1
            if j0 == 0:
                break
    answer = [0] * n
    for j in range(1, n + 1):
        answer[match[j] - 1] = j - 1
    return answer


def _pair_details(gold, pred) -> dict[str, Any]:
    start_delta = int(pred.start_span - gold.start_span)
    end_delta = int(pred.end_span - gold.end_span)
    intersection = _intersection(gold.start_span, gold.end_span, pred.start_span, pred.end_span)
    union = max(gold.end_span, pred.end_span) - min(gold.start_span, pred.start_span)
    boundary = None
    if start_delta and end_delta:
        boundary = "BOTH_BOUNDARIES_ERROR"
    elif start_delta:
        boundary = "LEFT_BOUNDARY_ERROR"
    elif end_delta:
        boundary = "RIGHT_BOUNDARY_ERROR"
    if pred.start_span <= gold.start_span and pred.end_span >= gold.end_span and boundary:
        relation = "PRED_CONTAINS_GOLD"
    elif gold.start_span <= pred.start_span and gold.end_span >= pred.end_span and boundary:
        relation = "GOLD_CONTAINS_PRED"
    elif pred.start_span < gold.start_span:
        relation = "LEFT_OVERLAP"
    elif pred.end_span > gold.end_span:
        relation = "RIGHT_OVERLAP"
    else:
        relation = "EXACT_BOUNDARY"
    pred_extra = str(pred.text).replace(str(gold.text), "", 1)
    gold_extra = str(gold.text).replace(str(pred.text), "", 1)
    return {
        "error_label_agreement": str(gold.label) == str(pred.label),
        "span_intersection": intersection,
        "span_iou": intersection / union if union else 0.0,
        "span_start_delta": start_delta, "span_end_delta": end_delta,
        "span_length_delta": (pred.end_span - pred.start_span) - (gold.end_span - gold.start_span),
        "span_abs_start_delta": abs(start_delta), "span_abs_end_delta": abs(end_delta),
        "span_boundary_error": boundary, "span_relation": relation,
        "span_one_character_offset": abs(start_delta) == 1 or abs(end_delta) == 1,
        "span_leading_whitespace": str(pred.text).startswith(" "),
        "span_trailing_whitespace": str(pred.text).endswith(" "),
        "span_punctuation_included": bool(pred_extra and all(c in string.punctuation or c.isspace() for c in pred_extra)),
        "span_punctuation_excluded": bool(gold_extra and all(c in string.punctuation or c.isspace() for c in gold_extra)),
    }


def _overlap_maps(gold, predicted, gold_ids, pred_ids):
    by_gold, by_pred = defaultdict(list), defaultdict(list)
    for gold_id in gold_ids:
        for pred_id in pred_ids:
            if str(gold.iloc[gold_id].filename) != str(predicted.iloc[pred_id].filename):
                continue
            if _intersection(gold.iloc[gold_id].start_span, gold.iloc[gold_id].end_span,
                             predicted.iloc[pred_id].start_span, predicted.iloc[pred_id].end_span):
                by_gold[gold_id].append(pred_id)
                by_pred[pred_id].append(gold_id)
    return by_gold, by_pred


def _duplicate_indices(predicted: pd.DataFrame) -> set[int]:
    keys = [tuple(row) for row in predicted[["filename", "label", "start_span", "end_span"]].itertuples(index=False, name=None)]
    counts = Counter(keys)
    return {index for index, key in enumerate(keys) if counts[key] > 1}


def _intersection(first_start, first_end, second_start, second_end) -> int:
    return max(0, min(int(first_end), int(second_end)) - max(int(first_start), int(second_start)))
