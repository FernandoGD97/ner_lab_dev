"""Reusable, half-open-offset NER error taxonomy and prediction comparison."""
from __future__ import annotations

from typing import Any

import pandas as pd

KEY = ["filename", "start_span", "end_span", "label"]


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    required = [*KEY, "text"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Span table is missing columns: {missing}")
    return frame[required].to_dict("records")


def overlaps(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        str(left["filename"]) == str(right["filename"])
        and int(left["start_span"]) < int(right["end_span"])
        and int(right["start_span"]) < int(left["end_span"])
    )


def error_taxonomy(gold: pd.DataFrame, predicted: pd.DataFrame) -> pd.DataFrame:
    """Classify gold and unmatched predictions without changing primary scoring."""
    gold_rows = _records(gold)
    predictions = _records(predicted)
    used: set[int] = set()
    output: list[dict[str, Any]] = []
    for truth in gold_rows:
        exact = [
            (index, candidate) for index, candidate in enumerate(predictions)
            if index not in used
            and str(candidate["filename"]) == str(truth["filename"])
            and int(candidate["start_span"]) == int(truth["start_span"])
            and int(candidate["end_span"]) == int(truth["end_span"])
        ]
        correct = next(((i, p) for i, p in exact if p["label"] == truth["label"]), None)
        if correct:
            selected, prediction, category = correct[0], correct[1], "correct"
        elif exact:
            selected, prediction, category = exact[0][0], exact[0][1], "wrong_label"
        else:
            candidates = [
                (index, candidate) for index, candidate in enumerate(predictions)
                if index not in used and overlaps(truth, candidate)
            ]
            same_label = next(((i, p) for i, p in candidates if p["label"] == truth["label"]), None)
            if same_label:
                selected, prediction = same_label
                left = int(prediction["start_span"]) != int(truth["start_span"])
                right = int(prediction["end_span"]) != int(truth["end_span"])
                category = "both_boundaries_error" if left and right else (
                    "left_boundary_error" if left else "right_boundary_error"
                )
            elif candidates:
                selected, prediction, category = candidates[0][0], candidates[0][1], "overlapping_prediction"
            else:
                selected, prediction, category = None, None, "false_negative"
        if selected is not None:
            used.add(selected)
        output.append({
            "document_id": str(truth["filename"]), "start": int(truth["start_span"]),
            "end": int(truth["end_span"]), "text": truth["text"],
            "gold_label": truth["label"], "prediction_label": None if prediction is None else prediction["label"],
            "prediction_start": None if prediction is None else int(prediction["start_span"]),
            "prediction_end": None if prediction is None else int(prediction["end_span"]),
            "error_type": category,
        })
    for index, prediction in enumerate(predictions):
        if index not in used:
            output.append({
                "document_id": str(prediction["filename"]), "start": int(prediction["start_span"]),
                "end": int(prediction["end_span"]), "text": prediction["text"],
                "gold_label": None, "prediction_label": prediction["label"],
                "prediction_start": int(prediction["start_span"]),
                "prediction_end": int(prediction["end_span"]), "error_type": "false_positive",
            })
    return pd.DataFrame(output)


def compare_predictions(
    gold: pd.DataFrame,
    baseline: pd.DataFrame,
    compressed: pd.DataFrame,
) -> tuple[dict[str, int], pd.DataFrame]:
    """Compare exact recoveries and false positives, retaining detailed examples."""
    def keys(frame):
        return {tuple(row[column] for column in KEY) for row in frame.to_dict("records")}

    gold_keys, baseline_keys, compressed_keys = keys(gold), keys(baseline), keys(compressed)
    baseline_tp, compressed_tp = baseline_keys & gold_keys, compressed_keys & gold_keys
    baseline_fp, compressed_fp = baseline_keys - gold_keys, compressed_keys - gold_keys
    summary = {
        "gold_recovered_by_both": len(baseline_tp & compressed_tp),
        "gold_recovered_only_by_baseline": len(baseline_tp - compressed_tp),
        "gold_recovered_only_by_compressed": len(compressed_tp - baseline_tp),
        "false_positives_shared": len(baseline_fp & compressed_fp),
        "new_false_positives": len(compressed_fp - baseline_fp),
        "removed_false_positives": len(baseline_fp - compressed_fp),
    }
    base_errors = error_taxonomy(gold, baseline)
    compressed_errors = error_taxonomy(gold, compressed)
    gold_error_columns = ["document_id", "start", "end", "gold_label"]
    merged = base_errors[base_errors.gold_label.notna()].merge(
        compressed_errors[compressed_errors.gold_label.notna()],
        on=gold_error_columns, suffixes=("_baseline", "_compressed"), how="outer",
    )
    merged["difference_type"] = merged.apply(
        lambda row: (
            "recovered_by_both" if row.error_type_baseline == row.error_type_compressed == "correct"
            else "baseline_only" if row.error_type_baseline == "correct"
            else "compressed_only" if row.error_type_compressed == "correct"
            else "label_change" if "wrong_label" in {row.error_type_baseline, row.error_type_compressed}
            else "boundary_change" if any("boundar" in str(value) for value in (row.error_type_baseline, row.error_type_compressed))
            else "shared_error"
        ), axis=1,
    )
    summary["boundary_changes"] = int((merged.difference_type == "boundary_change").sum())
    summary["label_changes"] = int((merged.difference_type == "label_change").sum())
    baseline_records = {tuple(row[column] for column in KEY): row for row in baseline.to_dict("records")}
    compressed_records = {tuple(row[column] for column in KEY): row for row in compressed.to_dict("records")}
    false_positive_rows = []
    for key in sorted(baseline_fp | compressed_fp):
        record = baseline_records.get(key) or compressed_records[key]
        difference = (
            "false_positive_shared" if key in baseline_fp and key in compressed_fp
            else "new_false_positive" if key in compressed_fp else "removed_false_positive"
        )
        false_positive_rows.append({
            "document_id": key[0], "start": key[1], "end": key[2], "gold_label": None,
            "text_baseline": record.get("text"), "text_compressed": record.get("text"),
            "prediction_label_baseline": key[3] if key in baseline_fp else None,
            "prediction_label_compressed": key[3] if key in compressed_fp else None,
            "error_type_baseline": "false_positive" if key in baseline_fp else None,
            "error_type_compressed": "false_positive" if key in compressed_fp else None,
            "difference_type": difference,
        })
    if false_positive_rows:
        merged = pd.concat([merged, pd.DataFrame(false_positive_rows)], ignore_index=True, sort=False)
    return summary, merged
