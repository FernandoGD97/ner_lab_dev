"""Compact aggregate diagnostics persisted with canonical event truth."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from lab.ner.analysis.subgroups import score_subgroup


def confusion_tables(events: pd.DataFrame) -> dict[str, list[dict]]:
    paired = events[(events["error_pair_role"] == "GOLD") & (events["error_primary"] == "LABEL_ERROR")]
    counts = Counter((str(row.gold_label), _paired_pred_label(events, row)) for row in paired.itertuples())
    row_totals, column_totals = Counter(), Counter()
    for (gold, pred), count in counts.items():
        row_totals[gold] += count
        column_totals[pred] += count
    exact = [
        {
            "gold_label": gold, "pred_label": pred, "count": count,
            "row_proportion": count / row_totals[gold],
            "column_proportion": count / column_totals[pred],
        }
        for (gold, pred), count in sorted(counts.items())
    ]
    flows = Counter()
    for row in events.itertuples():
        if row.strict_is_tp:
            flows[(str(row.gold_label), str(row.pred_label))] += 1
        elif row.strict_is_fn:
            flows[(str(row.gold_label), "[MISSED]")] += 1
        elif row.strict_is_fp:
            flows[("[SPURIOUS]", str(row.pred_label))] += 1
    return {
        "exact_boundary_label_confusion": exact,
        "error_flows": [
            {"source": source, "target": target, "count": count}
            for (source, target), count in sorted(flows.items())
        ],
    }


def document_analyses(gold, predicted, events, *, tags, min_overlap_percentage, documents=None) -> list[dict]:
    rows = []
    for document in sorted(set(gold["filename"].astype(str)) | set(predicted["filename"].astype(str))):
        gold_mask = gold["filename"].astype(str) == document
        pred_mask = predicted["filename"].astype(str) == document
        score = score_subgroup(
            gold, predicted, gold_mask, pred_mask, tags=tags,
            min_overlap_percentage=min_overlap_percentage,
        )
        document_events = events[events["document_id"] == document]
        gold_events = document_events[document_events["gold_exists"]]
        document_length = len(documents[document]) if documents and document in documents else None
        rows.append({
            "document_id": document,
            **score,
            "zero_shot_count": int(gold_events["zero_shot_default"].fillna(False).sum()),
            "zero_shot_rate": float(gold_events["zero_shot_default"].fillna(False).mean()) if len(gold_events) else 0.0,
            "mean_train_similarity": _mean(gold_events["train_nearest_similarity"]),
            "boundary_error_count": int((gold_events["error_primary"] == "BOUNDARY_ERROR").sum()),
            "label_error_count": int((gold_events["error_primary"] == "LABEL_ERROR").sum()),
            "document_length": document_length,
            "entity_density": (score["support"] / document_length) if document_length else None,
        })
    return rows


def concentration(events: pd.DataFrame) -> list[dict]:
    errors = events[events["strict_is_fn"] | events["strict_is_fp"]]
    dimensions = {
        "label": errors.apply(lambda r: r.gold_label if r.gold_exists else r.pred_label, axis=1),
        "surface": errors.apply(lambda r: r.gold_text if r.gold_exists else r.pred_text, axis=1),
        "document": errors["document_id"],
        "error_category": errors["error_primary"],
        "confusion_pair": errors.apply(
            lambda r: f"{r.gold_label}->{r.pred_label}"
            if r.error_primary == "LABEL_ERROR" else None, axis=1
        ),
    }
    rows = []
    for dimension, values in dimensions.items():
        counts = values.value_counts(dropna=False)
        total = int(counts.sum())
        for rank, (value, count) in enumerate(counts.head(10).items(), 1):
            rows.append({
                "dimension": dimension, "rank": rank, "value": str(value),
                "count": int(count), "proportion": float(count / total) if total else 0.0,
            })
    return rows


def label_analyses(events: pd.DataFrame, training: pd.DataFrame, subgroup_rows: list[dict]) -> list[dict]:
    rows = []
    labels = sorted(set(training["label"].astype(str)) | set(events["gold_label"].dropna().astype(str)) | set(events["pred_label"].dropna().astype(str)))
    indexed = {(row["family"], row["value"], row.get("label")): row for row in subgroup_rows}
    for label in labels:
        overall = indexed.get(("label", label, label), {})
        seen = indexed.get(("label_zero_shot", "seen", label), {})
        unseen = indexed.get(("label_zero_shot", "zero_shot", label), {})
        gold = events[(events["gold_exists"]) & (events["gold_label"] == label)]
        spurious = events[(events["strict_is_fp"]) & (events["pred_label"] == label)]
        support = len(gold)
        rows.append({
            "label": label,
            "training_support": int((training["label"].astype(str) == label).sum()),
            "evaluation_support": support,
            **{key: overall.get(key) for key in overall if key not in {"family", "value", "label"}},
            "exact_seen_rate": _boolean_mean(gold, "train_exact_seen"),
            "normalized_seen_rate": _boolean_mean(gold, "train_normalized_seen"),
            "fuzzy_seen_rate": 1 - _boolean_mean(gold, "zero_shot_fuzzy"),
            "zero_shot_rate": _boolean_mean(gold, "zero_shot_default"),
            "mean_train_similarity": _mean(gold["train_nearest_similarity"]),
            "median_train_similarity": _median(gold["train_nearest_similarity"]),
            "miss_count": int(gold["strict_is_fn"].sum()),
            "miss_rate": float(gold["strict_is_fn"].mean()) if support else 0.0,
            "spurious_count": int(len(spurious)),
            "boundary_error_count": int(gold["error_primary"].isin({"BOUNDARY_ERROR", "BOUNDARY_AND_LABEL_ERROR"}).sum()),
            "label_error_count": int(gold["error_primary"].isin({"LABEL_ERROR", "BOUNDARY_AND_LABEL_ERROR"}).sum()),
            "seen_support": seen.get("support", 0), "zero_shot_support": unseen.get("support", 0),
            "strict_f1_seen_zero_shot_gap": _difference(seen, unseen, "strict_f1"),
            "char_f1_seen_zero_shot_gap": _difference(seen, unseen, "char_f1"),
        })
    return rows


def dataset_shift(training: pd.DataFrame, gold: pd.DataFrame) -> dict:
    """Descriptive annotation distributions only; no causal interpretation."""
    return {
        "training": _annotation_distribution(training),
        "evaluation": _annotation_distribution(gold),
    }


def _annotation_distribution(frame: pd.DataFrame) -> dict:
    count = len(frame)
    labels = frame["label"].astype(str).value_counts()
    lengths = frame["text"].astype(str).str.len()
    acronym = frame["text"].astype(str).map(
        lambda text: bool(text and len(text) <= 12 and sum(c.isupper() for c in text if c.isalpha()) >= max(1, sum(c.isalpha() for c in text) * 0.6))
    )
    return {
        "support": int(count),
        "label_distribution": {str(key): int(value) for key, value in labels.items()},
        "mean_character_length": float(lengths.mean()) if count else 0.0,
        "median_character_length": float(lengths.median()) if count else 0.0,
        "acronym_prevalence": float(acronym.mean()) if count else 0.0,
        "digit_prevalence": float(frame["text"].astype(str).str.contains(r"\d").mean()) if count else 0.0,
        "punctuation_prevalence": float(frame["text"].astype(str).str.contains(r"[^\w\s]").mean()) if count else 0.0,
    }


def error_intersections(events: pd.DataFrame, minimum_support: int) -> list[dict]:
    flags = {
        "ZERO_SHOT": events["zero_shot_default"].fillna(False),
        "RARE": events["train_exact_frequency"].fillna(0).le(1),
        "ACRONYM": events["entity_acronym_like"].fillna(False),
        "LONG_ENTITY": events["entity_whitespace_token_count"].fillna(0).ge(5),
        "TOKEN_FRAGMENTED": events["tokenization_subtoken_to_word_ratio"].fillna(0).gt(2),
        "OVERLAPPING": events["gold_overlaps_other_gold"].fillna(False),
        "HIGH_CONFIDENCE": events["confidence_high_error"].fillna(False),
    }
    errors = {
        name: events["error_primary"] == name
        for name in ("MISSED", "SPURIOUS", "BOUNDARY_ERROR", "LABEL_ERROR", "BOUNDARY_AND_LABEL_ERROR")
    }
    rows = []
    for flag_name, flag in flags.items():
        for error_name, error in errors.items():
            count = int((flag & error).sum())
            if count >= minimum_support:
                rows.append({"feature": flag_name, "error": error_name, "support": count})
    return rows


def factual_summary(metrics: dict, events: pd.DataFrame, subgroup_rows: list[dict], confusion: dict) -> dict:
    gold = events[events["gold_exists"]]
    exposure = {
        "exact_seen_rate": _boolean_mean(gold, "train_exact_seen"),
        "normalized_seen_rate": _boolean_mean(gold, "train_normalized_seen"),
        "fuzzy_seen_rate": 1 - _boolean_mean(gold, "zero_shot_fuzzy"),
        "zero_shot_rate": _boolean_mean(gold, "zero_shot_default"),
    }
    lookup = {(row["family"], row["value"]): row for row in subgroup_rows if row.get("label") is None}
    seen = lookup.get(("zero_shot_default", "false"), {})
    unseen = lookup.get(("zero_shot_default", "true"), {})
    error_counts = gold.loc[gold["error_primary"] != "CORRECT", "error_primary"].value_counts()
    boundary_counts = gold["span_boundary_error"].value_counts()
    miss_counts = gold[gold["strict_is_fn"]]["gold_label"].value_counts()
    zero_mask = gold["zero_shot_default"].fillna(False).astype(bool)
    zero_counts = gold.loc[zero_mask, "gold_label"].value_counts()
    confusions = confusion["exact_boundary_label_confusion"]
    return {
        "overall": metrics,
        **exposure,
        "seen": seen,
        "zero_shot": unseen,
        "strict_f1_seen_zero_shot_gap": _difference(seen, unseen, "strict_f1"),
        "char_f1_seen_zero_shot_gap": _difference(seen, unseen, "char_f1"),
        "most_frequent_error_type": _first(error_counts),
        "most_common_boundary_deviation": _first(boundary_counts),
        "label_with_most_misses": _first(miss_counts),
        "label_with_most_zero_shot_examples": _first(zero_counts),
        "most_frequent_label_confusion": max(confusions, key=lambda row: row["count"], default=None),
    }


def _paired_pred_label(events, row):
    matches = events[(events["error_pair_id"] == row.error_pair_id) & (events["error_pair_role"] == "PRED")]
    return str(matches.iloc[0].pred_label)


def _mean(series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else None


def _median(series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if len(values) else None


def _boolean_mean(frame, column) -> float:
    values = frame[column].dropna().astype(bool)
    return float(values.mean()) if len(values) else 0.0


def _difference(first, second, key):
    return float(first[key] - second[key]) if key in first and key in second else None


def _first(counts):
    return None if counts.empty else {"value": str(counts.index[0]), "count": int(counts.iloc[0])}
