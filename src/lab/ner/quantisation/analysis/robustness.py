"""NER long-tail and fragmentation analysis from stored gold/prediction spans."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .errors import error_taxonomy

FREQUENCY_DEFINITION = "casefolded, whitespace-normalized gold mention string in the supplied training annotations"


def normalize_mention(text: str) -> str:
    return " ".join(str(text).casefold().split())


def frequency_bucket(count: int) -> str:
    if count == 0: return "unseen"
    if count == 1: return "singleton"
    if count <= 5: return "2-5"
    if count <= 20: return "6-20"
    return ">20"


def seen_unseen(count: int) -> str:
    return "seen" if count > 0 else "unseen"


def subword_bucket(count: int | None) -> str:
    if count is None: return "unavailable"
    if count == 1: return "1"
    if count == 2: return "2"
    if count <= 4: return "3-4"
    return "5+"


def length_bucket(length: int) -> str:
    if length <= 5: return "short"
    if length <= 15: return "medium"
    return "long"


def training_frequencies(training: pd.DataFrame) -> Counter[str]:
    if "text" not in training:
        raise ValueError("Training-frequency source must contain a text column.")
    return Counter(normalize_mention(text) for text in training["text"])


def annotate_gold(
    gold: pd.DataFrame,
    frequencies: Counter[str] | None = None,
    tokenizer: Any | None = None,
) -> pd.DataFrame:
    frame = gold.copy()
    frame["character_length"] = frame["text"].astype(str).str.len()
    frame["word_length"] = frame["text"].astype(str).map(lambda text: max(1, len(re.findall(r"\S+", text))))
    frame["mention_length_bucket"] = frame["character_length"].map(length_bucket)
    if frequencies is None:
        frame["training_frequency"] = None
        frame["frequency_bucket"] = "unavailable"
        frame["seen_unseen"] = "unavailable"
    else:
        frame["training_frequency"] = frame["text"].map(
            lambda text: frequencies[normalize_mention(text)]
        )
        frame["frequency_bucket"] = frame["training_frequency"].map(frequency_bucket)
        frame["seen_unseen"] = frame["training_frequency"].map(seen_unseen)
    if tokenizer is None:
        frame["subword_count"] = None
    else:
        frame["subword_count"] = frame["text"].map(
            lambda text: len(tokenizer(str(text), add_special_tokens=False)["input_ids"])
        )
    frame["subwords_per_word"] = frame.apply(
        lambda row: None if row.subword_count is None else row.subword_count / row.word_length, axis=1
    )
    frame["subword_bucket"] = frame["subword_count"].map(subword_bucket)
    return frame


def _scores(errors: pd.DataFrame) -> dict[str, float | int]:
    tp = int((errors.error_type == "correct").sum())
    fn = int((errors.error_type.isin({
        "false_negative", "wrong_label", "left_boundary_error", "right_boundary_error",
        "both_boundaries_error", "overlapping_prediction",
    })).sum())
    fp = int((errors.error_type == "false_positive").sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn, "tp": tp, "fp": fp, "fn": fn}


def robustness_table(
    gold: pd.DataFrame,
    predicted: pd.DataFrame,
    frequencies: Counter[str] | None = None,
    tokenizer: Any | None = None,
) -> pd.DataFrame:
    annotated = annotate_gold(gold, frequencies, tokenizer)
    errors = error_taxonomy(gold, predicted)
    joined = errors[errors.gold_label.notna()].merge(
        annotated.rename(columns={"filename": "document_id", "start_span": "start", "end_span": "end", "label": "gold_label"}),
        on=["document_id", "start", "end", "gold_label"], how="left", suffixes=("", "_gold"),
    )
    false_positives = errors[errors.error_type == "false_positive"]
    if not false_positives.empty:
        fp_spans = false_positives.rename(columns={
            "document_id": "filename", "start": "start_span", "end": "end_span",
            "prediction_label": "label",
        })[["filename", "start_span", "end_span", "label", "text"]]
        fp_attributes = annotate_gold(fp_spans, frequencies, tokenizer).rename(columns={
            "filename": "document_id", "start_span": "start", "end_span": "end",
            "label": "gold_label",
        })
        fp_attributes["gold_label"] = false_positives["prediction_label"].values
        false_positives = false_positives.drop(columns=["gold_label"]).merge(
            fp_attributes, on=["document_id", "start", "end", "text"], how="left"
        )
        joined = pd.concat([joined, false_positives], ignore_index=True, sort=False)
    output = []
    dimensions = {
        "entity_type": "gold_label", "character_length": "mention_length_bucket",
        "token_length": "word_length", "subwords": "subword_bucket",
        "frequency": "frequency_bucket", "seen_unseen": "seen_unseen",
    }
    for dimension, column in dimensions.items():
        for bucket, group in joined.groupby(column, dropna=False):
            scores = _scores(group)
            output.append({"dimension": dimension, "bucket": str(bucket), **scores})
    return pd.DataFrame(output)


def baseline_deltas(tables: dict[str, pd.DataFrame], baseline: str = "A0") -> pd.DataFrame:
    if baseline not in tables: raise ValueError(f"Missing baseline robustness table {baseline!r}.")
    base = tables[baseline][["dimension", "bucket", "f1"]].rename(columns={"f1": "baseline_f1"})
    output = []
    for model_id, table in tables.items():
        merged = table.merge(base, on=["dimension", "bucket"], how="left")
        merged["model_id"] = model_id
        merged["delta_f1"] = merged["f1"] - merged["baseline_f1"]
        output.append(merged)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()
