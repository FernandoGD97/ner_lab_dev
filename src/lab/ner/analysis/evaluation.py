"""Turn official span evaluation into one lossless row per strict contribution."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import pandas as pd
from nervaluate import Evaluator

from lab.core.scoring import group_by_document, normalize_spans, to_nervaluate
from lab.core.spans import SPAN_COLUMNS
from lab.ner.analysis.models import STRICT_FN, STRICT_FP, STRICT_TP
from lab.ner.evaluation import span_metrics

SIDE_COLUMNS = ("filename", "label", "start_span", "end_span", "text")
IDENTITY_COLUMNS = ("filename", "label", "start_span", "end_span")


def evaluate_to_events(
    gold: pd.DataFrame | str | Path,
    predicted: pd.DataFrame | str | Path,
    *,
    run_id: str,
    tags: list[str] | None = None,
    min_overlap_percentage: float = 40.0,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """
    Evaluate with the existing scorer and expose its STRICT accounting as events.

    Exact gold/prediction identities become a single TP row. Every remaining
    prediction and gold entity becomes an FP and FN respectively. This does not
    assign partial credit or diagnostically pair errors. Most importantly, the
    resulting contributions are checked against the official evaluator before
    they are returned, preventing an analytical artifact from silently defining
    different scores.
    """
    if not str(run_id).strip():
        raise ValueError("run_id must be a non-empty string.")

    gold_frame = _read_spans(gold, scored=False)
    predicted_frame = _read_spans(predicted, scored=True)
    labels = tags if tags is not None else sorted(
        set(gold_frame["label"].astype(str)) | set(predicted_frame["label"].astype(str))
    )
    metrics = span_metrics(
        gold=gold_frame,
        predicted=predicted_frame,
        tags=labels,
        min_overlap_percentage=min_overlap_percentage,
    )
    correct_predictions = _official_correct_predictions(
        gold_frame, predicted_frame, labels, min_overlap_percentage
    )
    events = _strict_events(gold_frame, predicted_frame, str(run_id), correct_predictions)
    _assert_official_accounting(events, metrics)

    return events, metrics


def read_span_input(value: pd.DataFrame | str | Path, *, scored: bool = False) -> pd.DataFrame:
    """Read and validate a canonical span table for public analysis APIs."""
    return _read_spans(value, scored=scored)


def _read_spans(value: pd.DataFrame | str | Path, *, scored: bool) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
    else:
        path = Path(value)

        if not path.exists():
            raise FileNotFoundError(f"Span input does not exist: {path}")

        frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(
            path, sep="\t"
        )

    # Use the scorer's own normalization as the span contract authority.
    normalized = normalize_spans(frame).rename(
        columns={"off0": "start_span", "off1": "end_span", "span": "text"}
    )
    normalized = normalized[SPAN_COLUMNS]

    if scored and "score" in frame.columns:
        scores = pd.to_numeric(frame["score"], errors="raise").astype("float32")

        if scores.isna().any() or ((scores < 0) | (scores > 1)).any():
            raise ValueError("Prediction scores must be finite values between 0 and 1.")

        normalized["score"] = scores.to_numpy()

    return normalized.reset_index(drop=True)


def _identity(row: Any) -> tuple[str, str, int, int]:
    return (str(row.filename), str(row.label), int(row.start_span), int(row.end_span))


def _strict_events(
    gold: pd.DataFrame,
    predicted: pd.DataFrame,
    run_id: str,
    correct_predictions: set[int],
) -> pd.DataFrame:
    unused_gold: dict[tuple[str, str, int, int], deque[int]] = defaultdict(deque)

    for index, row in enumerate(gold.itertuples(index=False)):
        unused_gold[_identity(row)].append(index)

    records: list[dict[str, Any]] = []
    matched_gold: set[int] = set()

    for pred_index in sorted(correct_predictions):
        pred_row = predicted.iloc[pred_index]
        candidates = unused_gold[_identity(pred_row)]

        if not candidates:
            raise RuntimeError("The official evaluator marked a non-identical span as STRICT correct.")

        gold_index = candidates.popleft()
        matched_gold.add(gold_index)
        records.append(_event(run_id, STRICT_TP, gold_index, gold.iloc[gold_index], pred_index, pred_row))

    for gold_index, gold_row in enumerate(gold.itertuples(index=False)):
        if gold_index not in matched_gold:
            records.append(_event(run_id, STRICT_FN, gold_index, gold_row, None, None))

    for pred_index, pred_row in enumerate(predicted.itertuples(index=False)):
        if pred_index not in correct_predictions:
            records.append(_event(run_id, STRICT_FP, None, None, pred_index, pred_row))

    records.sort(
        key=lambda row: (
            row["document_id"],
            row["gold_start"] if row["gold_exists"] else row["pred_start"],
            row["strict_outcome"],
        )
    )

    for event_id, record in enumerate(records):
        record["event_id"] = event_id

    return pd.DataFrame.from_records(records, columns=_event_columns())


def _official_correct_predictions(
    gold: pd.DataFrame,
    predicted: pd.DataFrame,
    tags: list[str],
    min_overlap_percentage: float,
) -> set[int]:
    """Ask the same nervaluate STRICT strategy which prediction rows were correct."""
    gold_documents = group_by_document(normalize_spans(gold))
    predicted_documents = group_by_document(normalize_spans(predicted))
    doc_ids = sorted(set(gold_documents) | set(predicted_documents))
    evaluator = Evaluator(
        true=[to_nervaluate(gold_documents.get(doc_id, [])) for doc_id in doc_ids],
        pred=[to_nervaluate(predicted_documents.get(doc_id, [])) for doc_id in doc_ids],
        tags=tags,
        loader="dict",
        min_overlap_percentage=min_overlap_percentage,
    )
    evaluated = evaluator.evaluate()
    strict_indices = evaluated.get("overall_indices", {}).get("strict")

    if strict_indices is None:
        return set()

    # nervaluate reports positions within each document. Translate those back
    # to input-frame positions without changing its order-sensitive matching.
    by_document: dict[str, list[int]] = defaultdict(list)

    for frame_index, row in enumerate(predicted.itertuples(index=False)):
        by_document[str(row.filename)].append(frame_index)

    return {
        by_document[doc_ids[doc_index]][prediction_index]
        for doc_index, prediction_index in strict_indices.correct_indices
    }


def _event(run_id, outcome, gold_index, gold, pred_index, pred) -> dict[str, Any]:
    gold_exists = gold is not None
    pred_exists = pred is not None
    document_id = str(gold.filename if gold_exists else pred.filename)

    return {
        "run_id": run_id,
        "event_id": None,
        "document_id": document_id,
        "gold_id": None if gold_index is None else int(gold_index),
        "gold_exists": gold_exists,
        "gold_start": None if not gold_exists else int(gold.start_span),
        "gold_end": None if not gold_exists else int(gold.end_span),
        "gold_text": None if not gold_exists else str(gold.text),
        "gold_label": None if not gold_exists else str(gold.label),
        "gold_labels": None if not gold_exists else [str(gold.label)],
        "pred_id": None if pred_index is None else int(pred_index),
        "pred_exists": pred_exists,
        "pred_start": None if not pred_exists else int(pred.start_span),
        "pred_end": None if not pred_exists else int(pred.end_span),
        "pred_text": None if not pred_exists else str(pred.text),
        "pred_label": None if not pred_exists else str(pred.label),
        "pred_labels": None if not pred_exists else [str(pred.label)],
        "pred_confidence": None if not pred_exists or not hasattr(pred, "score") else float(pred.score),
        "strict_outcome": outcome,
        "strict_is_tp": outcome == STRICT_TP,
        "strict_is_fp": outcome == STRICT_FP,
        "strict_is_fn": outcome == STRICT_FN,
        "strict_correct": outcome == STRICT_TP,
        # Reserved for a later, non-scoring diagnostic relation between FP/FN rows.
        "error_pair_id": None,
        "error_type": None,
    }


def _event_columns() -> list[str]:
    return [
        "run_id", "event_id", "document_id",
        "gold_id", "gold_exists", "gold_start", "gold_end", "gold_text", "gold_label", "gold_labels",
        "pred_id", "pred_exists", "pred_start", "pred_end", "pred_text", "pred_label", "pred_labels", "pred_confidence",
        "strict_outcome", "strict_is_tp", "strict_is_fp", "strict_is_fn", "strict_correct",
        "error_pair_id", "error_type",
    ]


def _assert_official_accounting(events: pd.DataFrame, metrics: dict[str, float | int]) -> None:
    represented = {
        "correct": int(events["strict_is_tp"].sum()),
        "missed": int(events["strict_is_fn"].sum()),
        "spurious": int(events["strict_is_fp"].sum()),
    }
    official = {
        "correct": int(metrics.get("span_strict_correct", 0)),
        "missed": int(metrics.get("span_strict_missed", 0))
        + int(metrics.get("span_strict_incorrect", 0))
        + int(metrics.get("span_strict_partial", 0)),
        "spurious": int(metrics.get("span_strict_spurious", 0))
        + int(metrics.get("span_strict_incorrect", 0))
        + int(metrics.get("span_strict_partial", 0)),
    }

    if represented != official:
        raise RuntimeError(
            "Canonical STRICT events disagree with the official evaluator: "
            f"events={represented}, official={official}."
        )
