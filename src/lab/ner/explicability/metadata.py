"""Stable observation identity and token-level analytical metadata."""

from __future__ import annotations

from typing import Any

import numpy as np

from lab.ner.encoding.rows import IGNORE_INDEX
from lab.ner.evaluation.spans import content_positions, ensure_int_list, ensure_offsets
from lab.ner.explicability.utils import stable_id


def token_metadata(
    row: Any,
    row_index: int,
    tokenizer,
    logits: np.ndarray,
    positions: list[int],
    id2label: dict[int, str],
) -> list[dict[str, Any]]:
    """Build metadata for selected model positions without repeating document text."""
    input_ids = ensure_int_list(row.input_ids)
    labels = ensure_int_list(row.labels)
    offsets = ensure_offsets(row.token_offsets) if hasattr(row, "token_offsets") else []
    words = ensure_int_list(row.word_ids) if hasattr(row, "word_ids") else list(range(len(offsets)))
    content = content_positions(input_ids, tokenizer)
    aligned = {position: (offsets[i], words[i]) for i, position in enumerate(content)}
    doc_id = str(getattr(row, "doc_id", row_index))
    window_start = int(getattr(row, "window_start", 0))
    window_end = int(getattr(row, "window_end", window_start))
    example_id = stable_id("example", doc_id, window_start, row_index)
    segment_id = stable_id("segment", doc_id, window_start, window_end)
    records = []

    for position in positions:
        token_id = int(input_ids[position])
        offset, word_id = aligned.get(position, ((None, None), None))
        probabilities = _softmax(logits[position])
        predicted = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted])
        best_non_gold = (
            float(np.max(np.delete(probabilities, gold_id)))
            if gold_id != IGNORE_INDEX and len(probabilities) > 1 else None
        )
        entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))
        gold_id = int(labels[position]) if position < len(labels) else IGNORE_INDEX
        gold_label = id2label.get(gold_id) if gold_id != IGNORE_INDEX else None
        predicted_label = id2label.get(predicted, str(predicted))
        start, end = offset
        word_observation_id = (
            stable_id("word", doc_id, start, end, word_id) if word_id is not None else None
        )
        records.append({
            "observation_id": stable_id("subword", doc_id, window_start, position, token_id, start, end),
            "example_id": example_id,
            "segment_id": segment_id,
            "document_id": doc_id,
            "row_index": row_index,
            "window_start": window_start,
            "token_position": position,
            "word_id": word_id,
            "word_observation_id": word_observation_id,
            "token_id": token_id,
            "token": tokenizer.convert_ids_to_tokens(token_id),
            "char_start": start,
            "char_end": end,
            "gold_label": gold_label,
            "predicted_label": predicted_label,
            "gold_entity_type": _entity_type(gold_label),
            "predicted_entity_type": _entity_type(predicted_label),
            "logits": logits[position].astype(np.float32, copy=False).tolist(),
            "probabilities": probabilities.astype(np.float32, copy=False).tolist(),
            "confidence": confidence,
            "gold_confidence": (
                float(probabilities[gold_id]) if gold_id != IGNORE_INDEX else None
            ),
            "entropy": entropy,
            "gold_vs_second_best_margin": (
                float(probabilities[gold_id] - best_non_gold)
                if best_non_gold is not None else None
            ),
            "correct": predicted == gold_id if gold_id != IGNORE_INDEX else None,
        })
    return records


def entity_id(source: str, document_id: str, start: int, end: int, label: str) -> str:
    if source not in {"gold", "predicted"}:
        raise ValueError("Entity source must be gold or predicted.")
    return stable_id(f"{source}_entity", document_id, int(start), int(end), label)


def _softmax(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    shifted = values - values.max()
    result = np.exp(shifted)
    return result / result.sum()


def _entity_type(label: str | None) -> str | None:
    if label is None or label == "O":
        return None
    return label.partition("-")[2] or label
