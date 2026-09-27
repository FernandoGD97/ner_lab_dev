"""Tokenizer fragmentation statistics measured on the fixed corpus order."""
from __future__ import annotations

import json
import math
import statistics
from typing import Any


def percentile(values: list[int], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return float(ordered[index])


def tokenizer_statistics(tokenizer, corpus) -> dict[str, Any]:
    token_counts: list[int] = []
    word_counts: list[int] = []
    entity_counts: list[int] = []
    fragmentation: dict[str, int] = {}
    for row in corpus[["text", "entities_json"]].itertuples(index=False):
        ids = tokenizer(row.text, add_special_tokens=False)["input_ids"]
        token_counts.append(len(ids))
        word_counts.append(max(1, len(row.text.split())))
        for entity in json.loads(row.entities_json):
            text = row.text[int(entity["start"]):int(entity["end"])]
            pieces = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            entity_counts.append(pieces)
            fragmentation[str(pieces)] = fragmentation.get(str(pieces), 0) + 1
    return {
        "mean_tokens_per_document": statistics.fmean(token_counts) if token_counts else None,
        "median_tokens_per_document": statistics.median(token_counts) if token_counts else None,
        "p95_tokens_per_document": percentile(token_counts, 0.95),
        "mean_subwords_per_word": (
            sum(token_counts) / sum(word_counts) if word_counts else None
        ),
        "mean_subwords_per_entity": (
            statistics.fmean(entity_counts) if entity_counts else None
        ),
        "entity_fragmentation_distribution": fragmentation,
    }
