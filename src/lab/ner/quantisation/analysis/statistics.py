"""Document-paired bootstrap confidence intervals for strict span F1."""
from __future__ import annotations

import random
from typing import Any
import pandas as pd

KEY = ["filename", "start_span", "end_span", "label"]


def _f1(gold: set[tuple], predicted: set[tuple]) -> float:
    tp = len(gold & predicted); fp = len(predicted - gold); fn = len(gold - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _by_document(frame: pd.DataFrame) -> dict[str, set[tuple]]:
    output: dict[str, set[tuple]] = {}
    for row in frame[KEY].itertuples(index=False, name=None):
        output.setdefault(str(row[0]), set()).add(tuple(row[1:]))
    return output


def paired_bootstrap(
    gold: pd.DataFrame,
    baseline: pd.DataFrame,
    compressed: pd.DataFrame,
    iterations: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    if iterations < 1: raise ValueError("iterations must be positive.")
    if not 0 < confidence_level < 1: raise ValueError("confidence_level must be between 0 and 1.")
    sources = [_by_document(frame) for frame in (gold, baseline, compressed)]
    documents = sorted(set().union(*(source for source in sources)))
    if not documents: raise ValueError("Paired bootstrap requires at least one document.")
    rng = random.Random(seed); deltas = []
    for _ in range(iterations):
        sampled = [rng.choice(documents) for _ in documents]
        aggregate = []
        for source in sources:
            values = set()
            for sample_index, document in enumerate(sampled):
                values.update((sample_index, *span) for span in source.get(document, set()))
            aggregate.append(values)
        deltas.append(_f1(aggregate[0], aggregate[2]) - _f1(aggregate[0], aggregate[1]))
    alpha = 1 - confidence_level
    series = pd.Series(deltas)
    aggregate_full = [
        {(document, *span) for document, spans in source.items() for span in spans}
        for source in sources
    ]
    base_f1 = _f1(aggregate_full[0], aggregate_full[1])
    compressed_f1 = _f1(aggregate_full[0], aggregate_full[2])
    return {
        "baseline_f1": base_f1, "compressed_f1": compressed_f1,
        "delta": compressed_f1 - base_f1,
        "confidence_level": confidence_level,
        "ci_low": float(series.quantile(alpha / 2)),
        "ci_high": float(series.quantile(1 - alpha / 2)),
        "iterations": iterations, "seed": seed, "unit": "document",
    }
