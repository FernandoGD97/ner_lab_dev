"""Compact nearest-neighbor identities and longitudinal neighborhood turnover."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lab.ner.explicability.analyses.geometry import entropy_from_labels, top_k_neighbors


def neighborhood_evolution(
    checkpoints: list[str], ids: np.ndarray, tensors: list[np.ndarray], labels: np.ndarray,
    k: int = 10, block_size: int = 1024,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = ids.astype(str)
    labels = labels.astype(str)
    neighbor_sets: list[list[set[str]]] = []
    detail_rows, summary_rows = [], []
    for time, (checkpoint, values) in enumerate(zip(checkpoints, tensors)):
        indices, distances = top_k_neighbors(values, k, block_size)
        sets = []
        for row, observation_id in enumerate(ids):
            neighbor_ids = ids[indices[row]]
            neighbor_labels = labels[indices[row]]
            current = set(neighbor_ids.tolist())
            sets.append(current)
            for rank, (neighbor_id, distance) in enumerate(zip(neighbor_ids, distances[row]), 1):
                detail_rows.append({
                    "checkpoint": checkpoint, "observation_id": observation_id,
                    "neighbor_id": neighbor_id, "rank": rank, "distance": float(distance),
                    "similarity": float(1.0 - distance),
                    "same_label": bool(labels[row] == labels[indices[row, rank - 1]]),
                })
            previous = sets[row] if time == 0 else neighbor_sets[time - 1][row]
            baseline = sets[row] if time == 0 else neighbor_sets[0][row]
            summary_rows.append({
                "checkpoint": checkpoint, "observation_id": observation_id,
                "k": len(current), "label_purity": float(np.mean(neighbor_labels == labels[row])),
                "neighborhood_entropy": entropy_from_labels(neighbor_labels),
                "jaccard_previous": _jaccard(current, previous),
                "jaccard_pretrained": _jaccard(current, baseline),
                "neighborhood_turnover": 1.0 - _jaccard(current, previous),
            })
        neighbor_sets.append(sets)
    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0
