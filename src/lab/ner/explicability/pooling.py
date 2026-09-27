"""Pooling primitives for future word- and entity-level analyses."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def pool_groups(
    embeddings: np.ndarray, group_ids: Sequence[object], method: str = "first"
) -> tuple[np.ndarray, list[object]]:
    """Pool rows sharing an ID, retaining first-seen deterministic group order."""
    if len(embeddings) != len(group_ids):
        raise ValueError("embeddings and group_ids must have equal length.")
    if method not in {"first", "mean", "max"}:
        raise ValueError("Pooling method must be first, mean, or max.")
    order = list(dict.fromkeys(group_ids))
    pooled = []
    ids = np.asarray(group_ids, dtype=object)
    for group_id in order:
        values = embeddings[ids == group_id]
        pooled.append(values[0] if method == "first" else getattr(values, method)(axis=0))
    shape = (0, *embeddings.shape[1:])
    return (np.stack(pooled) if pooled else np.empty(shape, dtype=embeddings.dtype), order)


def pool_spans(
    embeddings: np.ndarray,
    token_offsets: Sequence[tuple[int, int]],
    spans: Iterable[tuple[int, int]],
    method: str = "mean",
) -> np.ndarray:
    """Pool tokens overlapping each gold or predicted character-offset span."""
    results = []
    for start, end in spans:
        indices = [i for i, (left, right) in enumerate(token_offsets) if left < end and right > start]
        if not indices:
            raise ValueError(f"Span ({start}, {end}) has no overlapping token embedding.")
        values = embeddings[indices]
        results.append(values[0] if method == "first" else getattr(values, method)(axis=0))
    return np.stack(results) if results else np.empty((0, *embeddings.shape[1:]), dtype=embeddings.dtype)
