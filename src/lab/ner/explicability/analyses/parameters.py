"""Parameter-space adaptation grouped by Transformer structure."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def parameter_drift(
    baseline: dict[str, np.ndarray], current: dict[str, np.ndarray], checkpoint: str
) -> pd.DataFrame:
    accumulators: dict[tuple[str, str], dict[str, float | int]] = {}
    for name in sorted(set(baseline) & set(current)):
        before = np.asarray(baseline[name], dtype=np.float64)
        after = np.asarray(current[name], dtype=np.float64)
        if before.shape != after.shape:
            continue
        layer, component = classify_parameter(name)
        bucket = accumulators.setdefault((layer, component), {
            "delta_squared": 0.0, "weight_squared": 0.0, "parameter_count": 0
        })
        bucket["delta_squared"] += float(np.sum((after - before) ** 2))
        bucket["weight_squared"] += float(np.sum(before**2))
        bucket["parameter_count"] += before.size
    return _parameter_rows(accumulators, checkpoint)


def parameter_drift_files(
    baseline_path: str | Path, current_path: str | Path, checkpoint: str
) -> pd.DataFrame:
    """Compare compressed states one tensor at a time instead of materializing two models."""
    with np.load(baseline_path, allow_pickle=False) as baseline, np.load(
        current_path, allow_pickle=False
    ) as current:
        accumulators: dict[tuple[str, str], dict[str, float | int]] = {}
        for name in sorted(set(baseline.files) & set(current.files)):
            before = baseline[name].astype(np.float64, copy=False)
            after = current[name].astype(np.float64, copy=False)
            if before.shape != after.shape:
                continue
            layer, component = classify_parameter(name)
            bucket = accumulators.setdefault((layer, component), {
                "delta_squared": 0.0, "weight_squared": 0.0, "parameter_count": 0
            })
            bucket["delta_squared"] += float(np.sum((after - before) ** 2))
            bucket["weight_squared"] += float(np.sum(before**2))
            bucket["parameter_count"] += before.size
            del before, after
    return _parameter_rows(accumulators, checkpoint)


def _parameter_rows(accumulators, checkpoint):
    rows = []
    for (layer, component), totals in sorted(accumulators.items()):
        delta = np.sqrt(totals["delta_squared"])
        weight = np.sqrt(totals["weight_squared"])
        relative = float(delta / weight) if weight else np.nan
        rows.append({"checkpoint": checkpoint, "layer": layer, "component": component,
                     "delta_norm": float(delta), "relative_delta": relative,
                     "update_weight_ratio": relative,
                     "parameter_count": int(totals["parameter_count"])})
    return pd.DataFrame(rows)


def classify_parameter(name: str) -> tuple[str, str]:
    lower = name.lower()
    match = re.search(r"(?:layer|layers|block|h)\.(\d+)", lower)
    layer = f"layer_{int(match.group(1)):02d}" if match else "global"
    if "embed" in lower:
        return "embeddings", "embeddings"
    if any(term in lower for term in ("classifier", "score", "crf")):
        return "classification_head", "classification_head"
    if any(term in lower for term in ("attention", "query", "key", "value")):
        return layer, "attention"
    if any(term in lower for term in ("intermediate", "mlp", "ffn", "feed_forward")):
        return layer, "ffn"
    if any(term in lower for term in ("norm", "layernorm")):
        return layer, "normalization"
    return layer, "other"
