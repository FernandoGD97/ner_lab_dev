"""Deterministic frozen-representation linear probes implemented with NumPy."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def probe_targets(gold_labels: np.ndarray, task: str) -> np.ndarray:
    labels = gold_labels.astype(str)
    if task == "entity_vs_non_entity":
        return np.where(labels == "O", "non_entity", "entity")
    if task == "bio_boundary":
        return np.asarray([label.partition("-")[0] if "-" in label else label for label in labels])
    if task == "entity_type":
        return np.asarray([label.partition("-")[2] if "-" in label else label for label in labels])
    raise ValueError(f"Unknown probe task: {task}")


def fixed_split(ids: np.ndarray, seed: int, train_fraction: float, validation_fraction: float) -> np.ndarray:
    """Hash IDs into splits so every checkpoint and layer uses exactly the same partition."""
    boundaries = (train_fraction, train_fraction + validation_fraction)
    values = []
    for observation_id in ids.astype(str):
        digest = hashlib.sha256(f"{seed}:{observation_id}".encode()).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        values.append("train" if value < boundaries[0] else "validation" if value < boundaries[1] else "test")
    return np.asarray(values)


def fit_probe(
    values: np.ndarray, targets: np.ndarray, splits: np.ndarray,
    iterations: int = 200, learning_rate: float = 0.1, regularization: float = 1e-4,
) -> dict[str, float | int | list[str]]:
    """Fit multinomial logistic regression on train only and report held-out test metrics."""
    values = np.asarray(values, dtype=np.float64)
    targets = targets.astype(str)
    classes = np.asarray(sorted(set(targets)))
    class_index = {label: index for index, label in enumerate(classes)}
    encoded = np.asarray([class_index[label] for label in targets])
    train = splits == "train"
    test = splits == "test"
    if train.sum() < len(classes) or test.sum() == 0:
        return {"precision": np.nan, "recall": np.nan, "f1": np.nan,
                "accuracy": np.nan, "sample_count": int(test.sum()),
                "classes": classes.tolist(), "warning": "insufficient_split"}
    mean = values[train].mean(axis=0)
    scale = values[train].std(axis=0)
    scale[scale == 0] = 1.0
    normalized = (values - mean) / scale
    weights = np.zeros((values.shape[1] + 1, len(classes)), dtype=np.float64)
    design = np.column_stack([normalized[train], np.ones(train.sum())])
    one_hot = np.eye(len(classes))[encoded[train]]
    for _ in range(iterations):
        probabilities = _softmax(design @ weights)
        gradient = design.T @ (probabilities - one_hot) / len(design)
        gradient[:-1] += regularization * weights[:-1]
        weights -= learning_rate * gradient
    test_design = np.column_stack([normalized[test], np.ones(test.sum())])
    predicted = np.argmax(test_design @ weights, axis=1)
    gold = encoded[test]
    precision, recall, f1 = _macro_metrics(gold, predicted, len(classes))
    return {"precision": precision, "recall": recall, "f1": f1,
            "accuracy": float(np.mean(gold == predicted)), "sample_count": int(test.sum()),
            "classes": classes.tolist(), "warning": None}


def run_probes(
    checkpoint: str, layer: int, ids: np.ndarray, values: np.ndarray, gold_labels: np.ndarray,
    seed: int = 42, train_fraction: float = 0.7, validation_fraction: float = 0.15,
    iterations: int = 200, learning_rate: float = 0.1,
) -> pd.DataFrame:
    splits = fixed_split(ids, seed, train_fraction, validation_fraction)
    rows = []
    for task in ("entity_vs_non_entity", "bio_boundary", "entity_type"):
        metrics = fit_probe(
            values, probe_targets(gold_labels, task), splits, iterations, learning_rate
        )
        rows.append({"checkpoint": checkpoint, "layer": layer, "probe_task": task,
                     "seed": seed, **metrics})
    return pd.DataFrame(rows)


def mdl_probe(
    ids: np.ndarray, values: np.ndarray, targets: np.ndarray, seed: int = 42,
    fractions: tuple[float, ...] = (0.1, 0.2, 0.4, 0.8), regularization: float = 1e-3,
) -> dict[str, float | int | str]:
    """Online-code MDL estimate using deterministic ridge-linear probabilistic probes."""
    ids, values, targets = ids.astype(str), np.asarray(values, float), targets.astype(str)
    classes = np.asarray(sorted(set(targets)))
    encoded = np.asarray([{label: i for i, label in enumerate(classes)}[label] for label in targets])
    order = np.argsort([
        hashlib.sha256(f"mdl:{seed}:{item}".encode()).hexdigest() for item in ids
    ], kind="stable")
    x = values[order]
    y = encoded[order]
    boundaries = sorted(set([max(1, int(len(x) * fraction)) for fraction in fractions] + [len(x)]))
    uniform_bits = boundaries[0] * np.log2(max(len(classes), 1))
    online_bits = uniform_bits
    for train_end, test_end in zip(boundaries[:-1], boundaries[1:]):
        train_x = np.column_stack([x[:train_end], np.ones(train_end)])
        target = np.eye(len(classes))[y[:train_end]]
        gram = train_x.T @ train_x + regularization * np.eye(train_x.shape[1])
        weights = np.linalg.solve(gram, train_x.T @ target)
        test_x = np.column_stack([x[train_end:test_end], np.ones(test_end - train_end)])
        probabilities = _softmax(test_x @ weights)
        online_bits += float(-np.log2(np.maximum(probabilities[np.arange(len(test_x)), y[train_end:test_end]], 1e-12)).sum())
    uniform_total = len(x) * np.log2(max(len(classes), 1))
    return {
        "online_code_bits": float(online_bits), "uniform_code_bits": float(uniform_total),
        "compression": float(uniform_total / online_bits) if online_bits else np.nan,
        "sample_count": len(x), "fractions": ",".join(map(str, fractions)),
        "interpretation": "decodability_not_causality",
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(logits)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _macro_metrics(gold: np.ndarray, predicted: np.ndarray, classes: int) -> tuple[float, float, float]:
    precision, recall, f1 = [], [], []
    for label in range(classes):
        tp = np.sum((gold == label) & (predicted == label))
        fp = np.sum((gold != label) & (predicted == label))
        fn = np.sum((gold == label) & (predicted != label))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(precision)), float(np.mean(recall)), float(np.mean(f1))
