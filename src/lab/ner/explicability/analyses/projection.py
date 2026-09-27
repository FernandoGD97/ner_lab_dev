"""Temporally aligned projections fitted once across checkpoints."""

from __future__ import annotations

import numpy as np
import pandas as pd


def shared_pca(
    representations: dict[str, tuple[np.ndarray, np.ndarray]], dimensions: int = 2
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Fit one PCA basis to all checkpoints and transform every state in that basis."""
    if dimensions < 1:
        raise ValueError("dimensions must be positive.")
    ordered = list(representations)
    if not ordered:
        return pd.DataFrame(), {}
    common = set(representations[ordered[0]][0].astype(str))
    for ids, _ in representations.values():
        common &= set(ids.astype(str))
    common_ids = np.asarray(sorted(common), dtype=str)
    aligned = {}
    for checkpoint, (ids, values) in representations.items():
        lookup = {item: index for index, item in enumerate(ids.astype(str))}
        aligned[checkpoint] = values[[lookup[item] for item in common_ids]].astype(np.float64)
    count = sum(len(values) for values in aligned.values())
    mean = sum(values.sum(axis=0) for values in aligned.values()) / max(count, 1)
    covariance = sum((values - mean).T @ (values - mean) for values in aligned.values())
    covariance /= max(count - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    basis = eigenvectors[:, np.argsort(eigenvalues)[::-1][:dimensions]]
    rows, transformed = [], {}
    for checkpoint in ordered:
        coordinates = (aligned[checkpoint] - mean) @ basis
        transformed[checkpoint] = coordinates.astype(np.float32)
        for index, observation_id in enumerate(common_ids):
            row = {"observation_id": observation_id, "checkpoint": checkpoint,
                   "projection_method": "shared_pca"}
            for axis in range(dimensions):
                row["xy"[axis] if axis < 2 else f"component_{axis + 1}"] = float(coordinates[index, axis])
            rows.append(row)
    return pd.DataFrame(rows), {"mean": mean, "basis": basis, "ids": common_ids}


def temporal_projection(
    representations: dict[str, tuple[np.ndarray, np.ndarray]], method: str = "shared_pca",
    dimensions: int = 2, seed: int = 42,
) -> pd.DataFrame:
    """Project snapshots without independently fitting their coordinate systems."""
    if method == "shared_pca":
        return shared_pca(representations, dimensions)[0]
    try:
        import umap
    except ImportError as error:
        raise ImportError(
            f"{method} requires the optional 'umap-learn' dependency; use shared_pca otherwise."
        ) from error
    ordered = list(representations)
    common = sorted(set.intersection(*(set(representations[key][0].astype(str)) for key in ordered)))
    matrices = []
    for checkpoint in ordered:
        ids, values = representations[checkpoint]
        lookup = {item: index for index, item in enumerate(ids.astype(str))}
        matrices.append(values[[lookup[item] for item in common]])
    if method == "fixed_umap":
        reducer = umap.UMAP(n_components=dimensions, random_state=seed)
        reducer.fit(np.concatenate(matrices, axis=0))
        coordinates = [reducer.transform(matrix) for matrix in matrices]
    elif method == "aligned_umap":
        relations = [{index: index for index in range(len(common))} for _ in range(len(matrices) - 1)]
        coordinates = umap.AlignedUMAP(
            n_components=dimensions, random_state=seed
        ).fit_transform(matrices, relations=relations)
    else:
        raise ValueError(f"Unknown temporal projection method: {method}")
    rows = []
    for checkpoint, matrix in zip(ordered, coordinates):
        for index, observation_id in enumerate(common):
            rows.append({
                "observation_id": observation_id, "checkpoint": checkpoint,
                "projection_method": method, "x": float(matrix[index, 0]),
                "y": float(matrix[index, 1]) if dimensions > 1 else 0.0,
            })
    return pd.DataFrame(rows)
