"""Representation similarity methods: linear CKA, SVCCA, and PWCCA."""

from __future__ import annotations

import numpy as np


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Feature-space linear CKA, invariant to isotropic scaling and orthogonal transforms."""
    x, y = _paired(x, y)
    x = x.astype(np.float64) - x.mean(axis=0)
    y = y.astype(np.float64) - y.mean(axis=0)
    cross = np.linalg.norm(x.T @ y, ord="fro") ** 2
    normalizer = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    return float(cross / normalizer) if normalizer > 0 else np.nan


class StreamingLinearCKA:
    """Sufficient-statistic CKA accumulator that never stores observation matrices."""

    def __init__(self, features_x: int, features_y: int):
        self.count = 0
        self.sum_x = np.zeros(features_x, dtype=np.float64)
        self.sum_y = np.zeros(features_y, dtype=np.float64)
        self.xx = np.zeros((features_x, features_x), dtype=np.float64)
        self.yy = np.zeros((features_y, features_y), dtype=np.float64)
        self.xy = np.zeros((features_x, features_y), dtype=np.float64)

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        x, y = _paired(x, y)
        self.count += len(x)
        self.sum_x += x.sum(axis=0)
        self.sum_y += y.sum(axis=0)
        self.xx += x.T @ x
        self.yy += y.T @ y
        self.xy += x.T @ y

    def score(self) -> float:
        if self.count < 2:
            return np.nan
        centered_xx = self.xx - np.outer(self.sum_x, self.sum_x) / self.count
        centered_yy = self.yy - np.outer(self.sum_y, self.sum_y) / self.count
        centered_xy = self.xy - np.outer(self.sum_x, self.sum_y) / self.count
        numerator = np.linalg.norm(centered_xy, "fro") ** 2
        denominator = np.linalg.norm(centered_xx, "fro") * np.linalg.norm(centered_yy, "fro")
        return float(numerator / denominator) if denominator else np.nan


def svcca(x: np.ndarray, y: np.ndarray, variance_retained: float = 0.99) -> float:
    """SVCCA mean canonical correlation after SVD variance truncation."""
    x, y = _paired(x, y)
    reduced_x = _svd_reduce(x, variance_retained)
    reduced_y = _svd_reduce(y, variance_retained)
    correlations, _, _ = _cca(reduced_x, reduced_y)
    return float(np.mean(correlations)) if len(correlations) else np.nan


def pwcca(x: np.ndarray, y: np.ndarray, variance_retained: float = 0.99) -> float:
    """Projection-weighted CCA using activation projection magnitudes as weights."""
    x, y = _paired(x, y)
    reduced_x = _svd_reduce(x, variance_retained)
    reduced_y = _svd_reduce(y, variance_retained)
    correlations, directions_x, _ = _cca(reduced_x, reduced_y)
    if not len(correlations):
        return np.nan
    canonical = (reduced_x - reduced_x.mean(axis=0)) @ directions_x
    weights = np.sum(np.abs(canonical), axis=0)
    weights /= weights.sum() if weights.sum() else 1.0
    return float(weights @ correlations)


def _svd_reduce(values: np.ndarray, retained: float) -> np.ndarray:
    centered = values.astype(np.float64) - values.mean(axis=0)
    u, singular, _ = np.linalg.svd(centered, full_matrices=False)
    energy = singular**2
    rank = int(np.searchsorted(np.cumsum(energy) / max(energy.sum(), 1e-15), retained) + 1)
    return u[:, :rank] * singular[:rank]


def _cca(x: np.ndarray, y: np.ndarray, regularization: float = 1e-8):
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    cxx = x.T @ x / max(len(x) - 1, 1) + regularization * np.eye(x.shape[1])
    cyy = y.T @ y / max(len(y) - 1, 1) + regularization * np.eye(y.shape[1])
    cxy = x.T @ y / max(len(x) - 1, 1)
    wx = _inverse_sqrt(cxx)
    wy = _inverse_sqrt(cyy)
    left, correlations, right = np.linalg.svd(wx @ cxy @ wy, full_matrices=False)
    return np.clip(correlations, 0, 1), wx @ left, wy @ right.T


def _inverse_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * (1.0 / np.sqrt(np.maximum(values, 1e-12)))) @ vectors.T


def _paired(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("Similarity inputs must be 2-D with equal sample counts.")
    return x, y
