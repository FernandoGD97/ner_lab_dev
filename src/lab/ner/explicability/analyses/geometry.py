"""Global/class geometry, anisotropy, and intrinsic-dimensionality diagnostics."""

from __future__ import annotations

import numpy as np


def geometry_metrics(
    values: np.ndarray, labels: np.ndarray, k: int = 10, minimum_group_size: int = 3,
    block_size: int = 1024,
) -> dict[str, float | int | str | list[float] | None]:
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels).astype(str)
    count = len(values)
    classes = [label for label in sorted(set(labels)) if np.sum(labels == label) >= minimum_group_size]
    warning = None if len(classes) >= 2 else "fewer_than_two_stable_classes"
    spectrum = covariance_spectrum(values)
    anisotropy = anisotropy_metrics(values, spectrum)
    result: dict[str, float | int | str | list[float] | None] = {
        "sample_count": count, "class_count": len(classes), "warning": warning,
        "effective_rank": anisotropy["effective_rank"],
        "participation_ratio": anisotropy["participation_ratio"],
        "mean_pairwise_cosine": anisotropy["mean_pairwise_cosine"],
        "dominant_component_fraction": anisotropy["dominant_component_fraction"],
        "isotropy_score": anisotropy["isotropy_score"],
        "covariance_spectrum": spectrum.astype(np.float32).tolist(),
    }
    if count < 3 or len(classes) < 2:
        return result
    valid = np.isin(labels, classes)
    x, y = values[valid], labels[valid]
    centroids = {label: x[y == label].mean(axis=0) for label in classes}
    within_sq = sum(np.sum((x[y == label] - centroids[label]) ** 2) for label in classes)
    overall = x.mean(axis=0)
    between_sq = sum(np.sum(y == label) * np.sum((centroids[label] - overall) ** 2) for label in classes)
    result["fisher_discriminant_ratio"] = float(between_sq / max(within_sq, 1e-15))
    result["calinski_harabasz"] = float(
        (between_sq / max(len(classes) - 1, 1)) /
        (within_sq / max(len(x) - len(classes), 1))
    )
    scatters = {label: np.mean(np.linalg.norm(x[y == label] - centroids[label], axis=1)) for label in classes}
    db_terms = []
    for left in classes:
        ratios = [
            (scatters[left] + scatters[right]) /
            max(np.linalg.norm(centroids[left] - centroids[right]), 1e-15)
            for right in classes if right != left
        ]
        db_terms.append(max(ratios))
    result["davies_bouldin"] = float(np.mean(db_terms))
    neighbors, distances = top_k_neighbors(x, min(k, len(x) - 1), block_size=block_size)
    neighbor_labels = y[neighbors]
    result["knn_label_purity"] = float(np.mean(neighbor_labels == y[:, None]))
    result["neighborhood_entropy"] = float(np.mean([
        entropy_from_labels(row) for row in neighbor_labels
    ]))
    occurrence = np.bincount(neighbors.ravel(), minlength=len(x))
    result["hubness_skewness"] = float(_skewness(occurrence))
    result["silhouette"] = float(_silhouette_from_neighbors(x, y, block_size))
    same_distances, different_distances = [], []
    for start in range(0, len(x), block_size):
        distance = _euclidean_block(x[start:start + block_size], x)
        same = y[start:start + block_size, None] == y[None, :]
        diagonal_rows = np.arange(start, min(start + block_size, len(x)))
        same[np.arange(len(diagonal_rows)), diagonal_rows] = False
        same_distances.extend(distance[same].tolist())
        different_distances.extend(distance[~same].tolist())
    result["intra_class_euclidean"] = float(np.mean(same_distances))
    result["inter_class_euclidean"] = float(np.mean(different_distances))
    normalized = _normalize(x)
    cosine = 1.0 - normalized @ normalized.T
    same = y[:, None] == y[None, :]
    off_diagonal = ~np.eye(len(y), dtype=bool)
    result["intra_class_cosine"] = float(np.mean(cosine[same & off_diagonal]))
    result["inter_class_cosine"] = float(np.mean(cosine[~same & off_diagonal]))
    return result


def intrinsic_dimension(values: np.ndarray) -> list[dict[str, float | int | str]]:
    values = np.asarray(values, dtype=np.float64)
    spectrum = covariance_spectrum(values)
    participation = float(spectrum.sum() ** 2 / max(np.sum(spectrum**2), 1e-15))
    rows: list[dict[str, float | int | str]] = [{
        "estimator": "participation_ratio", "estimate": participation,
        "sample_count": len(values), "parameter": "covariance_eigenvalues",
    }]
    if len(values) >= 3:
        neighbors, distances = top_k_neighbors(values, 2)
        del neighbors
        ratios = distances[:, 1] / np.maximum(distances[:, 0], 1e-12)
        logs = np.log(np.maximum(ratios, 1.0 + 1e-12))
        estimate = float(1.0 / max(np.mean(logs), 1e-12))
        rows.append({"estimator": "twonn", "estimate": estimate,
                     "sample_count": len(values), "parameter": "k=2"})
    return rows


def anisotropy_metrics(values: np.ndarray, spectrum: np.ndarray | None = None) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    spectrum = covariance_spectrum(values) if spectrum is None else spectrum
    total = max(float(spectrum.sum()), 1e-15)
    probabilities = spectrum[spectrum > 0] / total
    effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    participation = float(total**2 / max(float(np.sum(spectrum**2)), 1e-15))
    normalized = _normalize(values)
    summed = normalized.sum(axis=0)
    pair_count = len(values) * max(len(values) - 1, 1)
    mean_cosine = float((summed @ summed - len(values)) / pair_count)
    dominant = float(spectrum[0] / total) if len(spectrum) else np.nan
    ambient = max(min(len(values) - 1, values.shape[1]), 1)
    return {
        "mean_pairwise_cosine": mean_cosine, "effective_rank": effective_rank,
        "participation_ratio": participation, "dominant_component_fraction": dominant,
        "isotropy_score": float(effective_rank / ambient),
    }


def covariance_spectrum(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return np.zeros(values.shape[1] if values.ndim == 2 else 0)
    centered = values - values.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    spectrum = singular**2 / (len(values) - 1)
    return np.sort(np.maximum(spectrum, 0))[::-1]


def top_k_neighbors(
    values: np.ndarray, k: int, block_size: int = 1024, metric: str = "cosine"
) -> tuple[np.ndarray, np.ndarray]:
    """Exact blockwise kNN without retaining an all-pairs matrix."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        raise ValueError("Nearest-neighbor analysis requires at least two observations.")
    k = min(k, len(values) - 1)
    reference = _normalize(values) if metric == "cosine" else values
    all_indices, all_distances = [], []
    for start in range(0, len(values), block_size):
        query = reference[start:start + block_size]
        distances = 1.0 - query @ reference.T if metric == "cosine" else _euclidean_block(query, reference)
        rows = np.arange(len(query))
        distances[rows, start + rows] = np.inf
        candidates = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        candidate_distances = np.take_along_axis(distances, candidates, axis=1)
        order = np.argsort(candidate_distances, axis=1, kind="stable")
        all_indices.append(np.take_along_axis(candidates, order, axis=1))
        all_distances.append(np.take_along_axis(candidate_distances, order, axis=1))
    return np.vstack(all_indices), np.vstack(all_distances)


def entropy_from_labels(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-15))))


def _silhouette_from_neighbors(values: np.ndarray, labels: np.ndarray, block_size: int) -> float:
    scores = []
    for start in range(0, len(values), block_size):
        distance = _euclidean_block(values[start:start + block_size], values)
        for local, row in enumerate(distance):
            index = start + local
            own = labels == labels[index]
            own[index] = False
            a = row[own].mean() if own.any() else 0.0
            b = min(row[labels == other].mean() for other in set(labels) if other != labels[index])
            scores.append((b - a) / max(a, b, 1e-15))
    return float(np.mean(scores))


def _normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def _euclidean_block(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    squared = np.sum(left**2, axis=1, keepdims=True) + np.sum(right**2, axis=1) - 2 * left @ right.T
    return np.sqrt(np.maximum(squared, 0))


def _skewness(values: np.ndarray) -> float:
    deviation = values - values.mean()
    scale = np.sqrt(np.mean(deviation**2))
    return float(np.mean(deviation**3) / scale**3) if scale else 0.0
