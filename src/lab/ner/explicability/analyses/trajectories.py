"""Longitudinal representation movement and entity centroid dynamics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def trajectory_metrics(
    checkpoints: list[str], ids: np.ndarray, tensors: list[np.ndarray]
) -> pd.DataFrame:
    """Compute per-observation steps and whole-path diagnostics in stable ID order."""
    if len(checkpoints) != len(tensors) or len(tensors) < 2:
        raise ValueError("Trajectory analysis requires matching names and at least two states.")
    stack = np.stack([np.asarray(values, dtype=np.float64) for values in tensors])
    if stack.shape[1] != len(ids):
        raise ValueError("Every state must contain one row per observation ID.")
    steps = np.linalg.norm(np.diff(stack, axis=0), axis=2)
    cumulative = np.cumsum(steps, axis=0)
    net = np.linalg.norm(stack[-1] - stack[0], axis=1)
    maximum = np.max(steps, axis=0)
    maximum_at = np.argmax(steps, axis=0) + 1
    rows = []
    for time, checkpoint in enumerate(checkpoints):
        previous = stack[time - 1] if time else stack[time]
        dots = np.sum(stack[time] * previous, axis=1)
        denominators = np.linalg.norm(stack[time], axis=1) * np.linalg.norm(previous, axis=1)
        cosine = np.divide(dots, denominators, out=np.ones_like(dots), where=denominators > 0)
        cosine = np.clip(cosine, -1.0, 1.0)
        for index, observation_id in enumerate(ids.astype(str)):
            path = float(cumulative[time - 1, index]) if time else 0.0
            displacement = float(np.linalg.norm(stack[time, index] - stack[0, index]))
            rows.append({
                "observation_id": observation_id, "checkpoint": checkpoint,
                "distance_previous": float(steps[time - 1, index]) if time else 0.0,
                "cosine_similarity_previous": float(cosine[index]),
                "cosine_distance_previous": float(1.0 - cosine[index]),
                "angular_change_radians": float(np.arccos(cosine[index])),
                "distance_pretrained": displacement,
                "distance_final": float(np.linalg.norm(stack[time, index] - stack[-1, index])),
                "cumulative_path_length": path,
                "net_displacement": displacement,
                "path_displacement_ratio": path / displacement if displacement > 0 else 1.0,
                "maximum_single_step": float(maximum[index]),
                "maximum_movement_checkpoint": checkpoints[int(maximum_at[index])],
                "final_net_displacement": float(net[index]),
            })
    return pd.DataFrame(rows)


def centroid_dynamics(
    checkpoints: list[str], tensors: list[np.ndarray], labels: np.ndarray,
    minimum_group_size: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Track class centroids, compactness, velocity, acceleration, and separation."""
    classes = sorted(set(labels.astype(str)))
    centroids: dict[tuple[int, str], np.ndarray] = {}
    rows, pair_rows = [], []
    for time, (checkpoint, values) in enumerate(zip(checkpoints, tensors)):
        values = np.asarray(values, dtype=np.float64)
        groups = {}
        for label in classes:
            group = values[labels.astype(str) == label]
            groups[label] = group
            if len(group) < minimum_group_size:
                rows.append({"checkpoint": checkpoint, "class_label": label,
                             "sample_count": len(group), "warning": "insufficient_group"})
                continue
            centroid = group.mean(axis=0)
            centroids[time, label] = centroid
        for label in classes:
            group = groups[label]
            if len(group) < minimum_group_size:
                continue
            centroid = centroids[time, label]
            distances = np.linalg.norm(group - centroid, axis=1)
            previous = centroids.get((time - 1, label), centroid)
            velocity = centroid - previous
            earlier = centroids.get((time - 2, label), previous)
            acceleration = velocity - (previous - earlier)
            competitors = [
                (other, np.linalg.norm(centroid - centroids[time, other]))
                for other in classes if other != label and (time, other) in centroids
            ]
            nearest, nearest_distance = min(competitors, key=lambda item: item[1]) if competitors else (None, np.nan)
            rows.append({
                "checkpoint": checkpoint, "class_label": label, "sample_count": len(group),
                "centroid": centroid.astype(np.float32).tolist(),
                "displacement_pretrained": float(np.linalg.norm(centroid - centroids.get((0, label), centroid))),
                "displacement_previous": float(np.linalg.norm(velocity)),
                "centroid_velocity": float(np.linalg.norm(velocity)),
                "centroid_acceleration": float(np.linalg.norm(acceleration)),
                "within_class_radius": float(distances.mean()),
                "within_class_variance": float(np.mean(np.sum((group - centroid) ** 2, axis=1))),
                "nearest_competing_class": nearest, "nearest_centroid_distance": nearest_distance,
                "inter_class_margin": float(nearest_distance - distances.mean()) if competitors else np.nan,
                "warning": None,
            })
        for left_index, left in enumerate(classes):
            for right in classes[left_index + 1:]:
                if (time, left) in centroids and (time, right) in centroids:
                    pair_rows.append({
                        "checkpoint": checkpoint, "class_a": left, "class_b": right,
                        "centroid_distance": float(np.linalg.norm(centroids[time, left] - centroids[time, right])),
                    })
    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def representation_shifts(
    checkpoints: list[str], tensors: list[np.ndarray], labels: np.ndarray
) -> pd.DataFrame:
    rows = []
    baseline = tensors[0]
    for checkpoint, values in zip(checkpoints, tensors):
        for label in sorted(set(labels.astype(str))):
            mask = labels.astype(str) == label
            shift = values[mask].mean(axis=0) - baseline[mask].mean(axis=0)
            rows.append({"checkpoint": checkpoint, "class_label": label,
                         "shift_magnitude": float(np.linalg.norm(shift)),
                         "shift_vector": shift.astype(np.float32).tolist(),
                         "sample_count": int(mask.sum())})
    return pd.DataFrame(rows)
