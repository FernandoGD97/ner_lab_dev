"""Deterministic, chunk-aware access to Part 1 snapshot shards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from lab.ner.explicability.finalize import explicability_root, read_manifest
from lab.ner.explicability.storage import iter_chunks, read_table


@dataclass(frozen=True)
class RepresentationBatch:
    checkpoint: str
    layer: int
    observation_ids: np.ndarray
    embeddings: np.ndarray


class SnapshotCollection:
    """A validated chronological view over complete rank-sharded snapshots."""

    def __init__(self, run_dir: str | Path):
        self.root = explicability_root(run_dir)
        self.manifest = read_manifest(self.root)
        self.snapshots = sorted(
            self.manifest.get("snapshots", []), key=lambda item: _snapshot_key(item)
        )
        if not self.snapshots:
            raise ValueError("No complete explicability snapshots are registered.")

    @property
    def checkpoints(self) -> list[str]:
        return [str(item["snapshot_id"]) for item in self.snapshots]

    @property
    def layers(self) -> list[int]:
        common = set(self.snapshots[0].get("layers", []))
        for snapshot in self.snapshots[1:]:
            common &= set(snapshot.get("layers", []))
        return sorted(int(layer) for layer in common)

    def iter_layer(self, checkpoint: str, layer: int) -> Iterator[RepresentationBatch]:
        snapshot = self._snapshot(checkpoint)
        snapshot_dir = self.root / "_snapshots" / checkpoint
        if not (snapshot_dir / "COMPLETE").exists():
            raise ValueError(f"Snapshot is incomplete: {checkpoint}")
        layer_index = list(snapshot["layers"]).index(layer)
        for shard in sorted(snapshot_dir.glob("rank_*")):
            for embeddings, observation_ids in iter_chunks(shard / "embeddings"):
                yield RepresentationBatch(
                    checkpoint, layer, observation_ids.astype(str),
                    embeddings[layer_index].astype(np.float32, copy=False),
                )

    def load_layer(
        self, checkpoint: str, layer: int, observation_ids: set[str] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        ids, arrays = [], []
        for batch in self.iter_layer(checkpoint, layer):
            if observation_ids is None:
                mask = np.ones(len(batch.observation_ids), dtype=bool)
            else:
                mask = np.fromiter(
                    (item in observation_ids for item in batch.observation_ids),
                    dtype=bool, count=len(batch.observation_ids),
                )
            ids.extend(batch.observation_ids[mask].tolist())
            arrays.append(batch.embeddings[mask])
        if not arrays:
            hidden = int(self._snapshot(checkpoint).get("hidden_size", 0))
            return np.asarray([], dtype=str), np.empty((0, hidden), dtype=np.float32)
        order = np.argsort(np.asarray(ids), kind="stable")
        return np.asarray(ids)[order], np.concatenate(arrays, axis=0)[order]

    def metadata(self, checkpoint: str) -> pd.DataFrame:
        snapshot_dir = self.root / "_snapshots" / checkpoint
        frames = [read_table(path) for path in sorted(snapshot_dir.glob("rank_*/observations.parquet"))]
        if not frames:
            raise ValueError(f"No observation metadata for {checkpoint}.")
        return pd.concat(frames, ignore_index=True).sort_values(
            ["observation_id"], kind="stable"
        ).reset_index(drop=True)

    def common_observation_ids(self) -> np.ndarray:
        common: set[str] | None = None
        for checkpoint in self.checkpoints:
            ids = set(self.metadata(checkpoint)["observation_id"].astype(str))
            common = ids if common is None else common & ids
        return np.asarray(sorted(common or ()), dtype=str)

    def parameter_state(self, checkpoint: str) -> dict[str, np.ndarray] | None:
        path = self.parameter_path(checkpoint)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as values:
            return {name: values[name] for name in values.files}

    def parameter_path(self, checkpoint: str) -> Path:
        return self.root / "_snapshots" / checkpoint / "parameters.npz"

    def _snapshot(self, checkpoint: str) -> dict:
        for snapshot in self.snapshots:
            if snapshot.get("snapshot_id") == checkpoint:
                return snapshot
        raise KeyError(checkpoint)


def align_representations(
    ids_a: np.ndarray, values_a: np.ndarray, ids_b: np.ndarray, values_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inner-join two representation matrices by stable ID in lexical order."""
    common, positions_a, positions_b = np.intersect1d(
        ids_a.astype(str), ids_b.astype(str), assume_unique=False, return_indices=True
    )
    return common, values_a[positions_a], values_b[positions_b]


def deterministic_sample(
    ids: np.ndarray, size: int | None, seed: int, strata: np.ndarray | None = None
) -> np.ndarray:
    """Return explicitly seeded indices, stratified proportionally when labels exist."""
    if size is None or size >= len(ids):
        return np.arange(len(ids))
    rng = np.random.default_rng(seed)
    if strata is None:
        return np.sort(rng.choice(len(ids), size=size, replace=False))
    chosen: list[int] = []
    strata = np.asarray(strata)
    groups = sorted(set(strata.tolist()), key=str)
    for group in groups:
        candidates = np.flatnonzero(strata == group)
        quota = max(1, round(size * len(candidates) / len(ids)))
        chosen.extend(rng.choice(candidates, size=min(quota, len(candidates)), replace=False))
    chosen = sorted(set(chosen))
    if len(chosen) > size:
        chosen = sorted(rng.choice(chosen, size=size, replace=False).tolist())
    elif len(chosen) < size:
        remaining = np.setdiff1d(np.arange(len(ids)), np.asarray(chosen))
        chosen.extend(rng.choice(remaining, size=size - len(chosen), replace=False).tolist())
    return np.asarray(sorted(chosen), dtype=int)


def _snapshot_key(snapshot: dict) -> tuple[int, float, int, str]:
    identifier = str(snapshot.get("snapshot_id", ""))
    if identifier == "step_000_pretrained":
        return (0, 0.0, 0, identifier)
    if identifier == "step_final":
        return (2, float("inf"), int(snapshot.get("step") or 0), identifier)
    return (1, float(snapshot.get("epoch") or 0), int(snapshot.get("step") or 0), identifier)
