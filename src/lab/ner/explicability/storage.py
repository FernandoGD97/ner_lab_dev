"""Chunked temporary arrays and Parquet-first permanent table helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from lab.ner.explicability.utils import atomic_json


class ChunkedArrayWriter:
    """Incrementally write independently recoverable compressed NumPy chunks."""

    def __init__(self, path: str | Path, dtype: str, hidden_size: int, layers: list[int]):
        self.path = Path(path)
        self.incomplete = self.path.with_name(f".{self.path.name}.incomplete")
        if self.path.exists() or self.incomplete.exists():
            raise FileExistsError(f"Snapshot shard already exists: {self.path}")
        self.incomplete.mkdir(parents=True)
        (self.incomplete / "INCOMPLETE").write_text("", encoding="utf-8")
        self.dtype = np.dtype(dtype)
        self.hidden_size = int(hidden_size)
        self.layers = list(layers)
        self.chunks: list[dict] = []
        self.count = 0

    def write(self, embeddings: np.ndarray, observation_ids: list[str]) -> None:
        array = np.asarray(embeddings, dtype=self.dtype)
        if array.ndim != 3 or array.shape[0] != len(self.layers) or array.shape[2] != self.hidden_size:
            raise ValueError("Embedding chunk must have shape (layers, observations, hidden_size).")
        if array.shape[1] != len(observation_ids):
            raise ValueError("Embedding observations and IDs differ in length.")
        index = len(self.chunks)
        filename = f"chunk_{index:06d}.npz"
        temporary = self.incomplete / f".{filename}.tmp"
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, embeddings=array, observation_ids=np.asarray(observation_ids))
        target = self.incomplete / filename
        os.replace(temporary, target)
        self.chunks.append({"file": filename, "observations": len(observation_ids), "bytes": target.stat().st_size})
        self.count += len(observation_ids)

    def finalize(self) -> Path:
        atomic_json({
            "format": "chunked-npz-v1", "dtype": self.dtype.name,
            "hidden_size": self.hidden_size, "layers": self.layers,
            "observations": self.count, "chunks": self.chunks,
        }, self.incomplete / "index.json")
        (self.incomplete / "INCOMPLETE").unlink()
        (self.incomplete / "COMPLETE").write_text("", encoding="utf-8")
        os.replace(self.incomplete, self.path)
        return self.path

    def abort(self) -> None:
        """Leave the incomplete directory in place for diagnosis/recovery."""


def iter_chunks(path: str | Path) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    path = Path(path)
    if not (path / "COMPLETE").exists():
        raise ValueError(f"Incomplete chunked array: {path}")
    index = json.loads((path / "index.json").read_text(encoding="utf-8"))
    for chunk in index["chunks"]:
        with np.load(path / chunk["file"], allow_pickle=False) as values:
            yield values["embeddings"], values["observation_ids"]


def write_table(
    frame: pd.DataFrame, path: str | Path, *, sort_by: list[str] | None = None
) -> Path:
    """Write a deterministic ZSTD Parquet table atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.sort_values(sort_by, kind="stable") if sort_by else frame
    temporary = path.with_name(f".{path.name}.tmp")
    output.to_parquet(temporary, index=False, engine="pyarrow", compression="zstd")
    os.replace(temporary, path)
    return path


def read_table(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")


def validate_table(path: str | Path) -> bool:
    try:
        import pyarrow.parquet as pq
        pq.ParquetFile(path).schema
        return True
    except (OSError, ValueError):
        return False
