"""Streaming extraction of complete Transformer representation snapshots."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lab.ner.encoding.rows import IGNORE_INDEX
from lab.ner.evaluation.spans import ensure_int_list
from lab.ner.explicability.config import SnapshotConfig
from lab.ner.explicability.metadata import token_metadata
from lab.ner.explicability.storage import ChunkedArrayWriter, write_table
from lab.ner.explicability.utils import (
    atomic_json, available_bytes, directory_size, distributed_barrier, distributed_context,
)

LOGGER = logging.getLogger(__name__)


def estimate_snapshot_bytes(
    observations: int, hidden_size: int, layers: int, dtype: str,
    metadata_overhead_fraction: float = 0.10,
) -> int:
    raw = int(observations) * int(hidden_size) * int(layers) * np.dtype(dtype).itemsize
    return int(raw * (1.0 + metadata_overhead_fraction))


def check_disk_space(root: str | Path, expected_bytes: int, safety_margin_bytes: int) -> dict[str, int]:
    root = Path(root)
    current = directory_size(root)
    available = available_bytes(root)
    projected = current + expected_bytes
    stats = {
        "expected_new_bytes": expected_bytes, "current_temporary_bytes": current,
        "projected_temporary_bytes": projected, "available_bytes": available,
        "safety_margin_bytes": safety_margin_bytes,
    }
    LOGGER.info("Explicability snapshot disk projection: %s", stats)
    if expected_bytes + safety_margin_bytes > available:
        raise OSError(
            "Insufficient disk for explicability snapshot before writing: "
            f"expected={expected_bytes}, available={available}, safety_margin={safety_margin_bytes}."
        )
    return stats


def capture_snapshot(
    *, model, tokenizer, rows: pd.DataFrame, output_root: str | Path,
    snapshot_id: str, config: SnapshotConfig, checkpoint: str | None = None,
    epoch: float | None = None, step: int = 0, model_name: str | None = None,
    capture_parameters: bool = False,
) -> dict[str, Any]:
    """Extract one rank-local shard and atomically publish a complete snapshot."""
    import torch
    from transformers import DataCollatorForTokenClassification

    rank, world_size = distributed_context()
    root = Path(output_root)
    snapshots = root / "_snapshots"
    staging = snapshots / f".{snapshot_id}.incomplete"
    final = snapshots / snapshot_id
    snapshots.mkdir(parents=True, exist_ok=True)
    if final.exists():
        return json.loads((final / "snapshot.json").read_text(encoding="utf-8"))
    staging.mkdir(exist_ok=True)
    if rank == 0:
        (staging / "INCOMPLETE").write_text("", encoding="utf-8")

    hidden_size = int(getattr(model.config, "hidden_size"))
    available_layers = int(getattr(model.config, "num_hidden_layers", 0)) + 1
    layers = resolve_layers(config, available_layers)
    observation_estimate = sum(
        len(ensure_int_list(row.input_ids)) if config.include_all_tokens
        else sum(label != IGNORE_INDEX for label in ensure_int_list(row.labels))
        for row in rows.itertuples(index=False)
    )
    expected = estimate_snapshot_bytes(
        observation_estimate, hidden_size, len(layers), config.dtype,
        config.metadata_overhead_fraction,
    )
    if capture_parameters:
        expected += sum(parameter.numel() * 4 for parameter in model.parameters())
    disk = check_disk_space(snapshots, expected, config.disk_safety_margin_bytes)

    shard = staging / f"rank_{rank:05d}"
    writer = ChunkedArrayWriter(shard / "embeddings", config.dtype, hidden_size, layers)
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer, label_pad_token_id=IGNORE_INDEX)
    batch_size = config.batch_size or 8
    selected_indices = list(range(rank, len(rows), world_size))
    metadata_records: list[dict[str, Any]] = []
    extract_seconds = write_seconds = 0.0
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    id2label = {int(key): str(value) for key, value in getattr(model.config, "id2label", {}).items()}

    try:
        with torch.inference_mode():
            for offset in range(0, len(selected_indices), batch_size):
                indices = selected_indices[offset:offset + batch_size]
                row_tuples = [next(rows.iloc[[i]].itertuples(index=False)) for i in indices]
                features = []
                for row in row_tuples:
                    feature = {
                        "input_ids": ensure_int_list(row.input_ids),
                        "attention_mask": ensure_int_list(row.attention_mask),
                        "labels": ensure_int_list(row.labels),
                    }
                    if hasattr(row, "token_type_ids") and row.token_type_ids is not None:
                        feature["token_type_ids"] = ensure_int_list(row.token_type_ids)
                    features.append(feature)
                batch = collator(features)
                inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
                started = time.perf_counter()
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                extract_seconds += time.perf_counter() - started
                hidden_states = outputs.hidden_states
                if hidden_states is None:
                    raise RuntimeError("Model did not return hidden states for explicability.")
                logits = outputs.logits.detach().float().cpu().numpy()

                for batch_index, (row_index, row) in enumerate(zip(indices, row_tuples)):
                    length = len(ensure_int_list(row.input_ids))
                    labels = ensure_int_list(row.labels)
                    positions = list(range(length)) if config.include_all_tokens else [
                        i for i, label in enumerate(labels) if label != IGNORE_INDEX
                    ]
                    records = token_metadata(
                        row, row_index, tokenizer, logits[batch_index, :length], positions, id2label
                    )
                    array = torch.stack([
                        hidden_states[layer][batch_index, positions, :] for layer in layers
                    ]).detach().cpu().numpy()
                    started = time.perf_counter()
                    writer.write(array, [record["observation_id"] for record in records])
                    write_seconds += time.perf_counter() - started
                    metadata_records.extend(records)
                del outputs, hidden_states, logits, inputs, batch

        writer.finalize()
        write_table(
            pd.DataFrame(metadata_records), shard / "observations.parquet",
            sort_by=["row_index", "token_position"],
        )
        shard_summary = {
            "rank": rank, "world_size": world_size, "observations": writer.count,
            "examples": len(selected_indices), "extract_seconds": extract_seconds,
            "write_seconds": write_seconds, "bytes_written": directory_size(shard),
        }
        atomic_json(shard_summary, shard / "shard.json")
        (shard / "COMPLETE").write_text("", encoding="utf-8")
    finally:
        if was_training:
            model.train()

    if rank == 0 and capture_parameters:
        parameter_path = staging / ".parameters.npz.tmp"
        state = {
            name: parameter.detach().cpu().float().numpy()
            for name, parameter in model.named_parameters()
        }
        with parameter_path.open("wb") as stream:
            np.savez_compressed(stream, **state)
        os.replace(parameter_path, staging / "parameters.npz")
        del state

    distributed_barrier()
    if rank == 0:
        missing = [i for i in range(world_size) if not (staging / f"rank_{i:05d}" / "COMPLETE").exists()]
        if missing:
            raise RuntimeError(f"Snapshot {snapshot_id} has incomplete rank shards: {missing}")
        shards = [json.loads((staging / f"rank_{i:05d}" / "shard.json").read_text()) for i in range(world_size)]
        summary = {
            "schema_version": 1, "snapshot_id": snapshot_id, "checkpoint": checkpoint,
            "epoch": epoch, "step": int(step), "split": config.split,
            "embedding_scope": config.embedding_scope, "layers": layers,
            "hidden_size": hidden_size, "dtype": config.dtype, "model": model_name,
            "examples": sum(item["examples"] for item in shards),
            "observations": sum(item["observations"] for item in shards),
            "world_size": world_size, "shards": shards, "disk_projection": disk,
            "bytes_written": directory_size(staging),
            "extract_seconds": sum(item["extract_seconds"] for item in shards),
            "write_seconds": sum(item["write_seconds"] for item in shards),
            "parameters_captured": capture_parameters,
            "complete": True,
        }
        atomic_json(summary, staging / "snapshot.json")
        (staging / "INCOMPLETE").unlink(missing_ok=True)
        (staging / "COMPLETE").write_text("", encoding="utf-8")
        os.replace(staging, final)
    distributed_barrier()
    return json.loads((final / "snapshot.json").read_text(encoding="utf-8"))


def resolve_layers(config: SnapshotConfig, available_layers: int) -> list[int]:
    if config.embedding_scope == "last_hidden_state":
        return [available_layers - 1]
    requested = list(range(available_layers)) if config.embedding_scope == "all_hidden_states" else list(config.selected_layers)
    resolved = [layer if layer >= 0 else available_layers + layer for layer in requested]
    if any(layer < 0 or layer >= available_layers for layer in resolved):
        raise ValueError(f"Selected layer outside [0, {available_layers - 1}]: {requested}")
    if len(set(resolved)) != len(resolved):
        raise ValueError("Selected layers resolve to duplicates.")
    return resolved
