"""Artifact readiness and Phase 1 matrix definitions."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoConfig, AutoTokenizer

from .schemas import ModelReadiness

MATRIX = {
    "A0": "original FP32", "A1": "FP16", "A2": "BF16", "A3": "INT8 dynamic PTQ",
    "A4": "INT8 static/calibrated PTQ", "A5": "INT4", "A6": "vocabulary-pruned",
    "A7": "vocabulary-pruned INT8", "B1": "9-layer student", "B2": "6-layer student",
    "B3": "4-layer student", "B4": "hidden size 512", "B5": "hidden size 384",
    "B6": "reduced FFN", "B7": "structured pruning", "B8": "distilled student",
    "C1": "distilled + FP16", "C2": "distilled + INT8",
    "C3": "structured pruning + INT8", "C4": "vocabulary pruning + INT8",
    "C5": "distilled + vocabulary pruning", "C6": "distillation + vocabulary pruning + INT8",
}
SUPPORTED_BACKENDS = {"transformers", "pytorch_dynamic_int8", "pytorch_reference_static_int8",
                      "pytorch_int4_reference", "pytorch_low_rank"}


@dataclass(frozen=True)
class ModelStatus:
    id: str
    path: str
    status: ModelReadiness
    reason: str | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "status": self.status.value,
            "reason": self.reason,
            "metadata": self.metadata,
        }


def artifact_metadata(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    manifest = root / "compression_manifest.json"
    recipe = root / "compression_recipe.yaml"
    metadata: dict[str, Any] = {}
    if manifest.exists():
        metadata.update(json.loads(manifest.read_text(encoding="utf-8")))
    metadata["manifest_path"] = str(manifest) if manifest.exists() else None
    metadata["recipe_path"] = str(recipe) if recipe.exists() else None
    return metadata


def validate_model(model) -> ModelStatus:
    path = Path(model.path)
    if model.id not in MATRIX:
        return ModelStatus(model.id, str(path), ModelReadiness.INVALID, "Unknown matrix id.", {})
    if not path.exists():
        return ModelStatus(model.id, str(path), ModelReadiness.MISSING, "Artifact does not exist.", {})
    metadata = artifact_metadata(path)
    backend = model.backend or metadata.get("backend", "transformers")
    if backend not in SUPPORTED_BACKENDS:
        return ModelStatus(
            model.id, str(path), ModelReadiness.UNSUPPORTED,
            f"Backend {backend!r} is not supported by the canonical PyTorch inference pipeline.",
            metadata,
        )
    if not (path / "encoding.json").exists():
        return ModelStatus(
            model.id, str(path), ModelReadiness.INVALID,
            "encoding.json is required by canonical NER inference.", metadata,
        )
    try:
        AutoConfig.from_pretrained(path, local_files_only=True)
        AutoTokenizer.from_pretrained(path, local_files_only=True)
    except Exception as error:
        return ModelStatus(model.id, str(path), ModelReadiness.INVALID, str(error), metadata)
    return ModelStatus(model.id, str(path), ModelReadiness.READY, None, metadata)


def validate_matrix(config) -> list[ModelStatus]:
    return [validate_model(model) for model in config.models]
