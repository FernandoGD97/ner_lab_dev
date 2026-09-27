"""Manifest management, validation, recovery, and defensive cleanup."""

from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import yaml

from lab.ner.explicability.config import ExplicabilityConfig
from lab.ner.explicability.storage import validate_table
from lab.ner.explicability.utils import atomic_json, directory_size

SCHEMA_VERSION = 1
DIRECTORIES = ("_snapshots", "tables", "figures", "animations", "examples", "report")


def initialize_run(
    run_dir: str | Path, config: ExplicabilityConfig, *, model: str | None = None,
    dataset: str | None = None,
) -> Path:
    root = Path(run_dir) / "explicability"
    for name in DIRECTORIES:
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION, "run_id": str(uuid.uuid4()),
        "model": model, "dataset": dataset, "split": config.snapshots.split,
        "snapshot_trigger": config.snapshots.trigger, "snapshots": [],
        "completed_analyses": [], "mandatory_analyses": [], "required_outputs": [],
        "cleanup_policy": config.snapshots.cleanup_policy, "cleanup_state": "retained",
        "status": "collecting", "training_completed": False,
        "temporary_storage_bytes": 0, "permanent_storage_bytes": 0,
        "post_processing_seconds": None,
    }
    atomic_json(manifest, root / "manifest.json")
    return root


def read_manifest(root_or_run: str | Path) -> dict[str, Any]:
    root = explicability_root(root_or_run)
    import json
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def update_training_manifest(root_or_run: str | Path, snapshots: list[dict], succeeded: bool) -> Path:
    root = explicability_root(root_or_run)
    manifest = read_manifest(root)
    manifest["snapshots"] = [_snapshot_record(item) for item in snapshots]
    manifest["training_completed"] = bool(succeeded)
    manifest["status"] = "awaiting_postprocessing" if succeeded else "training_failed"
    manifest["temporary_storage_bytes"] = directory_size(root / "_snapshots")
    return atomic_json(manifest, root / "manifest.json")


def validate_manifest(manifest: dict[str, Any], root: str | Path | None = None) -> list[str]:
    errors = []
    required = {
        "schema_version", "run_id", "model", "dataset", "split", "snapshot_trigger",
        "snapshots", "completed_analyses", "cleanup_policy", "cleanup_state", "status",
    }
    missing = sorted(required - set(manifest))
    if missing:
        errors.append(f"Missing manifest fields: {missing}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"Unsupported schema_version: {manifest.get('schema_version')}")
    if manifest.get("cleanup_policy") not in {"on_success", "never"}:
        errors.append("Invalid cleanup_policy")
    ids = [snapshot.get("snapshot_id") for snapshot in manifest.get("snapshots", [])]
    if len(ids) != len(set(ids)):
        errors.append("Duplicate snapshot IDs")
    if root is not None and manifest.get("cleanup_state") != "removed":
        root = Path(root)
        for snapshot in manifest.get("snapshots", []):
            path = root / "_snapshots" / str(snapshot.get("snapshot_id"))
            if not snapshot.get("complete") or not (path / "COMPLETE").exists():
                errors.append(f"Incomplete snapshot: {snapshot.get('snapshot_id')}")
    return errors


def finalize_run(
    run_dir: str | Path, *, completed_analyses: Iterable[str] = (),
    required_outputs: Iterable[str] | None = None,
    mandatory_analyses: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Validate permanent outputs, publish SUCCESS, then (and only then) clean raw data."""
    started = time.perf_counter()
    root = explicability_root(run_dir)
    manifest = read_manifest(root)
    if not manifest.get("training_completed"):
        raise ValueError("Cannot finalize: training is not recorded as successfully completed.")
    if manifest.get("status") == "analyses_complete":
        raise ValueError(
            "Cannot finalize before Part 3 visualization/report generation completes. "
            "Run `lab explicability plot <run>` first."
        )
    if manifest.get("visualization_status") == "complete":
        from lab.ner.explicability.visualization.validation import validate_figure_outputs

        figure_errors = validate_figure_outputs(root)
        if figure_errors:
            raise ValueError(
                "Cannot finalize invalid visual outputs; snapshots retained: "
                + "; ".join(figure_errors)
            )
    outputs = list(required_outputs if required_outputs is not None else manifest.get("required_outputs", []))
    mandatory = list(mandatory_analyses if mandatory_analyses is not None else manifest.get("mandatory_analyses", []))
    complete = sorted(set(manifest.get("completed_analyses", [])) | set(completed_analyses))
    pending = sorted(set(mandatory) - set(complete))
    if pending:
        raise ValueError(f"Cannot finalize; mandatory analyses pending: {pending}")
    unreadable = []
    for relative in outputs:
        path = root / relative
        if not path.is_file() or (path.suffix == ".parquet" and not validate_table(path)):
            unreadable.append(relative)
    if unreadable:
        raise ValueError(f"Cannot finalize; required outputs missing or unreadable: {unreadable}")

    manifest.update({
        "completed_analyses": complete, "mandatory_analyses": mandatory,
        "required_outputs": outputs, "permanent_storage_bytes": sum(
            (root / path).stat().st_size for path in outputs
        ), "post_processing_seconds": time.perf_counter() - started, "status": "success",
    })
    errors = validate_manifest(manifest, root)
    if errors:
        raise ValueError("Cannot finalize invalid manifest: " + "; ".join(errors))
    atomic_json(manifest, root / "manifest.json")
    (root / "SUCCESS").write_text("explicability finalization complete\n", encoding="utf-8")

    if manifest["cleanup_policy"] == "on_success":
        try:
            shutil.rmtree(root / "_snapshots")
            manifest["cleanup_state"] = "removed"
            manifest["temporary_storage_bytes"] = 0
        except OSError as error:
            manifest["cleanup_state"] = "cleanup_failed"
            manifest["cleanup_error"] = str(error)
    else:
        manifest["cleanup_state"] = "retained"
    atomic_json(manifest, root / "manifest.json")
    return manifest


def explicability_root(root_or_run: str | Path) -> Path:
    path = Path(root_or_run).resolve()
    return path if path.name == "explicability" else path / "explicability"


def _snapshot_record(snapshot: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "snapshot_id", "checkpoint", "epoch", "step", "layers", "hidden_size",
        "observations", "examples", "dtype", "bytes_written", "extract_seconds",
        "write_seconds", "complete", "world_size",
        "parameters_captured",
    )
    return {key: snapshot.get(key) for key in keys}
