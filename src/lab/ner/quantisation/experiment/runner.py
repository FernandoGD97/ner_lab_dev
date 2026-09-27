"""Orchestration, subprocess isolation, resume, and result generation."""
from __future__ import annotations

import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lab.core.provenance import write_manifest

from .config import ExperimentConfig, load_experiment, write_resolved
from .environment import capture_environment
from .matrix import ModelReadiness, validate_matrix
from .results import collect_runs, is_successful_run, write_comparison, write_long, write_summary
from .schemas import RunStatus


def execution_order(config: ExperimentConfig) -> list[dict[str, Any]]:
    """Deterministic rotated random order that balances first-position bias."""
    identifiers = [model.id for model in config.models]
    random.Random(config.experiment.seed).shuffle(identifiers)
    order = []
    for repetition in range(1, config.inference.repetitions + 1):
        shift = (repetition - 1) % len(identifiers)
        rotated = identifiers[shift:] + identifiers[:shift]
        for position, model_id in enumerate(rotated, start=1):
            order.append({"repetition": repetition, "position": position, "model_id": model_id})
    return order


def _payload(config: ExperimentConfig, model, repetition: int, results: Path) -> dict[str, Any]:
    model_root = results / "models" / model.id
    run_dir = model_root / "runs" / f"run_{repetition:03d}"
    batch_size = model.batch_size or config.inference.batch_size
    return {
        "identity": {
            "experiment_id": config.experiment.id,
            "run_id": f"{model.id}-run-{repetition:03d}",
            "model_id": model.id,
        },
        "model_id": model.id, "model_path": str(model.path),
        "dataset": str(config.dataset.path), "split": config.dataset.split,
        "reference": str(config.dataset.reference) if config.dataset.reference else None,
        "batch_size": batch_size, "max_length": config.inference.max_length,
        "device": config.inference.device, "warmup_batches": config.inference.warmup_batches,
        "pad_to_multiple_of": config.inference.pad_to_multiple_of,
        "runtime": model.runtime, "backend": model.backend,
        "execution_provider": model.execution_provider,
        "representation": model.representation, "protocol": config.protocol,
        "energy_enabled": config.energy.enabled, "energy_modes": config.energy.modes,
        "min_duration_seconds": config.energy.min_duration_seconds,
        "measure_power_secs": config.energy.measure_power_secs,
        "run_dir": str(run_dir), "model_output_dir": str(model_root),
        "predictions_dir": str(model_root / "predictions"),
    }


def materialize_results(results_dir: str | Path) -> dict[str, Path]:
    root = Path(results_dir)
    rows = collect_runs(root)
    return {
        "long": write_long(rows, root / "benchmark_long.tsv"),
        "summary": write_summary(rows, root / "benchmark_summary.tsv"),
        "comparison": write_comparison(rows, root / "comparison.tsv"),
    }


def validate_experiment(config_or_path: ExperimentConfig | str | Path) -> dict[str, Any]:
    config = config_or_path if isinstance(config_or_path, ExperimentConfig) else load_experiment(config_or_path)
    statuses = validate_matrix(config)
    dataset_status = "READY" if config.dataset.path.exists() else "MISSING"
    return {
        "experiment_id": config.experiment.id,
        "dataset": {"path": str(config.dataset.path), "status": dataset_status},
        "models": [status.to_dict() for status in statuses],
        "ready": dataset_status == "READY" and all(
            status.status == ModelReadiness.READY for status in statuses
        ),
    }


def run_experiment(
    config_path: str | Path,
    *,
    dry_run: bool = False,
    resume: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    if resume and force:
        raise ValueError("--resume and --force are mutually exclusive.")
    config = load_experiment(config_path)
    results = config.output.directory / config.experiment.id
    if results.exists() and force:
        shutil.rmtree(results)
    results.mkdir(parents=True, exist_ok=True)

    validation = validate_experiment(config)
    write_resolved(config, results / "resolved_experiment.yaml")
    write_manifest(
        capture_environment(
            runtime=config.models[0].runtime,
            execution_provider=config.models[0].execution_provider,
        ),
        results / "environment.json",
    )
    order = execution_order(config)
    write_manifest(order, results / "execution_order.json")
    study = {
        "experiment_id": config.experiment.id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": config.protocol,
        "validation": validation,
        "dry_run": dry_run,
    }
    write_manifest(study, results / "study_manifest.json")
    if dry_run:
        return study

    models = {model.id: model for model in config.models}
    statuses = {status.id: status for status in validate_matrix(config)}
    for entry in order:
        model = models[entry["model_id"]]
        repetition = entry["repetition"]
        run_dir = results / "models" / model.id / "runs" / f"run_{repetition:03d}"
        run_file = run_dir / "run.json"
        status = statuses[model.id]
        run_dir.mkdir(parents=True, exist_ok=True)
        if validation["dataset"]["status"] != "READY" or status.status != ModelReadiness.READY:
            write_manifest({
                "experiment_id": config.experiment.id,
                "run_id": f"{model.id}-run-{repetition:03d}",
                "model_id": model.id,
                "status": RunStatus.UNSUPPORTED.value if status.status == ModelReadiness.UNSUPPORTED else RunStatus.SKIPPED.value,
                "readiness": status.to_dict(),
                "dataset_status": validation["dataset"]["status"],
            }, run_file)
            continue
        if resume and is_successful_run(run_file):
            continue
        if run_file.exists() and not force and not resume:
            raise FileExistsError(f"Run already exists: {run_file}. Use --resume or --force.")
        payload = _payload(config, model, repetition, results)
        payload_file = run_dir / "payload.json"
        write_manifest(payload, payload_file)
        completed = subprocess.run(
            [sys.executable, "-m", "lab.ner.quantisation.experiment.worker", str(payload_file)],
            text=True, capture_output=True, check=False,
        )
        (run_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (run_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        if not run_file.exists():
            write_manifest({
                **payload["identity"], "status": RunStatus.FAILED.value,
                "traceback": completed.stderr, "returncode": completed.returncode,
            }, run_file)

    paths = materialize_results(results)
    rows = collect_runs(results)
    study.update({
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "run_status_counts": {
            status.value: sum(row.get("status") == status.value for row in rows)
            for status in RunStatus
        },
        "outputs": {name: str(path) for name, path in paths.items()},
    })
    write_manifest(study, results / "study_manifest.json")
    return study


def experiment_status(results_dir: str | Path) -> dict[str, Any]:
    rows = collect_runs(results_dir)
    return {
        "results_dir": str(results_dir),
        "runs": len(rows),
        "statuses": {
            status.value: sum(row.get("status") == status.value for row in rows)
            for status in RunStatus
        },
    }
