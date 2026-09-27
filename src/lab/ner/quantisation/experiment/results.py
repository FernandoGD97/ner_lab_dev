"""Long-form, repeated-run summaries, and baseline comparisons."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

LONG_COLUMNS = [
    "experiment_id", "run_id", "model_id", "artifact_id", "source_model", "method_chain",
    "precision", "model_representation", "runtime", "backend", "execution_provider", "protocol",
    "task", "dataset", "split", "batch_size", "max_length", "n_documents", "n_sentences",
    "n_tokens", "n_padded_tokens", "parameters", "nonzero_parameters", "checkpoint_bytes",
    "vocab_size", "hidden_size", "intermediate_size", "layers", "attention_heads",
    "precision_score", "recall_score", "f1_score", "load_time_s", "preprocessing_time_s",
    "tokenization_time_s", "h2d_time_s", "model_time_s", "pipeline_inference_time_s", "postprocessing_time_s",
    "end_to_end_time_s", "cold_start_time_s", "documents_per_s", "sentences_per_s",
    "tokens_per_s", "cpu_rss_mb", "gpu_allocated_mb", "gpu_reserved_mb",
    "gpu_peak_allocated_mb", "gpu_peak_reserved_mb", "cpu_energy_kwh", "gpu_energy_kwh",
    "ram_energy_kwh", "total_energy_kwh", "co2_kg", "energy_mode", "joules_per_1000_documents",
    "joules_per_1000_tokens", "wh_per_1000_documents", "status",
]
SUMMARY_METRICS = [
    "parameters", "checkpoint_bytes", "gpu_peak_allocated_mb", "gpu_peak_reserved_mb",
    "precision_score", "recall_score", "f1_score", "model_time_s", "end_to_end_time_s",
    "documents_per_s", "tokens_per_s", "total_energy_kwh", "co2_kg",
]


def write_long(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    frame = pd.DataFrame([{column: row.get(column) for column in LONG_COLUMNS} for row in rows])
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, sep="\t", index=False)
    return destination


def collect_runs(results_dir: str | Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(Path(results_dir).glob("models/*/runs/run_*/run.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("status") in {"SUCCESS", "OOM", "FAILED", "UNSUPPORTED", "SKIPPED"}:
            rows.append(row)
    return rows


def is_successful_run(run_file: str | Path) -> bool:
    """Return true only for a readable, completed source-of-truth run record."""
    path = Path(run_file)
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "SUCCESS"
    except (OSError, json.JSONDecodeError):
        return False


def aggregate(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    output: list[dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame(columns=["model_id", "metric", "count", "mean", "median", "std", "p50", "p95", "ci95_low", "ci95_high"])
    successful = frame[frame["status"] == "SUCCESS"]
    for model_id, group in successful.groupby("model_id", sort=True):
        for metric in SUMMARY_METRICS:
            if metric not in group:
                continue
            values = [float(value) for value in group[metric].dropna().tolist()]
            if not values:
                continue
            mean = statistics.fmean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            margin = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
            output.append({
                "model_id": model_id, "metric": metric, "count": len(values), "mean": mean,
                "median": statistics.median(values), "std": std,
                "p50": float(pd.Series(values).quantile(0.50)),
                "p95": float(pd.Series(values).quantile(0.95)),
                "ci95_low": mean - margin, "ci95_high": mean + margin,
            })
    return pd.DataFrame(output)


def write_summary(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    destination = Path(path)
    aggregate(rows).to_csv(destination, sep="\t", index=False)
    return destination


def reduction(baseline: float | None, candidate: float | None) -> float | None:
    return None if baseline in (None, 0) or candidate is None else (baseline - candidate) / baseline * 100


def speedup(baseline_time: float | None, candidate_time: float | None) -> float | None:
    return None if baseline_time is None or candidate_time in (None, 0) else baseline_time / candidate_time


def difference(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or pd.isna(candidate) or pd.isna(baseline):
        return None
    return candidate - baseline


def comparisons(rows: Iterable[dict[str, Any]], baseline_id: str = "A0") -> pd.DataFrame:
    frame = pd.DataFrame([row for row in rows if row.get("status") == "SUCCESS"])
    columns = ["model_id", "baseline_id", "parameter_reduction_pct", "checkpoint_reduction_pct",
        "peak_vram_reduction_pct", "model_forward_speedup", "end_to_end_speedup",
        "energy_reduction_pct", "precision_delta", "recall_delta", "f1_delta"]
    if frame.empty or baseline_id not in set(frame["model_id"]):
        return pd.DataFrame(columns=columns)
    numeric = frame.select_dtypes(include="number").columns
    means = frame.groupby("model_id")[numeric].mean(numeric_only=True)
    baseline = means.loc[baseline_id]
    output = []
    for model_id, candidate in means.iterrows():
        output.append({
            "model_id": model_id, "baseline_id": baseline_id,
            "parameter_reduction_pct": reduction(baseline.get("parameters"), candidate.get("parameters")),
            "checkpoint_reduction_pct": reduction(baseline.get("checkpoint_bytes"), candidate.get("checkpoint_bytes")),
            "peak_vram_reduction_pct": reduction(baseline.get("gpu_peak_allocated_mb"), candidate.get("gpu_peak_allocated_mb")),
            "model_forward_speedup": speedup(baseline.get("model_time_s"), candidate.get("model_time_s")),
            "end_to_end_speedup": speedup(baseline.get("end_to_end_time_s"), candidate.get("end_to_end_time_s")),
            "energy_reduction_pct": reduction(baseline.get("total_energy_kwh"), candidate.get("total_energy_kwh")),
            "precision_delta": difference(candidate.get("precision_score"), baseline.get("precision_score")),
            "recall_delta": difference(candidate.get("recall_score"), baseline.get("recall_score")),
            "f1_delta": difference(candidate.get("f1_score"), baseline.get("f1_score")),
        })
    return pd.DataFrame(output, columns=columns)


def write_comparison(rows: Iterable[dict[str, Any]], path: str | Path, baseline_id: str = "A0") -> Path:
    destination = Path(path)
    comparisons(rows, baseline_id).to_csv(destination, sep="\t", index=False)
    return destination
