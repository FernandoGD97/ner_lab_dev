"""Two-objective Pareto fronts; quality is maximized and resource cost minimized."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

OBJECTIVES = {
    "size": "checkpoint_bytes",
    "parameters": "parameters",
    "memory": "gpu_peak_allocated_mb",
    "latency": "model_time_s",
    "end_to_end_latency": "end_to_end_time_s",
    "energy": "total_energy_kwh",
}


def pareto_front(frame: pd.DataFrame, quality: str = "f1_score", cost: str = "checkpoint_bytes") -> pd.DataFrame:
    data = frame.dropna(subset=[quality, cost]).copy()
    flags = []
    for index, candidate in data.iterrows():
        dominated = False
        for other_index, other in data.iterrows():
            if index == other_index: continue
            no_worse = other[quality] >= candidate[quality] and other[cost] <= candidate[cost]
            strictly_better = other[quality] > candidate[quality] or other[cost] < candidate[cost]
            if no_worse and strictly_better:
                dominated = True
                break
        flags.append(not dominated)
    data["pareto_optimal"] = flags
    return data.sort_values(["pareto_optimal", quality, cost], ascending=[False, False, True])


def export_pareto(summary: pd.DataFrame, results_dir: str | Path) -> dict[str, Path]:
    root = Path(results_dir); root.mkdir(parents=True, exist_ok=True); paths = {}
    for name, cost in OBJECTIVES.items():
        output = root / f"pareto_f1_{name}.tsv"
        frame = pareto_front(summary, cost=cost); frame.to_csv(output, sep="\t", index=False)
        frame.to_parquet(output.with_suffix(".parquet"), compression="zstd", index=False)
        paths[name] = output
    return paths
