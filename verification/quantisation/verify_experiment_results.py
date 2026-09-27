"""CPU-only checks for repeated-run aggregation and A0 comparisons."""
import tempfile
from pathlib import Path

from lab.ner.quantisation.experiment.results import (
    LONG_COLUMNS,
    aggregate,
    comparisons,
    is_successful_run,
    speedup,
    write_comparison,
    write_long,
    write_summary,
)

rows = []
for model, parameter, checkpoint, vram, model_time, end_time, energy, f1 in (
    ("A0", 100, 1000, 100, 2.0, 4.0, 0.010, 0.90),
    ("A1", 100, 500, 80, 1.0, 3.0, 0.008, 0.89),
):
    for repetition in (1, 2):
        rows.append({
            "experiment_id": "test", "run_id": f"{model}-{repetition}", "model_id": model,
            "status": "SUCCESS", "parameters": parameter, "checkpoint_bytes": checkpoint,
            "gpu_peak_allocated_mb": vram, "model_time_s": model_time,
            "end_to_end_time_s": end_time, "total_energy_kwh": energy,
            "precision_score": f1, "recall_score": f1, "f1_score": f1,
        })

summary = aggregate(rows)
assert {"mean", "median", "std", "p50", "p95", "ci95_low", "ci95_high"} <= set(summary)
comparison = comparisons(rows).set_index("model_id")
assert comparison.loc["A1", "parameter_reduction_pct"] == 0
assert comparison.loc["A1", "checkpoint_reduction_pct"] == 50
assert comparison.loc["A1", "peak_vram_reduction_pct"] == 20
assert comparison.loc["A1", "model_forward_speedup"] == 2
assert comparison.loc["A1", "end_to_end_speedup"] == 4 / 3
assert round(comparison.loc["A1", "energy_reduction_pct"], 6) == 20
assert round(comparison.loc["A1", "f1_delta"], 6) == -0.01
assert speedup(1, 0) is None

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    write_long(rows, root / "benchmark_long.tsv")
    write_summary(rows, root / "benchmark_summary.tsv")
    write_comparison(rows, root / "comparison.tsv")
    assert all((root / name).exists() for name in ("benchmark_long.tsv", "benchmark_summary.tsv", "comparison.tsv"))
    assert (root / "benchmark_long.tsv").read_text().splitlines()[0].split("\t") == LONG_COLUMNS
    run = root / "run.json"
    assert not is_successful_run(run)
    run.write_text('{"status":"FAILED"}')
    assert not is_successful_run(run)
    run.write_text('{"status":"SUCCESS"}')
    assert is_successful_run(run)
