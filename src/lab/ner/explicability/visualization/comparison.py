"""Cross-run/seed comparison using permanent fingerprint Parquet files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lab.ner.explicability.finalize import explicability_root, read_manifest
from lab.ner.explicability.storage import write_table
from lab.ner.explicability.visualization.heatmaps import heatmap_figure
from lab.ner.explicability.visualization.export import export_figure
from lab.ner.explicability.visualization.style import profile


def compare_runs(run_directories, output_directory, visualization_config) -> tuple[Path, Path]:
    """Persist and plot run×metric fingerprints without loading snapshots."""
    frames = []
    for run_directory in run_directories:
        root = explicability_root(run_directory)
        path = root / "tables" / "corpus_adaptation_fingerprint.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing fingerprint table: {path}")
        frame = pd.read_parquet(path)
        manifest = read_manifest(root)
        frame["run_id"] = manifest.get("run_id")
        frame["model"] = manifest.get("model")
        frame["seed"] = manifest.get("analysis_parameters", {}).get("sample_seed")
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    output = Path(output_directory)
    table = write_table(combined, output / "cross_run_fingerprints.parquet",
                        sort_by=["run_id", "metric"])
    figure = heatmap_figure(combined, "run_id", "metric", "value",
                            profile(visualization_config.profile, "wide"),
                            xlabel="Adaptation diagnostic", ylabel="Run")
    svg = output / "cross_run_adaptation_fingerprint.svg"
    export_figure(figure, svg, visualization_config)
    return table, svg
