"""Probe decodability heatmaps and checkpoint trajectories."""

from __future__ import annotations

import pandas as pd

from lab.ner.explicability.visualization.heatmaps import heatmap_figure
from lab.ner.explicability.visualization.style import StyleProfile


def probing_heatmap(frame: pd.DataFrame, style: StyleProfile, task: str) -> object:
    data = frame[frame["probe_task"] == task]
    return heatmap_figure(data, "layer", "checkpoint", "f1", style,
                          xlabel="Checkpoint", ylabel="Encoder layer", colormap="viridis")
