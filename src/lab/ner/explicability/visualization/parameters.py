"""Architecture-component parameter adaptation heatmaps."""

from __future__ import annotations

import pandas as pd

from lab.ner.explicability.visualization.heatmaps import heatmap_figure
from lab.ner.explicability.visualization.style import StyleProfile


def parameter_heatmap(frame: pd.DataFrame, style: StyleProfile, metric="relative_delta"):
    data = frame.copy()
    data["architecture_component"] = data["layer"].astype(str) + " / " + data["component"].astype(str)
    return heatmap_figure(data, "architecture_component", "checkpoint", metric, style,
                          xlabel="Checkpoint", ylabel="Architecture component", colormap="cividis")
