"""ECDF and temporal metric plots with honest common scales."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lab.ner.explicability.visualization.annotations import line_path
from lab.ner.explicability.visualization.layout import Axes, SvgFigure, padded_limits
from lab.ner.explicability.visualization.palette import entity_colors
from lab.ner.explicability.visualization.style import StyleProfile


def temporal_lines(
    frame: pd.DataFrame, checkpoint: str, metric: str, style: StyleProfile,
    *, group: str | None = None, ylabel: str | None = None,
) -> SvgFigure:
    data = frame[[checkpoint, metric] + ([group] if group else [])].dropna()
    if data.empty:
        raise ValueError(f"No finite values for {metric}.")
    checkpoints = list(dict.fromkeys(data[checkpoint].astype(str)))
    order = {label: index for index, label in enumerate(checkpoints)}
    colors = entity_colors(data[group].astype(str) if group else [metric])
    figure = SvgFigure(style)
    axes = Axes(figure, 64, 24, figure.width_px - 98, figure.height_px - 82,
                (-.1, len(checkpoints) - .9), padded_limits(data[metric]))
    axes.draw("Checkpoint", ylabel or metric.replace("_", " ").title())
    groups = data.groupby(group, sort=True) if group else [(metric, data)]
    for label, subset in groups:
        summary = subset.assign(_order=subset[checkpoint].astype(str).map(order)).groupby("_order")[metric].agg(["mean", "std"]).reset_index()
        line_path(axes, summary["_order"], summary["mean"], colors[str(label)])
        for x, y in zip(summary["_order"], summary["mean"]):
            figure.add(f'<circle cx="{axes.x(x):.2f}" cy="{axes.y(y):.2f}" r="{style.marker_size}" fill="{colors[str(label)]}"/>')
    for label, index in order.items():
        figure.text(axes.x(index), axes.top + axes.height + 31, label, anchor="middle", size=style.font_size - 2)
    return figure


def ecdf_figure(values, style: StyleProfile, xlabel: str) -> SvgFigure:
    values = np.sort(np.asarray(values, float))
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("ECDF cannot plot empty/all-NaN values.")
    figure = SvgFigure(style)
    axes = Axes(figure, 64, 24, figure.width_px - 94, figure.height_px - 76,
                padded_limits(values), (0, 1))
    axes.draw(xlabel, "Cumulative proportion")
    line_path(axes, values, np.arange(1, len(values) + 1) / len(values), "#0072B2")
    figure.text(axes.left + axes.width - 2, axes.top + 14, f"n = {len(values)}", anchor="end")
    return figure
