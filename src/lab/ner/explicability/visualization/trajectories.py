"""Selected observation and projected entity-centroid trajectories."""

from __future__ import annotations

import pandas as pd

from lab.ner.explicability.visualization.annotations import arrow
from lab.ner.explicability.visualization.layout import Axes, SvgFigure, padded_limits
from lab.ner.explicability.visualization.legends import external_legend
from lab.ner.explicability.visualization.palette import entity_colors
from lab.ner.explicability.visualization.style import StyleProfile


def selected_trajectories(
    projection: pd.DataFrame, trajectory: pd.DataFrame, style: StyleProfile,
    *, maximum: int = 20, palette="okabe_ito",
) -> tuple[SvgFigure, dict[str, str]]:
    observed = trajectory[trajectory["record_type"] == "observation"] if "record_type" in trajectory else trajectory
    score = observed.groupby("observation_id")["cumulative_path_length"].max().nlargest(maximum)
    data = projection[projection["observation_id"].isin(score.index)].copy()
    return _trajectory_plot(data, style, "observation_id", palette, label_end=False)


def centroid_trajectories(
    projection: pd.DataFrame, style: StyleProfile, *, palette="okabe_ito"
) -> tuple[SvgFigure, dict[str, str]]:
    data = projection[projection["gold_label"].notna() & (projection["gold_label"] != "O")]
    centroids = data.groupby(["checkpoint", "gold_label"], sort=False, as_index=False).agg(
        x=("x", "mean"), y=("y", "mean"), sample_count=("observation_id", "size")
    )
    return _trajectory_plot(centroids, style, "gold_label", palette, label_end=True)


def _trajectory_plot(data, style, identity, palette, label_end):
    if data.empty:
        raise ValueError("No projected trajectories are available.")
    checkpoint_order = {value: index for index, value in enumerate(dict.fromkeys(data["checkpoint"].astype(str)))}
    data = data.assign(_order=data["checkpoint"].astype(str).map(checkpoint_order))
    label_column = "gold_label" if "gold_label" in data else identity
    mapping = entity_colors(data[label_column].astype(str), palette)
    figure = SvgFigure(style)
    axes = Axes(figure, 64, 26, figure.width_px - 106, figure.height_px - 84,
                padded_limits(data["x"]), padded_limits(data["y"]))
    axes.draw("Projection 1", "Projection 2")
    for identity_value, group in data.groupby(identity, sort=True):
        group = group.sort_values("_order")
        label = str(group[label_column].iloc[-1])
        color = mapping.get(label, "#777777")
        points = [(axes.x(row.x), axes.y(row.y)) for row in group.itertuples()]
        for start, end in zip(points, points[1:]):
            arrow(figure, start, end, color)
        for index, point in enumerate(points):
            radius = style.marker_size if index == len(points) - 1 else style.marker_size / 2
            figure.add(f'<circle cx="{point[0]:.2f}" cy="{point[1]:.2f}" r="{radius}" fill="{color}" stroke="#222" stroke-width=".5"/>')
        if label_end:
            figure.text(points[-1][0] + 5, points[-1][1] - 4, label, size=style.font_size - 1, weight="bold")
    if label_end:
        external_legend(figure, mapping, x=figure.width_px - 155, y=30, columns=1)
    return figure, mapping
