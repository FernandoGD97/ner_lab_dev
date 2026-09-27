"""Temporal projected-representation figures from projection.parquet only."""

from __future__ import annotations

import hashlib

import pandas as pd

from lab.ner.explicability.visualization.layout import Axes, SvgFigure, padded_limits, panel_grid
from lab.ner.explicability.visualization.legends import external_legend
from lab.ner.explicability.visualization.palette import entity_colors, interpolate_color
from lab.ner.explicability.visualization.style import StyleProfile


def temporal_small_multiples(
    frame: pd.DataFrame, style: StyleProfile, *, palette="okabe_ito", entity_only=False,
    background_limit=2000, seed=42, color_by="gold_label",
) -> tuple[SvgFigure, dict[str, str], dict[str, object]]:
    required = {"observation_id", "checkpoint", "x", "y", color_by}
    if not required <= set(frame):
        raise ValueError(f"Projection table lacks: {sorted(required - set(frame))}")
    data = frame.dropna(subset=["x", "y"]).copy()
    if entity_only:
        data = data[data["gold_label"] != "O"]
        if background_limit and data["observation_id"].nunique() > background_limit:
            data = _stratified_visual_sample(data, background_limit, seed, color_by)
    elif background_limit and "gold_label" in data:
        entity = data[data["gold_label"] != "O"]
        outside = data[data["gold_label"] == "O"]
        outside = _visual_sample(outside, background_limit, seed)
        data = pd.concat([outside, entity], ignore_index=True)
        if data["observation_id"].nunique() > background_limit * 2:
            data = _stratified_visual_sample(data, background_limit * 2, seed, color_by)
    checkpoints = list(dict.fromkeys(data["checkpoint"].astype(str)))
    if not checkpoints or data.empty:
        raise ValueError("Projection table has no plottable observations.")
    figure = SvgFigure(style)
    boxes = panel_grid(figure, len(checkpoints), columns=min(5, len(checkpoints)), margins=(45, 45, 20, 46), gap=18)
    xlim, ylim = padded_limits(data["x"]), padded_limits(data["y"])
    mapping = entity_colors(data[color_by].dropna().astype(str), palette)
    dense = len(data) > 20_000
    for checkpoint, box in zip(checkpoints, boxes):
        axes = Axes(figure, *box, xlim, ylim)
        subset = data[data["checkpoint"].astype(str) == checkpoint]
        for record in subset.to_dict("records"):
            label = str(record.get(color_by, "unknown"))
            incorrect = color_by == "correctness" and label == "incorrect"
            figure.add(f'<circle cx="{axes.x(record["x"]):.2f}" cy="{axes.y(record["y"]):.2f}" r="{(1.8 if incorrect else 1.2) if dense else style.marker_size / 2}" fill="{mapping.get(label, "#777")}" fill-opacity="{.28 if dense else .62}" stroke="{"#111" if incorrect else "none"}" stroke-width="{1 if incorrect else 0}"/>')
        axes.draw("Projection 1", "Projection 2", ticks=2)
        figure.text(box[0] + box[2] / 2, box[1] - 10, checkpoint, anchor="middle", weight="bold")
    external_legend(figure, mapping, x=48, y=20, columns=min(4, len(mapping)))
    return figure, mapping, {"visual_sample_seed": seed, "visual_background_limit": background_limit,
                             "visual_subset_only": bool(background_limit), "shared_xlim": xlim, "shared_ylim": ylim,
                             "rasterized_artists": []}


def uncertainty_view(frame: pd.DataFrame, style: StyleProfile, metric: str = "confidence") -> SvgFigure:
    data = frame.dropna(subset=["x", "y", metric])
    if data.empty:
        raise ValueError(f"No projection uncertainty values for {metric}.")
    checkpoint = list(dict.fromkeys(data["checkpoint"].astype(str)))[-1]
    data = data[data["checkpoint"].astype(str) == checkpoint]
    figure = SvgFigure(style)
    axes = Axes(figure, 64, 28, figure.width_px - 104, figure.height_px - 88,
                padded_limits(data["x"]), padded_limits(data["y"]))
    minimum, maximum = data[metric].min(), data[metric].max()
    for record in data.to_dict("records"):
        color = interpolate_color(float(record[metric]), minimum, maximum, "cividis")
        figure.add(f'<circle cx="{axes.x(record["x"]):.2f}" cy="{axes.y(record["y"]):.2f}" r="2" fill="{color}" fill-opacity=".65"/>')
    axes.draw("Projection 1", "Projection 2")
    figure.text(axes.left + axes.width, axes.top + 12, f"{metric}: {minimum:.2g}–{maximum:.2g}", anchor="end")
    return figure


def _visual_sample(frame: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    identifiers = frame["observation_id"].astype(str).drop_duplicates()
    if len(identifiers) <= size:
        return frame
    hashes = identifiers.map(
        lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    )
    selected = set(identifiers.loc[hashes.sort_values(kind="stable").index[:size]])
    return frame[frame["observation_id"].astype(str).isin(selected)]


def _stratified_visual_sample(frame, size, seed, label_column):
    labels = sorted(frame[label_column].dropna().astype(str).unique())
    selected = []
    for label in labels:
        group = frame[frame[label_column].astype(str) == label]
        quota = max(1, round(size * group["observation_id"].nunique() /
                             frame["observation_id"].nunique()))
        selected.append(_visual_sample(group, quota, seed))
    return pd.concat(selected, ignore_index=True)
