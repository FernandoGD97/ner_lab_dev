"""Dataset Cartography, forgetting, learning, and prediction-transition figures."""

from __future__ import annotations

import pandas as pd

from lab.ner.explicability.visualization.distributions import ecdf_figure
from lab.ner.explicability.visualization.heatmaps import heatmap_figure
from lab.ner.explicability.visualization.layout import Axes, SvgFigure, padded_limits
from lab.ner.explicability.visualization.legends import external_legend
from lab.ner.explicability.visualization.palette import entity_colors
from lab.ner.explicability.visualization.style import StyleProfile
from lab.ner.explicability.visualization.annotations import line_path


def cartography_figure(frame: pd.DataFrame, style: StyleProfile, palette="okabe_ito"):
    data = frame[frame["representation_level"] == "token"] if "representation_level" in frame else frame
    data = data.dropna(subset=["mean_confidence", "confidence_variability"])
    if data.empty:
        raise ValueError("No token-level Dataset Cartography data.")
    mapping = entity_colors(data["cartography_region"].astype(str), palette)
    figure = SvgFigure(style)
    axes = Axes(figure, 64, 24, figure.width_px - 105, figure.height_px - 80,
                padded_limits(data["mean_confidence"]), padded_limits(data["confidence_variability"]))
    axes.draw("Mean gold-label confidence", "Confidence variability")
    for row in data.to_dict("records"):
        label = str(row["cartography_region"])
        figure.add(f'<circle cx="{axes.x(row["mean_confidence"]):.2f}" cy="{axes.y(row["confidence_variability"]):.2f}" r="2.2" fill="{mapping[label]}" fill-opacity=".55" stroke="none"/>')
    external_legend(figure, mapping, x=figure.width_px - 155, y=30, columns=1)
    return figure, mapping


def forgetting_distribution(frame: pd.DataFrame, style: StyleProfile) -> SvgFigure:
    data = frame[frame["representation_level"] == "token"] if "representation_level" in frame else frame
    return ecdf_figure(data["forgetting_events"], style, "Forgetting events per observation")


def first_learning_distribution(frame: pd.DataFrame, style: StyleProfile) -> SvgFigure:
    data = frame[frame["representation_level"] == "token"] if "representation_level" in frame else frame
    learned = data["first_correct_checkpoint"].dropna().astype(str)
    order = {label: index for index, label in enumerate(dict.fromkeys(learned))}
    return ecdf_figure(learned.map(order), style, "First-correct checkpoint order")


def transition_heatmap(frame: pd.DataFrame, style: StyleProfile, normalized=True) -> SvgFigure:
    value = "transition_probability" if normalized else "transition_count"
    summary = frame.groupby(["label_from", "label_to"], as_index=False)[value].sum()
    return heatmap_figure(summary, "label_from", "label_to", value, style,
                          xlabel="Predicted label after", ylabel="Predicted label before",
                          colormap="cividis")


def learning_timeline(frame: pd.DataFrame, style: StyleProfile, observation_id: str | None = None) -> SvgFigure:
    data = frame[frame["record_type"] == "checkpoint_history"] if "record_type" in frame else frame
    if "representation_level" in data:
        data = data[data["representation_level"] == "token"]
    if data.empty:
        raise ValueError("No checkpoint histories for learning timeline.")
    observation_id = observation_id or str(data["observation_id"].iloc[0])
    data = data[data["observation_id"].astype(str) == observation_id].sort_values("checkpoint_order")
    x = list(range(len(data)))
    confidence = data["gold_confidence"].astype(float)
    figure = SvgFigure(style)
    axes = Axes(figure, 64, 25, figure.width_px - 100, figure.height_px - 82,
                (-.1, max(len(x) - .9, .9)), (0, 1))
    axes.draw("Checkpoint", "Gold-label confidence")
    line_path(axes, x, confidence, "#0072B2")
    for index, (value, correct) in enumerate(zip(confidence, data["correct"].fillna(False))):
        color = "#0072B2" if correct else "#E69F00"
        marker = 4 if correct else 5
        figure.add(f'<circle cx="{axes.x(index):.2f}" cy="{axes.y(value):.2f}" r="{marker}" fill="{color}" stroke="#222" stroke-width=".6"/>')
    figure.text(axes.left, axes.top + 13, f"Observation: {observation_id}", weight="bold")
    return figure
