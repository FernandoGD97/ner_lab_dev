"""Multi-panel publication summaries and numerical adaptation fingerprints."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lab.ner.explicability.visualization.annotations import line_path
from lab.ner.explicability.visualization.layout import Axes, SvgFigure, padded_limits, panel_grid, panel_label
from lab.ner.explicability.visualization.palette import PALETTES
from lab.ner.explicability.visualization.style import StyleProfile


def adaptation_composite(series: list[tuple[str, pd.DataFrame, str]], style: StyleProfile) -> SvgFigure:
    usable = [(title, frame, metric) for title, frame, metric in series if not frame.empty and metric in frame]
    if not usable:
        raise ValueError("No data for adaptation composite.")
    figure = SvgFigure(style)
    boxes = panel_grid(figure, len(usable), columns=min(2, len(usable)), margins=(55, 36, 25, 45))
    for index, ((title, frame, metric), box) in enumerate(zip(usable, boxes)):
        summary = frame.dropna(subset=[metric]).groupby("checkpoint", sort=False)[metric].mean().reset_index()
        x = np.arange(len(summary))
        axes = Axes(figure, *box, (-.1, max(len(x) - .9, .9)), padded_limits(summary[metric]))
        axes.draw("Checkpoint", title, ticks=min(3, max(len(x) - 1, 1)))
        line_path(axes, x, summary[metric], PALETTES["okabe_ito"][index])
        for xv, yv in zip(x, summary[metric]):
            figure.add(f'<circle cx="{axes.x(xv):.2f}" cy="{axes.y(yv):.2f}" r="3" fill="{PALETTES["okabe_ito"][index]}"/>')
        panel_label(figure, box, index)
    return figure


def fingerprint_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    metrics = []
    specifications = [
        ("trajectories", "distance_pretrained", "representation_drift"),
        ("geometry", "silhouette", "entity_separability"),
        ("geometry", "knn_label_purity", "knn_purity"),
        ("geometry", "effective_rank", "effective_rank"),
        ("geometry", "mean_pairwise_cosine", "anisotropy"),
        ("intrinsic_dimension", "estimate", "intrinsic_dimension"),
        ("parameter_drift", "relative_delta", "parameter_drift"),
        ("forgetting", "forgetting_events", "forgetting_events"),
    ]
    for table_name, column, metric in specifications:
        frame = tables.get(table_name, pd.DataFrame())
        if frame.empty or column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if len(values):
            mean, lower, upper = bootstrap_interval(values, seed=42)
            metrics.append({"metric": metric, "value": float(values.mean()),
                            "median": float(values.median()), "sample_count": len(values),
                            "ci_method": "percentile_bootstrap", "ci_level": .95,
                            "ci_lower": lower, "ci_upper": upper,
                            "source_table": f"tables/{table_name}.parquet"})
    return pd.DataFrame(metrics)


def fingerprint_figure(frame: pd.DataFrame, style: StyleProfile) -> SvgFigure:
    if frame.empty:
        raise ValueError("Fingerprint table is empty.")
    figure = SvgFigure(style)
    left, top, width = 175, 30, figure.width_px - 230
    normalized = []
    for value in frame["value"]:
        normalized.append(float(value))
    scale = max(max(abs(value) for value in normalized), 1e-12)
    for index, row in enumerate(frame.itertuples()):
        y = top + index * 27
        figure.text(left - 8, y + 12, str(row.metric).replace("_", " ").title(), anchor="end")
        bar = abs(float(row.value)) / scale * width
        figure.add(f'<rect x="{left}" y="{y}" width="{bar:.2f}" height="16" fill="#0072B2" fill-opacity=".8"/>')
        figure.text(left + bar + 5, y + 12, f"{row.value:.3g}")
    figure.axis_labels.extend(["Normalized summary magnitude", "Adaptation diagnostic"])
    return figure


def bootstrap_interval(values, seed=42, draws=1000, confidence=.95):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    estimates = np.mean(rng.choice(values, (draws, len(values)), replace=True), axis=1)
    alpha = (1 - confidence) / 2
    return float(np.mean(values)), float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1 - alpha))
