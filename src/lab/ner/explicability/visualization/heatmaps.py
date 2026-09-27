"""Sequential scientific heatmaps for checkpoint/layer matrices."""

from __future__ import annotations

import math

import pandas as pd

from lab.ner.explicability.visualization.layout import SvgFigure
from lab.ner.explicability.visualization.palette import interpolate_color
from lab.ner.explicability.visualization.style import StyleProfile


def heatmap_figure(
    frame: pd.DataFrame, row: str, column: str, value: str, style: StyleProfile,
    *, xlabel: str, ylabel: str, title: str | None = None, colormap: str = "viridis",
) -> SvgFigure:
    data = frame[[row, column, value]].dropna()
    if data.empty:
        raise ValueError(f"No finite values for heatmap {value}.")
    rows = sorted(data[row].astype(str).unique())
    columns = sorted(data[column].astype(str).unique(), key=_natural_key)
    values = data[value].astype(float)
    minimum, maximum = values.min(), values.max()
    figure = SvgFigure(style, title)
    left, top, right, bottom = 105, 30, 48, 70
    width, height = figure.width_px - left - right, figure.height_px - top - bottom
    cell_width, cell_height = width / len(columns), height / len(rows)
    lookup = {(str(record[row]), str(record[column])): float(record[value]) for record in data.to_dict("records")}
    for rindex, rlabel in enumerate(rows):
        figure.text(left - 8, top + (rindex + .6) * cell_height, rlabel, anchor="end")
        for cindex, clabel in enumerate(columns):
            number = lookup.get((rlabel, clabel))
            fill = "#F2F2F2" if number is None else interpolate_color(number, minimum, maximum, colormap)
            figure.add(f'<rect x="{left + cindex * cell_width:.2f}" y="{top + rindex * cell_height:.2f}" width="{cell_width + .2:.2f}" height="{cell_height + .2:.2f}" fill="{fill}"/>')
    for cindex, label in enumerate(columns):
        figure.text(left + (cindex + .5) * cell_width, top + height + 17, label, anchor="middle", size=style.font_size - 1)
    figure.text(left + width / 2, figure.height_px - 15, xlabel, anchor="middle", size=style.label_size)
    figure.text(18, top + height / 2, ylabel, anchor="middle", size=style.label_size, css="transform-box:fill-box;transform-origin:center;transform:rotate(-90deg)")
    figure.axis_labels.extend([xlabel, ylabel])
    _colorbar(figure, figure.width_px - 27, top, 10, height, minimum, maximum, colormap)
    return figure


def _colorbar(figure, x, y, width, height, minimum, maximum, colormap):
    for index in range(50):
        fraction = index / 49
        figure.add(f'<rect x="{x}" y="{y + (1 - fraction) * height:.2f}" width="{width}" height="{height / 49 + 1:.2f}" fill="{interpolate_color(fraction, 0, 1, colormap)}"/>')
    figure.text(x + width + 3, y + 4, f"{maximum:.2g}", size=figure.style.font_size - 2)
    figure.text(x + width + 3, y + height, f"{minimum:.2g}", size=figure.style.font_size - 2)


def _natural_key(value: str):
    return tuple(int(part) if part.isdigit() else part for part in __import__("re").split(r"(\d+)", value))
