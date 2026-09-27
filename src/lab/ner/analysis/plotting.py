"""Centralized dependency-light scientific SVG plotting and export style."""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

THEME_VERSION = "1.0.0"
OKABE_ITO = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000")
SIZE_PRESETS = {
    "single-column": (3.5, 2.7),
    "1.5-column": (5.2, 3.4),
    "double-column": (7.2, 4.4),
    "presentation": (10.0, 5.625),
}


@dataclass(frozen=True)
class PlotTheme:
    """One style contract shared by every paper figure."""

    font_family: str = "Arial, Helvetica, sans-serif"
    base_font_size: float = 9.0
    title_font_size: float = 10.5
    axis_width: float = 1.0
    tick_size: float = 3.5
    marker_size: float = 5.0
    line_width: float = 1.6
    palette: tuple[str, ...] = OKABE_ITO
    background: str = "#FFFFFF"
    savefig_dpi: int = 2100
    svg_fonttype: str = "none"
    top_spine: bool = False
    right_spine: bool = False
    default_grid: bool = False
    legend_location: str = "outside-above"


class SVGFigure:
    """A vector figure with inspectable style properties for structural tests."""

    def __init__(self, title: str, size="double-column", theme: PlotTheme | None = None):
        self.theme = theme or PlotTheme()
        self.width_in, self.height_in = SIZE_PRESETS[size]
        self.width, self.height = self.width_in * 96, self.height_in * 96
        self.title = title
        self.elements: list[str] = []
        self._text(self.width / 2, 22, title, anchor="middle", size=self.theme.title_font_size, weight="bold")

    def _text(self, x, y, value, *, anchor="start", size=None, weight="normal", rotate=None):
        transform = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate else ""
        self.elements.append(
            f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
            f'font-size="{size or self.theme.base_font_size}pt" font-weight="{weight}"{transform}>'
            f'{html.escape(str(value))}</text>'
        )

    def line(self, x1, y1, x2, y2, color="#333333", width=None, dash=None):
        dashed = f' stroke-dasharray="{dash}"' if dash else ""
        self.elements.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="{width or self.theme.axis_width}"{dashed}/>'
        )

    def rect(self, x, y, width, height, color, opacity=1.0, stroke="none"):
        self.elements.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(0, width):.2f}" height="{max(0, height):.2f}" '
            f'fill="{color}" fill-opacity="{opacity}" stroke="{stroke}"/>'
        )

    def circle(self, x, y, radius, color, stroke="#222222"):
        self.elements.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{color}" stroke="{stroke}"/>'
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = "\n".join(self.elements)
        svg = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width_in}in" height="{self.height_in}in" '
            f'viewBox="0 0 {self.width:.2f} {self.height:.2f}" role="img">\n'
            f'<metadata>theme={THEME_VERSION}; canonical=vector; raster-dpi={self.theme.savefig_dpi}; '
            f'svg-fonttype={self.theme.svg_fonttype}</metadata>\n'
            f'<rect width="100%" height="100%" fill="{self.theme.background}"/>\n'
            f'<g font-family="{html.escape(self.theme.font_family)}" fill="#222222">\n{body}\n</g>\n</svg>\n'
        )
        path.write_text(svg, encoding="utf-8")
        return path


def categorical_metric_figure(
    frame: pd.DataFrame,
    *,
    category: str,
    metric: str,
    title: str,
    ylabel: str,
    path: str | Path,
    support: str = "support",
    threshold_position: int | None = None,
) -> SVGFigure:
    fig = SVGFigure(title)
    labels = frame[category].astype(str).tolist() if not frame.empty else []
    values = pd.to_numeric(frame.get(metric, pd.Series(dtype=float)), errors="coerce").fillna(0).tolist()
    left, top, right, bottom = 72, 48, fig.width - 22, fig.height - 72
    _axes(fig, left, top, right, bottom, ylabel)
    if not labels:
        fig._text((left + right) / 2, (top + bottom) / 2, "No eligible data", anchor="middle")
        fig.save(path)
        return fig
    step = (right - left) / max(1, len(labels))
    points = []
    for index, (label, value) in enumerate(zip(labels, values)):
        x, y = left + step * (index + 0.5), bottom - max(0, min(1, value)) * (bottom - top)
        points.append((x, y))
        fig._text(x, bottom + 16, label, anchor="middle", size=7.5, rotate=-25 if len(labels) > 6 else None)
        if support in frame:
            fig._text(x, y - 10, f"N={int(frame.iloc[index][support])}", anchor="middle", size=7)
        if {"ci_low", "ci_high"}.issubset(frame.columns) and pd.notna(frame.iloc[index]["ci_low"]):
            low = bottom - float(frame.iloc[index]["ci_low"]) * (bottom - top)
            high = bottom - float(frame.iloc[index]["ci_high"]) * (bottom - top)
            fig.line(x, high, x, low, color="#333333")
            fig.line(x - 4, high, x + 4, high, color="#333333")
            fig.line(x - 4, low, x + 4, low, color="#333333")
    for first, second in zip(points, points[1:]):
        fig.line(*first, *second, color=fig.theme.palette[0], width=fig.theme.line_width)
    for x, y in points:
        fig.circle(x, y, fig.theme.marker_size, fig.theme.palette[0])
    if threshold_position is not None and 0 < threshold_position < len(labels):
        x = left + step * threshold_position
        fig.line(x, top, x, bottom, color="#777777", dash="4,4")
        fig._text(x + 3, top + 10, "zero-shot threshold", size=7)
    fig.save(path)
    return fig


def grouped_bar_figure(frame, *, category, series, metric, title, ylabel, path):
    fig = SVGFigure(title)
    left, top, right, bottom = 72, 55, fig.width - 22, fig.height - 72
    _axes(fig, left, top, right, bottom, ylabel)
    categories = sorted(frame[category].astype(str).unique()) if not frame.empty else []
    groups = sorted(frame[series].astype(str).unique()) if not frame.empty else []
    if not categories or not groups:
        fig._text((left + right) / 2, (top + bottom) / 2, "No eligible data", anchor="middle")
    else:
        category_width = (right - left) / len(categories)
        bar_width = category_width * 0.72 / len(groups)
        for c_index, category_value in enumerate(categories):
            fig._text(left + category_width * (c_index + .5), bottom + 17, category_value, anchor="middle", size=7.5)
            for g_index, group in enumerate(groups):
                selected = frame[(frame[category].astype(str) == category_value) & (frame[series].astype(str) == group)]
                value = float(selected.iloc[0][metric]) if len(selected) else 0.0
                x = left + category_width * c_index + category_width * .14 + g_index * bar_width
                height = value * (bottom - top)
                fig.rect(x, bottom - height, bar_width * .88, height, fig.theme.palette[g_index % len(fig.theme.palette)])
        _legend(fig, groups)
    fig.save(path)
    return fig


def horizontal_bar_figure(frame, *, label, value, title, xlabel, path, color_index=1):
    fig = SVGFigure(title)
    ordered = (
        frame.sort_values(value, ascending=True).tail(15)
        if not frame.empty and label in frame and value in frame else pd.DataFrame(columns=[label, value])
    )
    left, top, right, bottom = 155, 48, fig.width - 30, fig.height - 48
    _axes(fig, left, top, right, bottom, xlabel, horizontal=True)
    maximum = max(float(ordered[value].max()), 1e-12) if len(ordered) else 1.0
    step = (bottom - top) / max(1, len(ordered))
    for index, row in enumerate(ordered.itertuples(index=False)):
        y = top + step * index + step * .15
        width = float(getattr(row, value)) / maximum * (right - left)
        fig.rect(left, y, width, step * .65, fig.theme.palette[color_index])
        fig._text(left - 7, y + step * .48, getattr(row, label), anchor="end", size=7.5)
        fig._text(left + width + 4, y + step * .48, f"{float(getattr(row, value)):.3f}", size=7)
    fig.save(path)
    return fig


def heatmap_figure(matrix: pd.DataFrame, *, title: str, path: str | Path, normalized=False):
    fig = SVGFigure(title)
    if matrix.empty or not len(matrix.columns):
        matrix = pd.DataFrame([[0]], index=["No eligible data"], columns=["No eligible data"])
    rows, columns = list(matrix.index.astype(str)), list(matrix.columns.astype(str))
    left, top, right, bottom = 125, 55, fig.width - 35, fig.height - 75
    maximum = float(matrix.to_numpy().max()) if matrix.size else 0.0
    for i, row in enumerate(rows):
        fig._text(left - 7, top + (i + .55) * (bottom - top) / len(rows), row, anchor="end", size=7.5)
    for j, column in enumerate(columns):
        fig._text(left + (j + .5) * (right - left) / len(columns), bottom + 17, column, anchor="middle", size=7.5, rotate=-25)
    for i, row in enumerate(rows):
        for j, column in enumerate(columns):
            value = float(matrix.loc[row, column])
            intensity = value / maximum if maximum else 0.0
            color = _blend("#FFFFFF", "#0072B2", intensity)
            x, y = left + j * (right - left) / len(columns), top + i * (bottom - top) / len(rows)
            width, height = (right - left) / len(columns), (bottom - top) / len(rows)
            fig.rect(x, y, width, height, color)
            shown = f"{value:.1%}" if normalized else f"{int(value)}"
            fig.elements.append(f'<text x="{x + width/2:.2f}" y="{y + height*.58:.2f}" text-anchor="middle" font-size="7.5pt" fill="{"#FFFFFF" if intensity > .55 else "#222222"}">{shown}</text>')
    fig.save(path)
    return fig


def histogram_figure(values, *, title, xlabel, path, bins=15, display_quantile=None):
    fig = SVGFigure(title)
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    out_of_range = 0
    if display_quantile is not None and len(series) > 2:
        tail = (1 - display_quantile) / 2
        low, high = series.quantile(tail), series.quantile(1 - tail)
        out_of_range = int(((series < low) | (series > high)).sum())
        series = series.clip(low, high)
    left, top, right, bottom = 72, 48, fig.width - 25, fig.height - 62
    _axes(fig, left, top, right, bottom, xlabel, horizontal=True)
    if len(series):
        counts, edges = _histogram(series.tolist(), bins)
        maximum = max(counts) or 1
        width = (right - left) / len(counts)
        for index, count in enumerate(counts):
            height = count / maximum * (bottom - top)
            fig.rect(left + index * width, bottom - height, width * .92, height, fig.theme.palette[4])
        fig._text(left, bottom + 16, f"{edges[0]:.1f}", anchor="middle", size=7)
        fig._text(right, bottom + 16, f"{edges[-1]:.1f}", anchor="middle", size=7)
        if out_of_range:
            fig._text(right, top + 10, f"Display clipped at central {display_quantile:.0%}; {out_of_range} outside", anchor="end", size=7)
    else:
        fig._text((left + right) / 2, (top + bottom) / 2, "No eligible data", anchor="middle")
    fig.save(path)
    return fig


def category_strip_figure(frame, *, category, value, title, xlabel, path):
    """Deterministic score distribution using color plus vertically separated categories."""
    fig = SVGFigure(title)
    left, top, right, bottom = 135, 48, fig.width - 28, fig.height - 58
    _axes(fig, left, top, right, bottom, xlabel, horizontal=True)
    categories = sorted(frame[category].dropna().astype(str).unique()) if not frame.empty else []
    step = (bottom - top) / max(1, len(categories))
    for category_index, name in enumerate(categories):
        y = top + step * (category_index + .5)
        fig._text(left - 7, y + 3, name, anchor="end", size=7.5)
        selected = pd.to_numeric(frame.loc[frame[category].astype(str) == name, value], errors="coerce").dropna()
        for point_index, score in enumerate(selected):
            jitter = ((point_index % 5) - 2) * 1.6
            x = left + max(0, min(1, float(score))) * (right - left)
            fig.circle(x, y + jitter, 2.7, fig.theme.palette[category_index % len(fig.theme.palette)], stroke="none")
    if not categories:
        fig._text((left + right) / 2, (top + bottom) / 2, "No confidence scores available", anchor="middle")
    fig.save(path)
    return fig


def _axes(fig, left, top, right, bottom, label, horizontal=False):
    fig.line(left, top, left, bottom)
    fig.line(left, bottom, right, bottom)
    # No top/right spines and no default gridlines by design.
    if horizontal:
        fig._text((left + right) / 2, fig.height - 12, label, anchor="middle")
    else:
        fig._text(15, (top + bottom) / 2, label, anchor="middle", rotate=-90)


def _legend(fig, labels):
    start = max(30, fig.width / 2 - len(labels) * 45)
    for index, label in enumerate(labels):
        x = start + index * 90
        fig.rect(x, 34, 10, 8, fig.theme.palette[index % len(fig.theme.palette)])
        fig._text(x + 14, 42, label, size=7.5)


def _blend(first, second, amount):
    parse = lambda color: tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
    a, b = parse(first), parse(second)
    values = [round(x + (y - x) * max(0, min(1, amount))) for x, y in zip(a, b)]
    return "#" + "".join(f"{value:02X}" for value in values)


def _histogram(values, bins):
    low, high = min(values), max(values)
    if low == high:
        return [len(values)], [low, high]
    width = (high - low) / bins
    counts = [0] * bins
    for value in values:
        counts[min(bins - 1, int((value - low) / width))] += 1
    return counts, [low + index * width for index in range(bins + 1)]
