"""Reusable restrained annotations and confidence bands."""

from __future__ import annotations

from lab.ner.explicability.visualization.layout import Axes, SvgFigure


def arrow(figure: SvgFigure, start, end, color: str, width: float = 1.2) -> None:
    x1, y1 = start
    x2, y2 = end
    figure.add(f'<path d="M {x1:.2f} {y1:.2f} L {x2:.2f} {y2:.2f}" stroke="{color}" stroke-width="{width}" fill="none" marker-end="url(#arrowhead)"/>')


def line_path(axes: Axes, x_values, y_values, color: str, *, width=None, dash: str | None = None) -> None:
    points = " ".join(f"{axes.x(float(x)):.2f},{axes.y(float(y)):.2f}" for x, y in zip(x_values, y_values))
    style = f' stroke-dasharray="{dash}"' if dash else ""
    axes.figure.add(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{width or axes.figure.style.line_width}"{style}/>' )
