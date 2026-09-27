"""SVG-native scientific axes and multi-panel layout primitives."""

from __future__ import annotations

import html
import math
from dataclasses import dataclass, field

from lab.ner.explicability.visualization.style import StyleProfile


@dataclass
class SvgFigure:
    style: StyleProfile
    title: str | None = None
    elements: list[str] = field(default_factory=list)
    axis_labels: list[str] = field(default_factory=list)
    rasterized_artists: list[str] = field(default_factory=list)

    @property
    def width_px(self) -> int:
        return round(self.style.width * 96)

    @property
    def height_px(self) -> int:
        return round(self.style.height * 96)

    def add(self, element: str) -> None:
        self.elements.append(element)

    def text(self, x, y, text, *, anchor="start", size=None, weight="normal", color="#222222", css=""):
        self.add(
            f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
            f'font-size="{size or self.style.font_size}" font-weight="{weight}" '
            f'fill="{color}" style="{css}">{html.escape(str(text))}</text>'
        )

    def to_svg(self) -> str:
        title = f"<title>{html.escape(self.title)}</title>" if self.title else ""
        metadata = " ".join(f'data-axis-label-{i}="{html.escape(label)}"' for i, label in enumerate(self.axis_labels))
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.style.width}in" '
            f'height="{self.style.height}in" viewBox="0 0 {self.width_px} {self.height_px}" '
            f'data-profile="{self.style.name}" data-grid="false" data-top-spine="false" '
            f'data-right-spine="false" {metadata}>{title}'
            '<defs><marker id="arrowhead" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><path d="M0,0 L7,2.5 L0,5 Z" fill="context-stroke"/></marker></defs>'
            f'<rect width="100%" height="100%" fill="{self.style.background}"/>'
            f'<g font-family="{self.style.family}">{"".join(self.elements)}</g></svg>'
        )


@dataclass(frozen=True)
class Axes:
    figure: SvgFigure
    left: float
    top: float
    width: float
    height: float
    xlim: tuple[float, float]
    ylim: tuple[float, float]

    def x(self, value: float) -> float:
        low, high = self.xlim
        return self.left + (value - low) / max(high - low, 1e-12) * self.width

    def y(self, value: float) -> float:
        low, high = self.ylim
        return self.top + self.height - (value - low) / max(high - low, 1e-12) * self.height

    def draw(self, xlabel: str, ylabel: str, ticks: int = 4) -> None:
        bottom = self.top + self.height
        color = self.figure.style.axis_color
        self.figure.add(f'<path d="M {self.left} {self.top} V {bottom} H {self.left + self.width}" fill="none" stroke="{color}" stroke-width="0.8"/>')
        for index in range(ticks + 1):
            fraction = index / ticks
            x = self.left + fraction * self.width
            y = bottom - fraction * self.height
            xvalue = self.xlim[0] + fraction * (self.xlim[1] - self.xlim[0])
            yvalue = self.ylim[0] + fraction * (self.ylim[1] - self.ylim[0])
            self.figure.add(f'<path d="M {x} {bottom} v 4 M {self.left - 4} {y} h 4" stroke="{color}" stroke-width="0.7"/>')
            self.figure.text(x, bottom + 16, f"{xvalue:.2g}", anchor="middle", size=self.figure.style.font_size - 1)
            self.figure.text(self.left - 7, y + 3, f"{yvalue:.2g}", anchor="end", size=self.figure.style.font_size - 1)
        self.figure.text(self.left + self.width / 2, bottom + 34, xlabel, anchor="middle", size=self.figure.style.label_size)
        self.figure.text(self.left - 39, self.top + self.height / 2, ylabel, anchor="middle", size=self.figure.style.label_size, css="transform-box:fill-box;transform-origin:center;transform:rotate(-90deg)")
        self.figure.axis_labels.extend([xlabel, ylabel])


def panel_grid(figure: SvgFigure, count: int, *, columns: int | None = None, margins=(58, 28, 28, 52), gap=28) -> list[tuple[float, float, float, float]]:
    columns = columns or min(count, 3)
    rows = math.ceil(count / columns)
    left, top, right, bottom = margins
    width = (figure.width_px - left - right - gap * (columns - 1)) / columns
    height = (figure.height_px - top - bottom - gap * (rows - 1)) / rows
    return [(left + (i % columns) * (width + gap), top + (i // columns) * (height + gap), width, height) for i in range(count)]


def panel_label(figure: SvgFigure, box, index: int) -> None:
    figure.text(box[0] - 18, box[1] - 8, chr(65 + index), weight="bold", size=figure.style.font_size + 2)


def padded_limits(values, fraction: float = 0.05) -> tuple[float, float]:
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        raise ValueError("Cannot plot an empty or all-NaN value range.")
    low, high = min(values), max(values)
    padding = (high - low) * fraction if high > low else max(abs(low) * fraction, 1.0)
    return low - padding, high + padding
