"""Stable external legends that cannot obscure plotted observations."""

from __future__ import annotations

from lab.ner.explicability.visualization.layout import SvgFigure


def external_legend(
    figure: SvgFigure, mapping: dict[str, str], *, x: float | None = None,
    y: float = 32, columns: int | None = None, marker: str = "circle",
) -> None:
    ordered = sorted(mapping)
    columns = columns or min(max(len(ordered), 1), 4)
    x = x if x is not None else figure.width_px - 150
    width = 135
    for index, label in enumerate(ordered):
        column, row = index % columns, index // columns
        left, top = x + column * width, y + row * 18
        color = mapping[label]
        if marker == "line":
            figure.add(f'<path d="M {left} {top} h 14" stroke="{color}" stroke-width="2"/>')
        else:
            figure.add(f'<circle cx="{left + 7}" cy="{top - 3}" r="4" fill="{color}" stroke="#333" stroke-width="0.4"/>')
        figure.text(left + 19, top, label, size=figure.style.font_size - 1)
