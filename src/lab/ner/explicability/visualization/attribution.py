"""Readable token-level diverging attribution displays."""

from __future__ import annotations

import pandas as pd

from lab.ner.explicability.visualization.layout import SvgFigure
from lab.ner.explicability.visualization.style import StyleProfile


def attribution_tokens(frame: pd.DataFrame, style: StyleProfile) -> SvgFigure:
    required = {"token", "attribution"}
    if not required <= set(frame) or frame.empty:
        raise ValueError("Attribution table requires token and attribution columns.")
    figure = SvgFigure(style)
    maximum = max(abs(frame["attribution"].min()), abs(frame["attribution"].max()), 1e-12)
    x, y = 30, 55
    for row in frame.itertuples():
        fraction = float(row.attribution) / maximum
        color = "#0072B2" if fraction >= 0 else "#D55E00"
        opacity = .15 + .75 * abs(fraction)
        width = max(34, len(str(row.token)) * style.font_size * .62 + 12)
        if x + width > figure.width_px - 30:
            x, y = 30, y + 34
        figure.add(f'<rect x="{x}" y="{y - 18}" width="{width}" height="24" rx="3" fill="{color}" fill-opacity="{opacity:.3f}"/>')
        figure.text(x + width / 2, y, row.token, anchor="middle", color="#111")
        x += width + 5
    figure.text(30, 25, "Blue: positive contribution   Orange: negative contribution", size=style.font_size - 1)
    return figure
