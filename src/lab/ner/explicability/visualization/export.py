"""Canonical SVG export plus dependency-gated PDF/PNG derivatives."""

from __future__ import annotations

import os
import re
from pathlib import Path

from lab.ner.explicability.config import VisualizationConfig
from lab.ner.explicability.visualization.layout import SvgFigure


def deterministic_filename(identifier: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", identifier.lower()).strip("_")
    if not cleaned or cleaned in {"plot", "final", "figure"}:
        raise ValueError("Figure identifier must be meaningful.")
    return f"{cleaned}.svg"


def export_figure(figure: SvgFigure, path: str | Path, config: VisualizationConfig) -> list[Path]:
    """Atomically export SVG unconditionally; optional formats never replace it."""
    path = Path(path)
    if path.suffix.lower() != ".svg":
        path = path.with_suffix(".svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(figure.to_svg(), encoding="utf-8")
    os.replace(temporary, path)
    outputs = [path]
    if config.pdf or config.png:
        try:
            import cairosvg
        except ImportError as error:
            raise ImportError("Optional PDF/PNG export requires cairosvg; canonical SVG was written.") from error
        if config.pdf:
            pdf = path.with_suffix(".pdf")
            cairosvg.svg2pdf(url=str(path), write_to=str(pdf))
            outputs.append(pdf)
        if config.png:
            png = path.with_suffix(".png")
            cairosvg.svg2png(url=str(path), write_to=str(png), output_width=round(figure.style.width * config.raster_dpi))
            outputs.append(png)
    return outputs
