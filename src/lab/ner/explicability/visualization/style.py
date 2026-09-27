"""Central paper/presentation typography and physical figure profiles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StyleProfile:
    name: str
    width: float
    height: float
    font_size: float
    label_size: float
    line_width: float
    marker_size: float
    family: str = "Arial, Helvetica, sans-serif"
    background: str = "#FFFFFF"
    axis_color: str = "#4D4D4D"
    grid: bool = False
    top_spine: bool = False
    right_spine: bool = False


PROFILES = {
    "paper": StyleProfile("paper", 7.2, 4.3, 9, 9, 1.25, 3.0),
    "presentation": StyleProfile("presentation", 12.8, 7.2, 18, 18, 2.2, 6.0),
}
SIZE_PRESETS = {
    "single_column": (3.45, 2.6), "double_column": (7.2, 4.3),
    "wide": (7.2, 3.2), "square": (4.5, 4.5), "presentation_16_9": (12.8, 7.2),
}


def profile(name: str = "paper", size: str | None = None) -> StyleProfile:
    base = PROFILES[name]
    if size is None:
        return base
    width, height = SIZE_PRESETS[size]
    return StyleProfile(**{**base.__dict__, "width": width, "height": height})
