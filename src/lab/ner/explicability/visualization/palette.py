"""Accessible categorical palettes and stable entity-color assignment."""

from __future__ import annotations

PALETTES = {
    "okabe_ito": ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"),
    "tol_bright": ("#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"),
    "colorbrewer_dark2": ("#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02", "#A6761D", "#666666"),
}
CONTINUOUS = {"viridis": ("#440154", "#3B528B", "#21918C", "#5EC962", "#FDE725"),
              "cividis": ("#00204C", "#414D6B", "#7C7B78", "#BCAF6F", "#FFEA46")}


def entity_colors(labels, palette: str = "okabe_ito") -> dict[str, str]:
    """Map lexical label order to stable colors, reserving gray for outside class."""
    colors = PALETTES[palette]
    ordered = sorted({str(label) for label in labels if label is not None})
    mapping, index = {}, 0
    for label in ordered:
        if label in {"O", "non_entity", "IGNORED"}:
            mapping[label] = "#B3B3B3"
        else:
            mapping[label] = colors[index % len(colors)]
            index += 1
    return mapping


def interpolate_color(value: float, minimum: float, maximum: float, name: str = "viridis") -> str:
    colors = CONTINUOUS[name]
    fraction = 0.5 if maximum <= minimum else max(0.0, min(1.0, (value - minimum) / (maximum - minimum)))
    position = fraction * (len(colors) - 1)
    left = int(position)
    right = min(left + 1, len(colors) - 1)
    blend = position - left
    a, b = colors[left].lstrip("#"), colors[right].lstrip("#")
    channels = [round(int(a[i:i + 2], 16) * (1 - blend) + int(b[i:i + 2], 16) * blend) for i in (0, 2, 4)]
    return "#" + "".join(f"{channel:02X}" for channel in channels)
