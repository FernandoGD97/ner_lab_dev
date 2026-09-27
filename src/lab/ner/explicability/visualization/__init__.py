"""SVG-first publication visualization and reporting for NER explicability."""

from lab.ner.explicability.visualization.catalog import generate_figures
from lab.ner.explicability.visualization.validation import (
    validate_figure_outputs, validate_svg,
)

__all__ = ["generate_figures", "validate_figure_outputs", "validate_svg"]
