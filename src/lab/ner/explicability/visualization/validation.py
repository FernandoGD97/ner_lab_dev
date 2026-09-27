"""Structural, source, style, and manifest validation for publication artifacts."""

from __future__ import annotations

import xml.etree.ElementTree as ET
import json
from pathlib import Path

import pandas as pd


def validate_svg(path: str | Path, expected_labels: tuple[str, ...] = ()) -> list[str]:
    path = Path(path)
    errors = []
    if not path.is_file() or path.stat().st_size < 200:
        return [f"Missing or empty SVG: {path}"]
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as error:
        return [f"Invalid SVG XML {path}: {error}"]
    if not root.tag.endswith("svg"):
        errors.append(f"Not an SVG root: {path}")
    for dimension in ("width", "height", "viewBox"):
        if not root.attrib.get(dimension):
            errors.append(f"SVG lacks {dimension}: {path}")
    try:
        _, _, width, height = map(float, root.attrib.get("viewBox", "0 0 0 0").split())
        for element in root.iter():
            if not element.tag.endswith("text"):
                continue
            x = float(element.attrib.get("x", 0))
            y = float(element.attrib.get("y", 0))
            if x < -1 or x > width + 1 or y < -1 or y > height + 1:
                errors.append(f"Text/legend is obviously clipped in {path}")
                break
    except (TypeError, ValueError):
        errors.append(f"SVG has invalid dimensions: {path}")
    if root.attrib.get("data-grid") != "false":
        errors.append(f"Grid is not disabled: {path}")
    if root.attrib.get("data-top-spine") != "false" or root.attrib.get("data-right-spine") != "false":
        errors.append(f"Top/right spines are not hidden: {path}")
    labels = set(root.attrib.values())
    for label in expected_labels:
        if label not in labels:
            errors.append(f"Expected axis label {label!r} absent from {path}")
    return errors


def validate_figure_outputs(root: str | Path, mandatory_ids: list[str] | None = None) -> list[str]:
    root = Path(root)
    manifest_path = root / "tables" / "figure_manifest.parquet"
    if not manifest_path.exists():
        return ["Missing tables/figure_manifest.parquet"]
    manifest = pd.read_parquet(manifest_path)
    errors = []
    mandatory_ids = mandatory_ids or manifest["figure_id"].astype(str).tolist()
    for figure_id in mandatory_ids:
        rows = manifest[manifest["figure_id"] == figure_id]
        if rows.empty:
            errors.append(f"Figure manifest lacks mandatory figure: {figure_id}")
            continue
        row = rows.iloc[0]
        figure = root / str(row["filename"])
        expected = tuple(json.loads(row.get("axis_labels", "[]") or "[]"))
        errors.extend(validate_svg(figure, expected))
        source = root / str(row["source_table"])
        if not source.is_file():
            errors.append(f"Figure source is missing: {source}")
        else:
            try:
                if pd.read_parquet(source).empty:
                    errors.append(f"Figure source is empty: {source}")
            except (OSError, ValueError) as error:
                errors.append(f"Figure source is unreadable: {source}: {error}")
    return errors
