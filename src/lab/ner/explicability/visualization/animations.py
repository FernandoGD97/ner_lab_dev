"""Stable-coordinate temporal animation generated from projection Parquet data."""

from __future__ import annotations

import html
import hashlib
from pathlib import Path

import pandas as pd
import yaml

from lab.ner.explicability.config import resolve_config
from lab.ner.explicability.finalize import explicability_root
from lab.ner.explicability.visualization.layout import padded_limits
from lab.ner.explicability.visualization.palette import entity_colors


def animation_frames(
    frame: pd.DataFrame, max_observations: int | None = None, seed: int = 42
) -> list[pd.DataFrame]:
    """Return identity-sorted frames after verifying a fixed observation population."""
    if max_observations and frame["observation_id"].nunique() > max_observations:
        identifiers = frame["observation_id"].astype(str).drop_duplicates()
        ranked = identifiers.map(
            lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
        ).sort_values(kind="stable")
        selected = set(identifiers.loc[ranked.index[:max_observations]])
        frame = frame[frame["observation_id"].astype(str).isin(selected)]
    frames = [group.sort_values("observation_id", kind="stable") for _, group in frame.groupby("checkpoint", sort=False)]
    if not frames:
        raise ValueError("No projection frames for animation.")
    baseline = frames[0]["observation_id"].astype(str).tolist()
    if any(group["observation_id"].astype(str).tolist() != baseline for group in frames[1:]):
        raise ValueError("Animation frames do not contain identical stable observation IDs.")
    return frames


def animated_svg(
    frame: pd.DataFrame, path: str | Path, palette="okabe_ito",
    seconds_per_frame=1.0, max_observations: int | None = None, seed: int = 42,
) -> Path:
    frames = animation_frames(frame, max_observations, seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height, left, top = 960, 540, 65, 30
    xlim, ylim = padded_limits(frame["x"]), padded_limits(frame["y"])
    colors = entity_colors(frame["gold_label"].astype(str), palette)
    duration = len(frames) * seconds_per_frame
    groups = []
    for index, group in enumerate(frames):
        circles = []
        for row in group.itertuples():
            x = left + (row.x - xlim[0]) / (xlim[1] - xlim[0]) * (width - 100)
            y = top + (ylim[1] - row.y) / (ylim[1] - ylim[0]) * (height - 90)
            circles.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2" fill="{colors.get(str(row.gold_label), "#777")}" fill-opacity=".6"/>')
        begin = index * seconds_per_frame
        groups.append(f'<g visibility="hidden">{"".join(circles)}<text x="{width - 30}" y="25" text-anchor="end">{html.escape(str(group.checkpoint.iloc[0]))}</text><set attributeName="visibility" to="visible" begin="{begin}s" dur="{seconds_per_frame}s" repeatCount="indefinite"/></g>')
    svg = f'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" width="10in" height="5.625in" viewBox="0 0 {width} {height}" data-fixed-limits="true"><rect width="100%" height="100%" fill="white"/><g font-family="Arial, Helvetica, sans-serif">{"".join(groups)}</g></svg>'
    path.write_text(svg, encoding="utf-8")
    return path


def export_mp4(
    frame: pd.DataFrame, path: str | Path, fps: int = 2,
    max_observations: int | None = None, seed: int = 42,
) -> Path | None:
    """Export MP4 when optional cairosvg/imageio-ffmpeg dependencies are installed."""
    try:
        import cairosvg
        import imageio.v3 as iio
    except ImportError:
        return None
    import tempfile
    frames = animation_frames(frame, max_observations, seed)
    images = []
    with tempfile.TemporaryDirectory() as directory:
        for index, single in enumerate(frames):
            svg_path = animated_svg(single, Path(directory) / f"frame_{index:04d}.svg", seconds_per_frame=1)
            png_path = Path(directory) / f"frame_{index:04d}.png"
            cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), output_width=1920)
            images.append(iio.imread(png_path))
        iio.imwrite(path, images, fps=fps, codec="libx264")
    return Path(path)


def regenerate_animation(run_dir: str | Path) -> dict[str, Path]:
    """Regenerate animation artifacts independently using projection.parquet only."""
    root = explicability_root(run_dir)
    config = resolve_config(
        yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8")) or {}
    ).visualization
    frame = pd.read_parquet(root / "tables" / "projection.parquet")
    if "representation_level" in frame:
        frame = frame[frame["representation_level"] == "token"]
    if "layer" in frame and frame["layer"].notna().any():
        frame = frame[frame["layer"] == frame["layer"].max()]
    svg = animated_svg(frame, root / "animations" / "temporal_embedding_evolution.svg",
                       config.palette, 1 / config.animation_fps,
                       config.visual_sample_size, config.visual_sample_seed)
    outputs = {"svg": svg}
    mp4 = export_mp4(frame, root / "animations" / "temporal_embedding_evolution.mp4",
                      config.animation_fps, config.visual_sample_size,
                      config.visual_sample_seed)
    if mp4:
        outputs["mp4"] = mp4
    return outputs
