from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from lab.ner.explicability.config import VisualizationConfig, resolve_config
from lab.ner.explicability.finalize import finalize_run, initialize_run, read_manifest
from lab.ner.explicability.storage import write_table
from lab.ner.explicability.utils import atomic_json
from lab.ner.explicability.visualization.animations import animation_frames
from lab.ner.explicability.visualization.composites import adaptation_composite
from lab.ner.explicability.visualization.embeddings import temporal_small_multiples
from lab.ner.explicability.visualization.export import deterministic_filename, export_figure
from lab.ner.explicability.visualization.layout import Axes, SvgFigure
from lab.ner.explicability.visualization.legends import external_legend
from lab.ner.explicability.visualization.palette import PALETTES, entity_colors
from lab.ner.explicability.visualization.style import profile
from lab.ner.explicability.visualization.validation import validate_svg


class Part3Tests(unittest.TestCase):
    def _projection(self) -> pd.DataFrame:
        rows = []
        for time, checkpoint in enumerate(("step_000_pretrained", "step_001", "step_final")):
            for index in range(8):
                label = "O" if index < 4 else "B-X"
                rows.append({
                    "observation_id": f"id-{index}", "checkpoint": checkpoint,
                    "layer": 0, "representation_level": "token",
                    "projection_method": "shared_pca", "x": index + time * .1,
                    "y": (index % 2) + time * .2, "gold_label": label,
                    "predicted_label": label, "correct": True, "confidence": .8,
                })
        return pd.DataFrame(rows)

    def test_profiles_palette_and_entity_consistency(self) -> None:
        paper = profile("paper")
        presentation = profile("presentation")
        self.assertLess(paper.font_size, presentation.font_size)
        self.assertAlmostEqual(presentation.width / presentation.height, 16 / 9)
        self.assertFalse(paper.top_spine or paper.right_spine or paper.grid)
        self.assertEqual(len(PALETTES["okabe_ito"]), 8)
        first = entity_colors(["B-X", "O", "B-Y"])
        second = entity_colors(["B-Y", "B-X", "O"])
        self.assertEqual(first, second)
        self.assertEqual(first["O"], "#B3B3B3")

    def test_svg_export_style_axis_labels_and_legend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            figure = SvgFigure(profile("paper"))
            axes = Axes(figure, 50, 20, 400, 220, (0, 1), (0, 1))
            axes.draw("Training epoch", "CKA")
            external_legend(figure, {"Entity": "#0072B2"}, x=470)
            path = Path(directory) / deterministic_filename("cka_to_pretrained")
            export_figure(figure, path, VisualizationConfig())
            self.assertFalse(validate_svg(path, ("Training epoch", "CKA")))
            root = ET.parse(path).getroot()
            self.assertEqual(root.attrib["data-grid"], "false")
            self.assertEqual(root.attrib["data-top-spine"], "false")
            self.assertIn("Entity", path.read_text())
            self.assertEqual(deterministic_filename("Entity Centroid-Trajectories"), "entity_centroid_trajectories.svg")

    def test_shared_temporal_limits_and_visual_sampling_metadata(self) -> None:
        figure, colors, metadata = temporal_small_multiples(
            self._projection(), profile("paper"), background_limit=2, seed=7
        )
        self.assertEqual(metadata["shared_xlim"], metadata["shared_xlim"])
        self.assertTrue(metadata["visual_subset_only"])
        self.assertEqual(metadata["visual_sample_seed"], 7)
        self.assertIn("B-X", colors)
        self.assertGreater(figure.to_svg().count("Projection 1"), 1)

    def test_composite_creation(self) -> None:
        frame = pd.DataFrame({"checkpoint": ["a", "b", "c"], "metric": [.1, .4, .6]})
        figure = adaptation_composite([
            ("Performance", frame, "metric"), ("Drift", frame, "metric")
        ], profile("paper"))
        svg = figure.to_svg()
        self.assertIn(">A<", svg)
        self.assertIn(">B<", svg)

    def test_animation_frame_consistency(self) -> None:
        frames = animation_frames(self._projection())
        self.assertEqual(len(frames), 3)
        broken = self._projection().query("not (checkpoint == 'step_final' and observation_id == 'id-0')")
        with self.assertRaisesRegex(ValueError, "identical stable observation"):
            animation_frames(broken)

    def test_parquet_only_replot_manifest_and_report(self) -> None:
        from lab.ner.explicability.visualization import generate_figures

        with tempfile.TemporaryDirectory() as directory:
            config = resolve_config({
                "enabled": True, "snapshots": {"cleanup_policy": "never"},
                "visualization": {"animation": False},
            })
            root = initialize_run(directory, config, model="tiny", dataset="synthetic")
            write_table(self._projection(), root / "tables" / "projection.parquet")
            outputs = generate_figures(
                directory, figure_id="04_temporal_embedding_small_multiples",
                animations=False,
            )
            self.assertTrue(outputs["04_temporal_embedding_small_multiples"].exists())
            self.assertFalse((root / "_snapshots" / "step_000_pretrained").exists())
            figure_manifest = pd.read_parquet(root / "tables" / "figure_manifest.parquet")
            expected = {"figure_id", "filename", "source_table", "palette", "profile",
                        "entity_colors", "creation_configuration"}
            self.assertTrue(expected <= set(figure_manifest))
            report = (root / "report" / "index.html").read_text()
            self.assertIn("NER corpus-adaptation report", report)
            self.assertIn("04_temporal_embedding_small_multiples", report)

    def test_validation_failure_prevents_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = initialize_run(directory, resolve_config({"enabled": True}))
            snapshots = root / "_snapshots"
            (snapshots / "sentinel").write_text("retain")
            bad_manifest = pd.DataFrame([{
                "figure_id": "required", "filename": "figures/missing.svg",
                "source_table": "tables/missing.parquet",
            }])
            write_table(bad_manifest, root / "tables" / "figure_manifest.parquet")
            manifest = read_manifest(root)
            manifest.update({"training_completed": True, "status": "visualizations_complete",
                             "visualization_status": "complete"})
            atomic_json(manifest, root / "manifest.json")
            with self.assertRaisesRegex(ValueError, "snapshots retained"):
                finalize_run(directory)
            self.assertTrue((snapshots / "sentinel").exists())
            self.assertFalse((root / "SUCCESS").exists())


if __name__ == "__main__":
    unittest.main()
