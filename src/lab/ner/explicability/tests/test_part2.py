from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from lab.ner.explicability.analyses.data import align_representations, deterministic_sample
from lab.ner.explicability.analyses.dynamics import (
    forgetting_events, prediction_transitions, training_dynamics,
)
from lab.ner.explicability.analyses.geometry import anisotropy_metrics, geometry_metrics, intrinsic_dimension
from lab.ner.explicability.analyses.neighborhoods import neighborhood_evolution
from lab.ner.explicability.analyses.parameters import classify_parameter, parameter_drift
from lab.ner.explicability.analyses.probing import fixed_split, fit_probe
from lab.ner.explicability.analyses.projection import shared_pca
from lab.ner.explicability.analyses.relationships import performance_relationships
from lab.ner.explicability.analyses.similarity import StreamingLinearCKA, linear_cka
from lab.ner.explicability.analyses.trajectories import centroid_dynamics, trajectory_metrics
from lab.ner.explicability.storage import write_table


class Part2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.ids = np.asarray(["a", "b", "c", "d"])
        self.base = np.asarray([[0, 0], [0, 1], [4, 4], [4, 5]], dtype=np.float64)
        self.middle = self.base + np.asarray([[1, 0], [1, 0], [-1, 0], [-1, 0]])
        self.final = self.middle + np.asarray([[0.5, 0], [0.5, 0], [-0.5, 0], [-0.5, 0]])
        self.labels = np.asarray(["O", "O", "B-X", "B-X"])

    def test_shared_projection_alignment_and_stable_matching(self) -> None:
        shuffled = np.asarray(["d", "b", "a", "c"])
        common, left, right = align_representations(
            self.ids, self.base, shuffled, self.final[[3, 1, 0, 2]]
        )
        self.assertEqual(common.tolist(), self.ids.tolist())
        np.testing.assert_array_equal(left, self.base)
        np.testing.assert_array_equal(right, self.final)
        projection, model = shared_pca({"pretrained": (self.ids, self.base), "final": (self.ids, self.final)})
        self.assertEqual(len(projection), 8)
        self.assertEqual(set(projection["projection_method"]), {"shared_pca"})
        self.assertEqual(model["basis"].shape, (2, 2))

    def test_trajectory_calculations(self) -> None:
        frame = trajectory_metrics(
            ["pretrained", "middle", "final"], self.ids,
            [self.base, self.middle, self.final],
        )
        final = frame[frame["checkpoint"] == "final"]
        np.testing.assert_allclose(final["cumulative_path_length"], 1.5)
        np.testing.assert_allclose(final["net_displacement"], 1.5)
        np.testing.assert_allclose(final["path_displacement_ratio"], 1.0)

    def test_centroid_dynamics(self) -> None:
        frame, pairs = centroid_dynamics(
            ["pretrained", "middle", "final"],
            [self.base, self.middle, self.final], self.labels, minimum_group_size=2,
        )
        self.assertEqual(len(frame), 6)
        self.assertEqual(len(pairs), 3)
        final = frame[(frame["checkpoint"] == "final") & (frame["class_label"] == "O")].iloc[0]
        self.assertAlmostEqual(final["displacement_pretrained"], 1.5)
        self.assertEqual(final["nearest_competing_class"], "B-X")

    def test_cka_sanity_and_streaming(self) -> None:
        transformed = self.base @ np.asarray([[0, -2], [2, 0]])
        self.assertAlmostEqual(linear_cka(self.base, transformed), 1.0)
        accumulator = StreamingLinearCKA(2, 2)
        accumulator.update(self.base[:2], transformed[:2])
        accumulator.update(self.base[2:], transformed[2:])
        self.assertAlmostEqual(accumulator.score(), linear_cka(self.base, transformed))

    def test_geometry_intrinsic_dimension_and_anisotropy(self) -> None:
        rng = np.random.default_rng(4)
        values = np.vstack([rng.normal(-2, 0.1, (20, 3)), rng.normal(2, 0.1, (20, 3))])
        labels = np.asarray(["a"] * 20 + ["b"] * 20)
        metrics = geometry_metrics(values, labels, k=3, minimum_group_size=3, block_size=8)
        self.assertGreater(metrics["silhouette"], 0.8)
        self.assertGreater(metrics["knn_label_purity"], 0.9)
        estimates = {row["estimator"]: row["estimate"] for row in intrinsic_dimension(values)}
        self.assertIn("participation_ratio", estimates)
        self.assertIn("twonn", estimates)
        anisotropy = anisotropy_metrics(np.column_stack([np.arange(1, 21), np.zeros(20)]))
        self.assertAlmostEqual(anisotropy["dominant_component_fraction"], 1.0)

    def test_neighbor_overlap(self) -> None:
        detail, summary = neighborhood_evolution(
            ["pretrained", "final"], self.ids, [self.base, self.base.copy()], self.labels, k=1
        )
        self.assertEqual(len(detail), 8)
        self.assertTrue((summary["jaccard_previous"] == 1.0).all())
        self.assertTrue((summary["neighborhood_turnover"] == 0.0).all())

    def _observation_frame(self) -> pd.DataFrame:
        rows = []
        sequences = {
            "easy": [(True, "X", .7), (True, "X", .8), (True, "X", .9)],
            "forgotten": [(False, "O", .2), (True, "X", .8), (False, "O", .3)],
        }
        for observation_id, sequence in sequences.items():
            for order, (correct, predicted, confidence) in enumerate(sequence):
                rows.append({"observation_id": observation_id, "checkpoint": f"s{order}",
                             "checkpoint_order": order, "gold_confidence": confidence,
                             "confidence": confidence, "entropy": 1 - confidence,
                             "correct": correct, "gold_vs_second_best_margin": confidence - .5,
                             "predicted_label": predicted})
        return pd.DataFrame(rows)

    def test_cartography_forgetting_and_transitions(self) -> None:
        observations = self._observation_frame()
        cartography = training_dynamics(observations)
        self.assertEqual(set(cartography["observation_id"]), {"easy", "forgotten"})
        forgetting = forgetting_events(observations).set_index("observation_id")
        self.assertEqual(forgetting.loc["forgotten", "forgetting_events"], 1)
        self.assertEqual(forgetting.loc["forgotten", "first_correct_checkpoint"], "s1")
        transitions = prediction_transitions(observations)
        self.assertEqual(int(transitions["transition_count"].sum()), 4)
        self.assertAlmostEqual(
            transitions.groupby(["checkpoint_from", "checkpoint_to", "label_from"])["transition_probability"].sum().iloc[0], 1.0
        )

    def test_probe_and_deterministic_seed(self) -> None:
        ids = np.asarray([f"id-{index}" for index in range(200)])
        targets = np.asarray(["left"] * 100 + ["right"] * 100)
        values = np.column_stack([np.r_[-np.ones(100), np.ones(100)], np.zeros(200)])
        first = fixed_split(ids, 9, .7, .15)
        np.testing.assert_array_equal(first, fixed_split(ids, 9, .7, .15))
        metrics = fit_probe(values, targets, first, iterations=100, learning_rate=.2)
        self.assertGreater(metrics["f1"], .95)
        np.testing.assert_array_equal(
            deterministic_sample(ids, 20, 12, targets),
            deterministic_sample(ids, 20, 12, targets),
        )

    def test_parameter_drift(self) -> None:
        baseline = {"encoder.layer.0.attention.query.weight": np.ones((2, 2)),
                    "classifier.weight": np.ones((1, 2))}
        current = {name: values + 1 for name, values in baseline.items()}
        frame = parameter_drift(baseline, current, "final")
        self.assertEqual(set(frame["component"]), {"attention", "classification_head"})
        self.assertTrue((frame["delta_norm"] > 0).all())
        self.assertEqual(classify_parameter("encoder.layer.3.output.LayerNorm.weight"), ("layer_03", "normalization"))

    def test_performance_join_and_parquet_schema(self) -> None:
        performance = pd.DataFrame({"checkpoint": ["a", "b", "c"], "f1": [.1, .5, .9]})
        internal = pd.DataFrame({"checkpoint": ["a", "b", "c"], "drift": [.0, .4, .8]})
        joined, correlations = performance_relationships(performance, internal)
        self.assertEqual(len(joined), 3)
        self.assertAlmostEqual(correlations.iloc[0]["correlation"], 1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = write_table(correlations, Path(directory) / "relationships.parquet")
            schema = pq.ParquetFile(path).schema_arrow
            self.assertIn("correlation", schema.names)
            self.assertEqual(pq.ParquetFile(path).metadata.row_group(0).column(0).compression, "ZSTD")


if __name__ == "__main__":
    unittest.main()
