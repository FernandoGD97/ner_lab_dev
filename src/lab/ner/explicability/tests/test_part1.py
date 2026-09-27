from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from lab.ner.explicability.config import SnapshotConfig, resolve_config
from lab.ner.explicability.finalize import (
    finalize_run, initialize_run, read_manifest, update_training_manifest, validate_manifest,
)
from lab.ner.explicability.metadata import entity_id
from lab.ner.explicability.pooling import pool_groups, pool_spans
from lab.ner.explicability.snapshot import check_disk_space, estimate_snapshot_bytes, resolve_layers
from lab.ner.explicability.storage import ChunkedArrayWriter, iter_chunks, write_table
from lab.ner.explicability.utils import stable_id


class Part1Tests(unittest.TestCase):
    def test_disabled_and_snapshot_triggers(self) -> None:
        self.assertFalse(resolve_config(None).enabled)
        self.assertTrue(resolve_config({"enabled": True, "snapshots": {"trigger": "evaluation"}}).enabled)
        self.assertEqual(resolve_config({"snapshots": {"trigger": "epoch"}}).snapshots.trigger, "epoch")
        with self.assertRaises(ValueError):
            resolve_config({"snapshots": {"trigger": "batch"}})

    def test_scopes_dtypes_and_complete_hidden_dimension(self) -> None:
        self.assertEqual(resolve_layers(SnapshotConfig(), 13), [12])
        self.assertEqual(resolve_layers(SnapshotConfig(embedding_scope="selected_layers", selected_layers=(0, -1)), 13), [0, 12])
        self.assertEqual(resolve_layers(SnapshotConfig(embedding_scope="all_hidden_states"), 4), [0, 1, 2, 3])
        self.assertEqual(estimate_snapshot_bytes(10, 768, 2, "float16", 0), 10 * 768 * 2 * 2)
        self.assertEqual(estimate_snapshot_bytes(10, 1024, 1, "float32", 0), 10 * 1024 * 4)

    def test_stable_ids_and_pooling_are_deterministic(self) -> None:
        self.assertEqual(stable_id("word", "d", 1), stable_id("word", "d", 1))
        self.assertNotEqual(entity_id("gold", "d", 1, 4, "X"), entity_id("predicted", "d", 1, 4, "X"))
        vectors = np.asarray([[1, 2], [3, 4], [10, 20]], dtype=np.float32)
        first, ids = pool_groups(vectors, ["a", "a", "b"], "first")
        mean, _ = pool_groups(vectors, ["a", "a", "b"], "mean")
        maximum, _ = pool_groups(vectors, ["a", "a", "b"], "max")
        self.assertEqual(ids, ["a", "b"])
        np.testing.assert_array_equal(first, [[1, 2], [10, 20]])
        np.testing.assert_array_equal(mean[0], [2, 3])
        np.testing.assert_array_equal(maximum[0], [3, 4])
        np.testing.assert_array_equal(pool_spans(vectors, [(0, 2), (2, 4), (5, 6)], [(0, 4)])[0], [2, 3])

    def test_chunked_write_dtypes_and_interruption(self) -> None:
        for dtype in ("float16", "float32"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                writer = ChunkedArrayWriter(root / "complete", dtype, 3, [0, 2])
                writer.write(np.ones((2, 2, 3)), ["a", "b"])
                writer.write(np.zeros((2, 1, 3)), ["c"])
                chunks = list(iter_chunks(writer.finalize()))
                self.assertEqual(sum(chunk[0].shape[1] for chunk in chunks), 3)
                self.assertEqual(chunks[0][0].dtype, np.dtype(dtype))
                interrupted = ChunkedArrayWriter(root / "broken", dtype, 3, [0])
                interrupted.write(np.ones((1, 1, 3)), ["a"])
                with self.assertRaisesRegex(ValueError, "Incomplete"):
                    list(iter_chunks(interrupted.incomplete))

    @staticmethod
    def _snapshot(root: Path) -> dict:
        path = root / "_snapshots" / "step_000_pretrained"
        path.mkdir(parents=True)
        (path / "COMPLETE").write_text("")
        return {"snapshot_id": "step_000_pretrained", "complete": True, "hidden_size": 3,
                "layers": [1], "dtype": "float16", "observations": 2, "examples": 1}

    def test_manifest_failure_preserves_and_success_cleans(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            root = initialize_run(run, resolve_config({"enabled": True}), model="tiny", dataset="synthetic")
            snapshot = self._snapshot(root)
            update_training_manifest(root, [snapshot], succeeded=False)
            with self.assertRaisesRegex(ValueError, "training"):
                finalize_run(run)
            self.assertTrue((root / "_snapshots").exists())
            update_training_manifest(root, [snapshot], succeeded=True)
            table = write_table(pd.DataFrame({"id": [2, 1]}), root / "tables" / "result.parquet", sort_by=["id"])
            manifest = finalize_run(run, completed_analyses=["part1"], mandatory_analyses=["part1"], required_outputs=["tables/result.parquet"])
            self.assertTrue(table.exists() and (root / "SUCCESS").exists())
            self.assertEqual(manifest["cleanup_state"], "removed")
            self.assertFalse((root / "_snapshots").exists())
            self.assertFalse(validate_manifest(read_manifest(root)))

    def test_cleanup_never_and_pending_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            root = initialize_run(run, resolve_config({"enabled": True, "snapshots": {"cleanup_policy": "never"}}))
            snapshot = self._snapshot(root)
            update_training_manifest(root, [snapshot], succeeded=True)
            with self.assertRaisesRegex(ValueError, "pending"):
                finalize_run(run, mandatory_analyses=["part2"])
            self.assertTrue((root / "_snapshots").exists())
            self.assertEqual(finalize_run(run)["cleanup_state"], "retained")

    def test_disk_protection_and_rank_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "lab.ner.explicability.snapshot.available_bytes", return_value=100
        ):
            with self.assertRaisesRegex(OSError, "Insufficient disk"):
                check_disk_space(directory, expected_bytes=80, safety_margin_bytes=30)
        with patch.dict("os.environ", {"RANK": "1", "WORLD_SIZE": "3"}, clear=False):
            from lab.ner.explicability.utils import distributed_context
            self.assertEqual(distributed_context(), (1, 3))
            self.assertEqual(list(range(1, 10, 3)), [1, 4, 7])


if __name__ == "__main__":
    unittest.main()
