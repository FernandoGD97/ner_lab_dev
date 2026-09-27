"""Verify evaluator-backed events and the canonical evaluation parquet artifact."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from _harness import Checks, run
from lab.ner.analysis import analyze_evaluation, evaluation_schema, load_evaluation
from lab.ner.analysis.parquet import METADATA_KEY
from lab.ner.evaluation import span_metrics

COLUMNS = ["filename", "label", "start_span", "end_span", "text"]


def spans(rows: list[list[object]], *, scored: bool = False) -> pd.DataFrame:
    columns = [*COLUMNS, "score"] if scored else COLUMNS
    return pd.DataFrame(rows, columns=columns)


def verify_events(checks: Checks, root: Path) -> None:
    gold = spans(
        [
            ["doc-1", "DISEASE", 0, 10, "myocardial"],
            ["doc-1", "DISEASE", 20, 30, "infarction"],
            ["doc-2", "DRUG", 3, 8, "aspir"],
        ]
    )
    predicted = spans(
        [
            ["doc-1", "DISEASE", 0, 10, "myocardial", 0.9],
            ["doc-1", "DISEASE", 18, 30, "xxinfarction", 0.6],
            ["doc-3", "DRUG", 1, 4, "new", 0.2],
        ],
        scored=True,
    )
    training = spans(
        [
            ["train-1", "DISEASE", 0, 10, "myocardial"],
            ["train-2", "DRUG", 5, 12, "aspirin"],
        ]
    )
    result = analyze_evaluation(
        predicted,
        gold,
        training,
        root,
        run_id="fold-1",
    )

    checks.equal("the canonical filename is fixed", result.path.name, "evaluation.parquet")
    checks.equal("strict events expose exact TP/FP/FN contributions", result.strict_counts, {"tp": 1, "fp": 2, "fn": 2})
    checks.equal("a boundary mismatch receives no partial strict credit", int(result.events["strict_is_tp"].sum()), 1)
    checks.equal("one TP preserves both sides", int((result.events["gold_exists"] & result.events["pred_exists"]).sum()), 1)
    checks.equal("prediction confidence is optional and preserved", round(float(result.events["pred_confidence"].max()), 1), 0.9)
    checks.equal("boundary mismatch receives one diagnostic relationship", int(result.events["error_pair_id"].notna().sum()), 2)

    official = span_metrics(gold, predicted, tags=["DISEASE", "DRUG"])
    checks.equal("metrics are the official evaluator's output", result.metrics, official)
    checks.equal("training gold is a first-class recorded input", result.metadata["training_gold"]["n_entities"], 2)
    checks.check("training provenance is compact", len(result.metadata["training_gold"]["sha256"]) == 64)

    reopened = load_evaluation(result.path)
    checks.equal("persisted accounting round-trips", reopened.strict_counts, result.strict_counts)
    checks.equal("official metrics live in parquet metadata", reopened.metrics, result.metrics)

    parquet = pq.ParquetFile(result.path)
    checks.equal("the artifact uses ZSTD", {codec.upper() for codec in parquet.metadata.row_group(0).column(0).compression.split()}, {"ZSTD"})
    checks.check("artifact metadata is namespaced", METADATA_KEY in parquet.schema_arrow.metadata)
    checks.equal("the physical schema is stable", parquet.schema_arrow.remove_metadata(), evaluation_schema())


def verify_absent_confidence_and_empty_inputs(checks: Checks, root: Path) -> None:
    empty = spans([])
    unscored = spans([["d", "X", 0, 2, "xx"]])
    result = analyze_evaluation(unscored, empty, empty, root, run_id="no-confidence")

    checks.equal("unscored predictions still create an FP", result.strict_counts, {"tp": 0, "fp": 1, "fn": 0})
    checks.check("absent confidence remains null", result.events["pred_confidence"].isna().all())

    gold = spans([["d", "X", 10, 20, "abcdefghij"]])
    overlap_before_exact = spans(
        [
            ["d", "X", 8, 20, "xxabcdefghij"],
            ["d", "X", 10, 20, "abcdefghij"],
        ]
    )
    ordered = analyze_evaluation(
        overlap_before_exact,
        gold,
        empty,
        root / "official-order",
        run_id="official-order",
    )
    checks.equal(
        "event accounting follows the official scorer's order-sensitive matching",
        ordered.strict_counts,
        {"tp": 0, "fp": 2, "fn": 1},
    )


def main() -> int:
    checks = Checks("analysis")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        verify_events(checks, root / "rich")
        verify_absent_confidence_and_empty_inputs(checks, root / "unscored")

    return checks.report()


if __name__ == "__main__":
    run(main)
