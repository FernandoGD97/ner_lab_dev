"""Verify diagnostic oracles, document bootstrap, and scientific SVG reporting."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from _harness import Checks, run
from lab.ner.analysis import (
    AnalysisConfig,
    BootstrapConfig,
    PublicationConfig,
    analyze_evaluation,
    bootstrap_intervals,
    create_publication_report,
    oracle_error_budget,
    paired_seen_zero_shot_difference,
)
from lab.ner.analysis.plotting import OKABE_ITO, PlotTheme, SIZE_PRESETS, SVGFigure

COLUMNS = ["filename", "label", "start_span", "end_span", "text"]


def frame(rows, scored=False):
    return pd.DataFrame(rows, columns=[*COLUMNS, "score"] if scored else COLUMNS)


def oracle_events() -> pd.DataFrame:
    rows = []
    event_id = 0

    def add(outcome, primary, *, pair=None, role=None, label="D", pred_label="D", exact=True,
            normalized=True, zero=False, confidence=None):
        nonlocal event_id
        rows.append({
            "event_id": event_id, "document_id": f"d{event_id % 2}",
            "strict_is_tp": outcome == "TP", "strict_is_fp": outcome == "FP", "strict_is_fn": outcome == "FN",
            "error_primary": primary, "error_pair_id": pair, "error_pair_role": role,
            "gold_exists": outcome != "FP", "pred_exists": outcome != "FN",
            "gold_label": label if outcome != "FP" else None,
            "pred_label": pred_label if outcome != "FN" else None,
            "diagnostic_pred_id": pair if pair is not None else None,
            "train_exact_seen": exact, "train_normalized_seen": normalized,
            "zero_shot_fuzzy": zero, "zero_shot_default": zero,
            "pred_confidence": confidence,
        })
        event_id += 1

    add("TP", "CORRECT")
    for pair, primary, gold_label, pred_label in (
        (0, "BOUNDARY_ERROR", "D", "D"),
        (1, "LABEL_ERROR", "D", "X"),
        (2, "BOUNDARY_AND_LABEL_ERROR", "X", "D"),
    ):
        add("FN", primary, pair=pair, role="GOLD", label=gold_label, pred_label=pred_label)
        add("FP", primary, pair=pair, role="PRED", label=gold_label, pred_label=pred_label, confidence=.8)
    add("FN", "MISSED", exact=True, normalized=True, zero=False)
    add("FN", "MISSED", exact=False, normalized=False, zero=True, label="X")
    add("FP", "SPURIOUS", pred_label="X", confidence=.95)
    return pd.DataFrame(rows)


def verify_oracles(checks: Checks) -> None:
    budget = oracle_error_budget(oracle_events()).set_index("error_family")
    checks.equal("baseline official contributions are retained", tuple(budget.loc["Boundary correction", ["current_tp", "current_fp", "current_fn"]]), (1, 4, 5))
    checks.equal("boundary oracle repairs one FP/FN pair", tuple(budget.loc["Boundary correction", ["oracle_tp", "oracle_fp", "oracle_fn"]]), (2, 3, 4))
    checks.equal("label oracle repairs one pair", tuple(budget.loc["Label correction", ["oracle_tp", "oracle_fp", "oracle_fn"]]), (2, 3, 4))
    checks.equal("boundary-plus-label oracle remains separate", tuple(budget.loc["Boundary + label correction", ["oracle_tp", "oracle_fp", "oracle_fn"]]), (2, 3, 4))
    checks.equal("all-miss oracle recovers only true misses", tuple(budget.loc["All missed entities", ["oracle_tp", "oracle_fp", "oracle_fn"]]), (3, 4, 3))
    checks.equal("exact-seen miss oracle is available", int(budget.loc["Exact-seen miss recovery", "affected_events"]), 1)
    checks.equal("fuzzy-seen miss oracle is available", int(budget.loc["Fuzzy-seen miss recovery", "affected_events"]), 1)
    checks.equal("zero-shot miss oracle is available", int(budget.loc["Fuzzy-zero-shot miss recovery", "affected_events"]), 1)
    checks.equal("spurious oracle removes only true spurious prediction", tuple(budget.loc["Spurious-prediction removal", ["oracle_tp", "oracle_fp", "oracle_fn"]]), (1, 3, 5))
    checks.equal("combined oracle corrects all diagnostic errors", tuple(budget.loc["All diagnostic errors corrected", ["oracle_tp", "oracle_fp", "oracle_fn"]]), (6, 0, 0))
    checks.check("confusion-pair oracle is emitted", "Confusion: D -> X" in budget.index)
    individual = budget.loc["Boundary correction", "delta_f1"] + budget.loc["Label correction", "delta_f1"]
    combined = budget.loc["All boundary-related errors fixed", "delta_f1"]
    checks.check("oracle gains are demonstrably non-additive", individual != combined)
    checks.check("every oracle reports full metrics and delta", {"current_precision", "oracle_precision", "delta_f1", "relative_error_reduction"}.issubset(budget.columns))


def analysis_artifact(root: Path):
    training = frame([["t", "D", 0, 5, "alpha"], ["t", "X", 6, 11, "theta"]])
    gold = frame([
        ["d1", "D", 0, 5, "alpha"], ["d1", "X", 10, 14, "beta"],
        ["d2", "D", 0, 5, "gamma"], ["d2", "X", 10, 15, "theta"],
    ])
    predicted = frame([
        ["d1", "D", 0, 5, "alpha", .95], ["d1", "D", 10, 14, "beta", .85],
        ["d2", "D", 0, 4, "gamm", .7], ["d2", "X", 20, 25, "extra", .92],
    ], scored=True)
    return analyze_evaluation(
        predicted, gold, training, root, run_id="publication",
        config=AnalysisConfig(confidence_thresholds=(.5, .8, .9)),
    )


def verify_bootstrap(checks: Checks, analysis) -> None:
    config = BootstrapConfig(samples=30, seed=17)
    first = bootstrap_intervals(analysis.events, config)
    second = bootstrap_intervals(analysis.events, config)
    checks.frames_equal("fixed-seed document bootstrap is deterministic", first, second)
    valid = first[first["status"] == "ok"]
    checks.check("bootstrap CI bounds are ordered", bool((valid["ci_low"] <= valid["ci_high"]).all()))
    overall = first[(first["family"] == "overall") & (first["metric"] == "strict_f1")].iloc[0]
    checks.equal("bootstrap resamples the observed canonical baseline", overall.observed, analysis.metrics["span_strict_f1"])
    paired = paired_seen_zero_shot_difference(analysis.events, config)
    checks.check("paired seen-zero-shot effects include uncertainty", paired["ci_low"].notna().all())
    single = analysis.events[analysis.events["document_id"] == "d1"]
    tiny = bootstrap_intervals(single, config)
    checks.equal("single-document uncertainty is gracefully disabled", set(tiny["status"]), {"insufficient_documents"})
    empty = bootstrap_intervals(analysis.events.iloc[0:0], config)
    checks.equal("empty input is gracefully handled", set(empty["status"]), {"insufficient_documents"})


def verify_figures(checks: Checks, analysis, root: Path) -> None:
    report = create_publication_report(
        analysis.path, root / "paper", config=PublicationConfig(bootstrap=BootstrapConfig(samples=20, seed=5))
    )
    expected = {
        "generalization_strict_recall", "generalization_strict_f1", "generalization_chrf_f1",
        "seen_vs_zeroshot_by_label", "error_landscape_counts", "error_landscape_normalized",
        "oracle_error_budget", "label_confusion_counts", "label_confusion_normalized",
        "rapidfuzz_vs_strict_f1", "training_frequency_vs_strict_f1", "boundary_start_delta",
        "boundary_end_delta", "entity_length_vs_performance", "confidence_by_error_type",
        "confidence_threshold_sweep", "error_intersections", "error_pareto",
    }
    checks.check("all main deterministic SVG figures are generated", expected.issubset(report.figures))
    checks.check("every figure is a non-empty SVG", all(path.suffix == ".svg" and path.stat().st_size > 200 for path in report.figures.values()))
    sample = report.figures["generalization_strict_f1"].read_text(encoding="utf-8")
    checks.check("SVG output is valid and text remains editable", sample.startswith("<?xml") and "<svg" in sample and "<text" in sample)
    checks.check("Okabe-Ito color appears in SVG", OKABE_ITO[0] in sample)
    theme = PlotTheme()
    checks.equal("style has no default grid", theme.default_grid, False)
    checks.equal("top and right spines are disabled", (theme.top_spine, theme.right_spine), (False, False))
    checks.equal("SVG text is not converted to paths", theme.svg_fonttype, "none")
    checks.equal("raster fallback targets 2100 DPI", theme.savefig_dpi, 2100)
    figure = SVGFigure("size test", "single-column")
    checks.equal("physical figure preset is respected", (figure.width_in, figure.height_in), SIZE_PRESETS["single-column"])
    checks.check("publication tables are Parquet", all(path.suffix == ".parquet" for path in report.tables.values()))
    checks.check("captions and findings are generated", report.captions.exists() and report.findings.exists())
    checks.check("manifest records reproducibility", "plotting_theme_version" in report.manifest.read_text(encoding="utf-8"))


def main() -> int:
    checks = Checks("analysis publication")
    verify_oracles(checks)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        analysis = analysis_artifact(root / "analysis")
        verify_bootstrap(checks, analysis)
        verify_figures(checks, analysis, root)
    return checks.report()


if __name__ == "__main__":
    run(main)
