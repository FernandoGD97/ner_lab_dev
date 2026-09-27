"""End-to-end research reporting from canonical ``evaluation.parquet``."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from lab import __version__
from lab.ner.analysis.api import load_evaluation
from lab.ner.analysis.oracles import oracle_error_budget
from lab.ner.analysis.plotting import (
    THEME_VERSION,
    category_strip_figure,
    categorical_metric_figure,
    grouped_bar_figure,
    heatmap_figure,
    histogram_figure,
    horizontal_bar_figure,
)
from lab.ner.analysis.statistics import (
    BootstrapConfig,
    bootstrap_intervals,
    paired_seen_zero_shot_difference,
)


@dataclass(frozen=True)
class PublicationConfig:
    bootstrap: BootstrapConfig = BootstrapConfig()
    minimum_label_support: int = 1
    minimum_confusion_support: int = 1
    boundary_display_quantile: float = 0.99


@dataclass(frozen=True)
class PublicationReport:
    output_dir: Path
    figures: dict[str, Path]
    tables: dict[str, Path]
    captions: Path
    findings: Path
    manifest: Path


def create_publication_report(
    evaluation_parquet: str | Path,
    output_dir: str | Path | None = None,
    *,
    config: PublicationConfig | None = None,
) -> PublicationReport:
    """Create tables, uncertainty, captions, findings, and scientific SVG figures."""
    analysis = load_evaluation(evaluation_parquet)
    config = config or PublicationConfig()
    output = Path(output_dir) if output_dir is not None else Path(evaluation_parquet).parent
    figures_dir, tables_dir = output / "figures", output / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    subgroups = analysis.subgroup_metrics()
    labels = pd.DataFrame(analysis.metadata.get("aggregates", {}).get("labels", []))
    confusion = pd.DataFrame(
        analysis.metadata.get("aggregates", {}).get("confusion", {}).get("exact_boundary_label_confusion", [])
    )
    oracles = oracle_error_budget(analysis.events, config.minimum_confusion_support)
    bootstrap = bootstrap_intervals(analysis.events, config.bootstrap)
    paired = paired_seen_zero_shot_difference(analysis.events, config.bootstrap)

    table_frames = {
        "overall_metrics": pd.DataFrame([analysis.metrics]),
        "metrics_by_exposure": subgroups[subgroups["family"].isin(["exact_exposure", "normalized_exposure", "fuzzy_exposure", "zero_shot_default"])],
        "metrics_by_similarity": subgroups[subgroups["family"] == "similarity"],
        "metrics_by_label": labels,
        "major_label_confusions": confusion,
        "oracle_error_budget": oracles,
        "bootstrap_intervals": bootstrap,
        "paired_comparisons": paired,
    }
    tables = {}
    for name, frame in table_frames.items():
        path = tables_dir / f"{name}.parquet"
        frame.reset_index(drop=True).to_parquet(path, compression="zstd", index=False)
        tables[name] = path

    captions_text = _captions(analysis.metadata, config.bootstrap)
    captions_path = output / "captions.md"
    captions_path.write_text(_render_captions(captions_text), encoding="utf-8")
    figures = _figures(analysis, subgroups, labels, confusion, oracles, bootstrap, figures_dir,
                       boundary_display_quantile=config.boundary_display_quantile,
                       minimum_label_support=config.minimum_label_support)
    findings_path = output / "findings.md"
    findings_path.write_text(_findings(analysis, oracles, paired), encoding="utf-8")
    manifest_path = output / "report_manifest.json"
    manifest = {
        "analysis_schema_version": analysis.metadata.get("schema_version"),
        "lab_version": __version__,
        "git_commit": _git_commit(),
        "model_id": analysis.metadata.get("model_id"),
        "dataset": analysis.metadata.get("dataset"),
        "split": analysis.metadata.get("split"),
        "analysis_config": analysis.metadata.get("analysis_config", {}),
        "bootstrap": config.bootstrap.to_dict(),
        "publication": {**asdict(config), "bootstrap": config.bootstrap.to_dict()},
        "plotting_theme_version": THEME_VERSION,
        "canonical_source": str(Path(evaluation_parquet).resolve()),
        "svg_is_resolution_independent": True,
        "rasterized_artist_dpi": 2100,
        "figures": sorted(path.name for path in figures.values()),
        "tables": sorted(path.name for path in tables.values()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return PublicationReport(output, figures, tables, captions_path, findings_path, manifest_path)


def _figures(
    analysis, subgroups, labels, confusion, oracles, bootstrap, directory, *,
    boundary_display_quantile, minimum_label_support,
):
    paths: dict[str, Path] = {}
    def target(name):
        path = directory / name
        paths[name.removesuffix(".svg")] = path
        return path

    similarity = subgroups[subgroups["family"] == "similarity"].copy()
    similarity = _stable_order(similarity, "value", _similarity_order(analysis.metadata))
    threshold_index = next((i for i, value in enumerate(similarity["value"]) if str(value).startswith("<")), None)
    for metric, filename, label in (
        ("strict_recall", "generalization_strict_recall.svg", "STRICT recall"),
        ("strict_f1", "generalization_strict_f1.svg", "STRICT F1"),
        ("char_f1", "generalization_chrf_f1.svg", "Character F1"),
    ):
        plotted = _with_intervals(similarity, bootstrap, "similarity", metric)
        categorical_metric_figure(plotted, category="value", metric=metric,
                                  title=f"Generalization by lexical similarity — {label}", ylabel=label,
                                  path=target(filename), threshold_position=threshold_index)
    categorical_metric_figure(_with_intervals(similarity, bootstrap, "similarity", "strict_f1"), category="value", metric="strict_f1",
                              title="RapidFuzz similarity versus STRICT F1", ylabel="STRICT F1",
                              path=target("rapidfuzz_vs_strict_f1.svg"), threshold_position=threshold_index)

    by_label = subgroups[subgroups["family"] == "label_zero_shot"].copy()
    if not labels.empty:
        eligible = set(labels.loc[labels["evaluation_support"] >= minimum_label_support, "label"].astype(str))
        by_label = by_label[by_label["label"].astype(str).isin(eligible)]
    for metric, name, ylabel in (
        ("strict_f1", "seen_vs_zeroshot_by_label.svg", "STRICT F1"),
        ("strict_precision", "seen_vs_zeroshot_by_label_precision.svg", "STRICT precision"),
        ("strict_recall", "seen_vs_zeroshot_by_label_recall.svg", "STRICT recall"),
        ("char_f1", "seen_vs_zeroshot_by_label_chrf_f1.svg", "Character F1"),
    ):
        grouped_bar_figure(by_label, category="label", series="value", metric=metric,
                           title=f"Seen versus zero-shot {ylabel} by label", ylabel=ylabel,
                           path=target(name))

    landscape = _error_landscape(analysis.events)
    heatmap_figure(landscape, title="Error landscape — counts", path=target("error_landscape_counts.svg"))
    normalized = landscape.div(landscape.sum(axis=1).replace(0, 1), axis=0)
    heatmap_figure(normalized, title="Error landscape — row-normalized", path=target("error_landscape_normalized.svg"), normalized=True)

    budget = oracles[oracles["scenario_kind"] == "single_family"]
    horizontal_bar_figure(budget, label="error_family", value="delta_f1",
                          title="STRICT diagnostic oracle upper bounds", xlabel="Absolute ΔF1",
                          path=target("oracle_error_budget.svg"))
    confusion_matrix = _confusion_matrix(confusion)
    heatmap_figure(confusion_matrix, title="Exact-boundary label confusion — counts",
                   path=target("label_confusion_counts.svg"))
    normalized_confusion = confusion_matrix.div(confusion_matrix.sum(axis=1).replace(0, 1), axis=0)
    heatmap_figure(normalized_confusion, title="Exact-boundary label confusion — row-normalized",
                   path=target("label_confusion_normalized.svg"), normalized=True)

    frequency = _stable_order(subgroups[subgroups["family"] == "training_frequency"], "value", ["0", "1", "2-5", "6-10", "11-20", ">20"])
    categorical_metric_figure(frequency, category="value", metric="strict_f1",
                              title="Training frequency versus STRICT F1", ylabel="STRICT F1",
                              path=target("training_frequency_vs_strict_f1.svg"))
    gold_errors = analysis.events[(analysis.events["error_pair_role"] == "GOLD") & analysis.events["span_boundary_error"].notna()]
    histogram_figure(gold_errors["span_start_delta"], title="Start-boundary deviation", xlabel="Prediction start − gold start (characters)", path=target("boundary_start_delta.svg"), display_quantile=boundary_display_quantile)
    histogram_figure(gold_errors["span_end_delta"], title="End-boundary deviation", xlabel="Prediction end − gold end (characters)", path=target("boundary_end_delta.svg"), display_quantile=boundary_display_quantile)
    absolute = pd.concat([gold_errors["span_abs_start_delta"], gold_errors["span_abs_end_delta"]], ignore_index=True)
    histogram_figure(absolute, title="Absolute boundary deviation", xlabel="Absolute deviation (characters)", path=target("boundary_absolute_delta.svg"), display_quantile=boundary_display_quantile)
    direction = pd.crosstab(gold_errors["gold_label"], gold_errors["span_boundary_error"])
    heatmap_figure(direction, title="Boundary direction by label", path=target("boundary_direction_by_label.svg"))

    for family, filename, title in (
        ("word_length", "entity_length_vs_performance.svg", "Entity length versus STRICT F1"),
        ("tokenization", "tokenization_vs_performance.svg", "Tokenization fragmentation versus STRICT F1"),
    ):
        selected = subgroups[subgroups["family"] == family]
        categorical_metric_figure(selected, category="value", metric="strict_f1", title=title,
                                  ylabel="STRICT F1", path=target(filename))
    confidence = analysis.events[analysis.events["pred_confidence"].notna()]
    _confidence_figure(confidence, target("confidence_by_error_type.svg"))
    sweep = subgroups[subgroups["family"] == "confidence_threshold"]
    categorical_metric_figure(sweep, category="value", metric="strict_f1",
                              title="Confidence threshold sweep", ylabel="STRICT F1",
                              path=target("confidence_threshold_sweep.svg"))
    if not sweep.empty:
        sweep = sweep.copy()
        denominator = max(1, int(sweep["retained_predictions"].max()))
        sweep["coverage"] = sweep["retained_predictions"] / denominator
    categorical_metric_figure(sweep, category="value", metric="coverage",
                              title="Prediction coverage versus confidence threshold", ylabel="Retained-prediction coverage",
                              path=target("confidence_coverage_vs_threshold.svg"))
    intersections = pd.DataFrame(analysis.metadata.get("aggregates", {}).get("error_intersections", []))
    if not intersections.empty:
        intersections["combination"] = intersections["feature"] + " + " + intersections["error"]
    horizontal_bar_figure(intersections, label="combination", value="support",
                          title="Supported error intersections", xlabel="Events",
                          path=target("error_intersections.svg"))
    pareto = pd.DataFrame(analysis.metadata.get("aggregates", {}).get("error_concentration", []))
    selected_pareto = pareto[pareto["dimension"] == "surface"].copy() if not pareto.empty else pareto
    if not selected_pareto.empty:
        selected_pareto = selected_pareto.sort_values("rank")
        selected_pareto["cumulative_proportion"] = selected_pareto["proportion"].cumsum().clip(upper=1)
    categorical_metric_figure(selected_pareto, category="rank", metric="cumulative_proportion",
                              title="Cumulative error concentration by surface", ylabel="Cumulative proportion",
                              path=target("error_pareto.svg"), support="count")
    _dataset_shift_figure(analysis.metadata, target("dataset_shift_labels.svg"))
    _dataset_characteristics_figure(analysis.metadata, target("dataset_shift_characteristics.svg"))
    return paths


def _confidence_figure(events, path):
    category_strip_figure(events, category="error_primary", value="pred_confidence",
                          title="Prediction-score distribution by outcome", xlabel="Prediction score", path=path)


def _dataset_shift_figure(metadata, path):
    shift = metadata.get("aggregates", {}).get("dataset_shift", {})
    rows = []
    for source in ("training", "evaluation"):
        for label, count in shift.get(source, {}).get("label_distribution", {}).items():
            rows.append({"label": label, "source": source, "proportion": count / max(1, shift[source]["support"])})
    grouped_bar_figure(pd.DataFrame(rows), category="label", series="source", metric="proportion",
                       title="Training/evaluation label distributions", ylabel="Proportion",
                       path=path)


def _dataset_characteristics_figure(metadata, path):
    shift = metadata.get("aggregates", {}).get("dataset_shift", {})
    rows = []
    for source in ("training", "evaluation"):
        values = shift.get(source, {})
        for characteristic in ("acronym_prevalence", "digit_prevalence", "punctuation_prevalence"):
            rows.append({"characteristic": characteristic.replace("_prevalence", ""), "source": source,
                         "proportion": values.get(characteristic, 0.0)})
    grouped_bar_figure(pd.DataFrame(rows), category="characteristic", series="source", metric="proportion",
                       title="Training/evaluation entity characteristics", ylabel="Proportion",
                       path=path)


def _error_landscape(events):
    gold = events[events["gold_exists"]].copy()
    gold["exposure"] = gold.apply(_exposure_row, axis=1)
    mapping = {"CORRECT": "Correct", "MISSED": "Missed", "BOUNDARY_ERROR": "Boundary",
               "LABEL_ERROR": "Wrong label", "BOUNDARY_AND_LABEL_ERROR": "Boundary + label"}
    gold["outcome"] = gold["error_primary"].map(mapping)
    order = ["Exact seen", "Normalized-only seen", "Fuzzy seen", "Zero-shot"]
    columns = ["Correct", "Missed", "Boundary", "Wrong label", "Boundary + label"]
    return pd.crosstab(gold["exposure"], gold["outcome"]).reindex(index=order, columns=columns, fill_value=0)


def _exposure_row(row):
    if row.train_exact_seen:
        return "Exact seen"
    if row.train_normalized_seen:
        return "Normalized-only seen"
    if not row.zero_shot_fuzzy:
        return "Fuzzy seen"
    return "Zero-shot"


def _confusion_matrix(confusion):
    if confusion.empty:
        return pd.DataFrame([[0]], index=["No gold label"], columns=["No predicted label"])
    return confusion.pivot_table(index="gold_label", columns="pred_label", values="count", aggfunc="sum", fill_value=0)


def _stable_order(frame, column, order):
    if frame.empty:
        return frame
    positions = {str(value): index for index, value in enumerate(order)}
    return frame.assign(_order=frame[column].astype(str).map(positions).fillna(len(positions))).sort_values("_order").drop(columns="_order")


def _with_intervals(frame, bootstrap, family, metric):
    intervals = bootstrap[(bootstrap["family"] == family) & (bootstrap["metric"] == metric)][
        ["value", "ci_low", "ci_high"]
    ]
    return frame.merge(intervals, on="value", how="left")


def _similarity_order(metadata):
    return [entry[0] for entry in metadata.get("analysis_config", {}).get("similarity_bins", [])]


def _captions(metadata, bootstrap):
    threshold = metadata.get("analysis_config", {}).get("fuzzy_threshold", 0.80)
    ci = (f" Error bars, where shown, are {bootstrap.confidence_level:.0%} document-level bootstrap intervals from {bootstrap.samples} samples."
          if bootstrap.enabled else " Bootstrap intervals were disabled.")
    return {
        "generalization_strict_recall.svg": f"STRICT recall across nearest-training lexical-similarity strata. The configured fuzzy zero-shot threshold is {threshold:.2f}.{ci}",
        "generalization_strict_f1.svg": f"STRICT F1 across nearest-training lexical-similarity strata; support is shown at each point. The configured fuzzy threshold is {threshold:.2f}.{ci}",
        "generalization_chrf_f1.svg": f"Character-level F1 from the existing evaluator across lexical-similarity strata. The configured fuzzy threshold is {threshold:.2f}.{ci}",
        "seen_vs_zeroshot_by_label.svg": f"Observed STRICT F1 for fuzzy-seen and fuzzy-zero-shot mentions by entity label (threshold {threshold:.2f}).",
        "seen_vs_zeroshot_by_label_precision.svg": f"Observed STRICT precision for fuzzy-seen and fuzzy-zero-shot mentions by entity label (threshold {threshold:.2f}).",
        "seen_vs_zeroshot_by_label_recall.svg": f"Observed STRICT recall for fuzzy-seen and fuzzy-zero-shot mentions by entity label (threshold {threshold:.2f}).",
        "seen_vs_zeroshot_by_label_chrf_f1.svg": f"Observed character-level F1 for fuzzy-seen and fuzzy-zero-shot mentions by entity label (threshold {threshold:.2f}).",
        "error_landscape_counts.svg": "Counts of gold entities by training-exposure class and diagnostic outcome. Diagnostic overlap does not confer partial evaluation credit.",
        "error_landscape_normalized.svg": "Row-normalized diagnostic outcome composition within each training-exposure class.",
        "oracle_error_budget.svg": "STRICT diagnostic oracle upper bounds. Bars are counterfactual ceilings, not predicted achievable gains; individual gains are not additive.",
        "label_confusion_counts.svg": "Raw exact-boundary, wrong-label diagnostic associations. Boundary errors and missed/spurious pseudo-labels are excluded.",
        "label_confusion_normalized.svg": "Row-normalized exact-boundary label confusion, conditional on the gold label.",
        "rapidfuzz_vs_strict_f1.svg": f"STRICT F1 across binned normalized-mention RapidFuzz similarity; the zero-shot threshold is {threshold:.2f}.",
        "training_frequency_vs_strict_f1.svg": "STRICT F1 across configured exact training-mention frequency groups; support is annotated.",
        "boundary_start_delta.svg": "Distribution of prediction-start minus gold-start offsets for diagnostically associated boundary errors.",
        "boundary_end_delta.svg": "Distribution of prediction-end minus gold-end offsets for diagnostically associated boundary errors.",
        "boundary_absolute_delta.svg": "Distribution of absolute start/end deviations for diagnostically associated boundary errors; any display clipping is stated in the figure without altering source data.",
        "boundary_direction_by_label.svg": "Counts of left-, right-, and both-boundary diagnostic categories by gold entity label.",
        "boundary_absolute_delta.svg": "Absolute start and end deviations for diagnostically associated boundary errors.",
        "boundary_direction_by_label.svg": "Counts of left-, right-, and both-boundary diagnostic errors by gold label.",
        "entity_length_vs_performance.svg": "STRICT F1 across whitespace-token entity-length groups.",
        "tokenization_vs_performance.svg": "STRICT F1 across optional model-subtoken fragmentation groups.",
        "confidence_by_error_type.svg": "Mean model prediction score by diagnostic outcome; scores are not assumed to be calibrated probabilities.",
        "confidence_threshold_sweep.svg": "STRICT F1 after filtering predictions at configured score thresholds; this is not a calibration curve.",
        "confidence_coverage_vs_threshold.svg": "Retained-prediction coverage across configured model-score thresholds; scores are not assumed calibrated.",
        "error_intersections.svg": "Common supported intersections between entity characteristics and diagnostic error families.",
        "error_pareto.svg": "Concentration of strict error contributions among the most frequent failing entity surfaces.",
        "dataset_shift_labels.svg": "Descriptive training and evaluation label distributions; differences do not establish causes of model errors.",
        "dataset_shift_characteristics.svg": "Descriptive prevalence of acronym-like, digit-containing, and punctuation-containing mentions in training and evaluation annotations.",
    }


def _render_captions(captions):
    return "# Suggested scientific figure captions\n\n" + "\n\n".join(
        f"## `{name}`\n\n{caption}" for name, caption in captions.items()
    ) + "\n"


def _findings(analysis, oracles, paired):
    summary = analysis.summary()
    lines = ["# Factual diagnostic findings", ""]
    lines.append(f"- Fuzzy-zero-shot entities represent {summary.get('zero_shot_rate', 0):.1%} of evaluation gold.")
    recall = paired[paired["metric"] == "strict_recall"]
    if len(recall):
        row = recall.iloc[0]
        interval = f" (CI {row.ci_low:.3f} to {row.ci_high:.3f})" if pd.notna(row.ci_low) else ""
        lines.append(f"- Observed fuzzy-seen minus fuzzy-zero-shot STRICT recall is {row.observed_difference:.3f}{interval}.")
    single = oracles[oracles["scenario_kind"] == "single_family"]
    if len(single):
        largest = single.sort_values("delta_f1", ascending=False).iloc[0]
        lines.append(f"- {largest.error_family} has the largest measured single-family diagnostic ceiling (STRICT ΔF1={largest.delta_f1:.3f}) and is a candidate for targeted investigation.")
    confusion = summary.get("most_frequent_label_confusion")
    if confusion:
        lines.append(f"- The most frequent exact-boundary label confusion is {confusion['gold_label']} → {confusion['pred_label']} (N={confusion['count']}).")
    boundary = summary.get("most_common_boundary_deviation")
    if boundary:
        lines.append(f"- The most common observed boundary category is {boundary['value']} (N={boundary['count']}).")
    high = int(analysis.events["confidence_high_error"].fillna(False).sum())
    lines.append(f"- {high} strict error contributions have prediction scores at or above the configured high-confidence threshold.")
    lines.extend(["", "These are descriptive associations and diagnostic upper bounds; they do not establish causal explanations."])
    return "\n".join(lines) + "\n"


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
