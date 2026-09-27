"""Deterministic Parquet-driven core scientific figure catalogue."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import yaml

from lab.ner.explicability.config import VisualizationConfig, resolve_config
from lab.ner.explicability.finalize import explicability_root, read_manifest
from lab.ner.explicability.storage import write_table
from lab.ner.explicability.utils import atomic_json, directory_size
from lab.ner.explicability.visualization.animations import animated_svg, export_mp4
from lab.ner.explicability.visualization.attribution import attribution_tokens
from lab.ner.explicability.visualization.composites import (
    adaptation_composite, fingerprint_figure, fingerprint_table,
)
from lab.ner.explicability.visualization.distributions import temporal_lines
from lab.ner.explicability.visualization.dynamics import (
    cartography_figure, first_learning_distribution, forgetting_distribution,
    learning_timeline, transition_heatmap,
)
from lab.ner.explicability.visualization.embeddings import temporal_small_multiples, uncertainty_view
from lab.ner.explicability.visualization.export import deterministic_filename, export_figure
from lab.ner.explicability.visualization.heatmaps import heatmap_figure
from lab.ner.explicability.visualization.layout import SvgFigure
from lab.ner.explicability.visualization.neighborhoods import neighborhood_panel
from lab.ner.explicability.visualization.parameters import parameter_heatmap
from lab.ner.explicability.visualization.probing import probing_heatmap
from lab.ner.explicability.visualization.report import generate_report
from lab.ner.explicability.visualization.style import profile
from lab.ner.explicability.visualization.trajectories import centroid_trajectories, selected_trajectories
from lab.ner.explicability.visualization.validation import validate_figure_outputs


@dataclass
class FigureArtifact:
    figure_id: str
    filename: str
    category: str
    analysis: str
    source_table: str
    profile: str = "paper"
    representation_level: str | None = None
    layer: str | None = None
    checkpoints: str | None = None
    projection_method: str | None = None
    entity_colors: str = "{}"
    rasterized_artists: str = "[]"
    axis_labels: str = "[]"


CATEGORIES = ("representations", "trajectories", "geometry", "dynamics", "neighborhoods",
              "probing", "attribution", "parameters", "composites")


class TableCache:
    def __init__(self, root: Path):
        self.root = root
        self.frames: dict[str, pd.DataFrame] = {}

    def get(self, name: str, columns: list[str] | None = None) -> pd.DataFrame:
        path = self.root / "tables" / f"{name}.parquet"
        if not path.exists():
            return pd.DataFrame()
        key = f"{name}:{','.join(columns or ())}"
        if key not in self.frames:
            self.frames[key] = pd.read_parquet(path, columns=columns)
        return self.frames[key]


def generate_figures(
    run_dir: str | Path, *, profile_name: str | None = None,
    analysis: str | None = None, figure_id: str | None = None,
    animations: bool | None = None,
) -> dict[str, Path]:
    """Regenerate selected or all figures using permanent Parquet tables only."""
    started = time.perf_counter()
    root = explicability_root(run_dir)
    raw = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8")) or {}
    config = resolve_config(raw).visualization
    if profile_name is not None:
        config = VisualizationConfig(**{**asdict(config), "profile": profile_name})
    style = profile(config.profile, "presentation_16_9" if config.profile == "presentation" else "double_column")
    for category in CATEGORIES:
        (root / "figures" / category).mkdir(parents=True, exist_ok=True)
    cache = TableCache(root)
    manifest = read_manifest(root)
    records: list[FigureArtifact] = []
    outputs: dict[str, Path] = {}

    def emit(identifier, category, source, builder: Callable[[], object], analysis_name=None, **metadata):
        if figure_id and identifier != figure_id:
            return
        if analysis and (analysis_name or source) != analysis:
            return
        source_frame = cache.get(source)
        if source_frame.empty:
            return
        result = builder()
        if isinstance(result, tuple):
            figure = result[0]
            colors = result[1] if len(result) > 1 and isinstance(result[1], dict) else {}
            extra = result[2] if len(result) > 2 and isinstance(result[2], dict) else {}
        else:
            figure, colors, extra = result, {}, {}
        filename = Path("figures") / category / deterministic_filename(identifier)
        export_figure(figure, root / filename, config)
        outputs[identifier] = root / filename
        checkpoint_columns = [column for column in ("checkpoint", "checkpoint_a", "checkpoint_b") if column in source_frame]
        checkpoint_values = sorted({
            str(value) for column in checkpoint_columns
            for value in source_frame[column].dropna().unique()
        })
        if "representation_level" not in metadata and "representation_level" in source_frame:
            levels = sorted(source_frame["representation_level"].dropna().astype(str).unique())
            metadata["representation_level"] = ",".join(levels) or None
        if "layer" not in metadata:
            layer_column = next((column for column in ("layer", "layer_a") if column in source_frame), None)
            if layer_column:
                metadata["layer"] = ",".join(map(str, sorted(source_frame[layer_column].dropna().unique()))) or None
        metadata.setdefault("checkpoints", ",".join(checkpoint_values) or None)
        if "projection_method" not in metadata and "projection_method" in source_frame:
            methods = sorted(source_frame["projection_method"].dropna().astype(str).unique())
            metadata["projection_method"] = ",".join(methods) or None
        records.append(FigureArtifact(
            identifier, str(filename), category, analysis_name or source,
            f"tables/{source}.parquet", profile=config.profile,
            **metadata, entity_colors=json.dumps(colors, sort_keys=True),
            rasterized_artists=json.dumps(extra.get("rasterized_artists", [])),
            axis_labels=json.dumps(figure.axis_labels),
        ))

    _emit_catalog(emit, cache, style, config)
    fingerprint = fingerprint_table({name: cache.get(name) for name in (
        "trajectories", "geometry", "intrinsic_dimension", "parameter_drift", "forgetting"
    )})
    if not fingerprint.empty and (not figure_id or figure_id == "26_corpus_adaptation_fingerprint") and (not analysis or analysis == "fingerprint"):
        fingerprint_path = write_table(fingerprint, root / "tables" / "corpus_adaptation_fingerprint.parquet", sort_by=["metric"])
        identifier = "26_corpus_adaptation_fingerprint"
        filename = Path("figures/composites") / deterministic_filename(identifier)
        export_figure(fingerprint_figure(fingerprint, style), root / filename, config)
        outputs[identifier] = root / filename
        records.append(FigureArtifact(identifier, str(filename), "composites", "fingerprint",
                                      str(fingerprint_path.relative_to(root)), profile=config.profile,
                                      axis_labels=json.dumps(["Normalized summary magnitude", "Adaptation diagnostic"])))

    animation_enabled = config.animation if animations is None else animations
    projection = cache.get("projection")
    if not projection.empty:
        if "representation_level" in projection:
            projection = projection[projection["representation_level"] == "token"]
        if "layer" in projection and projection["layer"].notna().any():
            projection = projection[projection["layer"] == projection["layer"].max()]
    if animation_enabled and not projection.empty and not figure_id and not analysis:
        animation_path = animated_svg(
            projection, root / "animations" / "temporal_embedding_evolution.svg",
            config.palette, 1 / config.animation_fps,
            config.visual_sample_size, config.visual_sample_seed,
        )
        outputs["temporal_embedding_animation"] = animation_path
        mp4 = export_mp4(
            projection, root / "animations" / "temporal_embedding_evolution.mp4",
            config.animation_fps, config.visual_sample_size, config.visual_sample_seed,
        )
        if mp4:
            outputs["temporal_embedding_animation_mp4"] = mp4

    if not records:
        raise ValueError("No applicable figures were generated for the requested selection.")
    figure_manifest = _figure_manifest(records, config, manifest)
    existing_path = root / "tables" / "figure_manifest.parquet"
    if existing_path.exists() and (figure_id or analysis):
        existing = pd.read_parquet(existing_path)
        existing = existing[~existing["figure_id"].isin(figure_manifest["figure_id"])]
        figure_manifest = pd.concat([existing, figure_manifest], ignore_index=True)
    figure_manifest_path = write_table(
        figure_manifest, root / "tables" / "figure_manifest.parquet", sort_by=["figure_id"]
    )
    outputs["figure_manifest"] = figure_manifest_path
    report = generate_report(root, figure_manifest, manifest)
    outputs["report"] = report
    errors = validate_figure_outputs(root, figure_manifest["figure_id"].tolist())
    if errors:
        raise ValueError("Figure validation failed; snapshots were retained: " + "; ".join(errors))
    manifest.update({
        "completed_visualizations": sorted(figure_manifest["figure_id"].tolist()),
        "visualization_profile": config.profile,
        "visualization_runtime_seconds": time.perf_counter() - started,
        "figure_manifest": "tables/figure_manifest.parquet",
        "report": "report/index.html",
        "visualization_status": "partial" if (figure_id or analysis) else "complete",
        "status": manifest.get("status") if (figure_id or analysis) else "visualizations_complete",
        "permanent_storage_bytes": directory_size(root / "tables") + directory_size(root / "figures") + directory_size(root / "report"),
    })
    required = set(manifest.get("required_outputs", []))
    required.update(str(path.relative_to(root)) for path in outputs.values())
    manifest["required_outputs"] = sorted(required)
    atomic_json(manifest, root / "manifest.json")
    return outputs


def _emit_catalog(emit, cache, style, config):
    projection = cache.get("projection")
    if not projection.empty:
        if "representation_level" in projection:
            projection = projection[projection["representation_level"] == "token"]
        if "layer" in projection and projection["layer"].notna().any():
            projection = projection[projection["layer"] == projection["layer"].max()]
    trajectories = cache.get("trajectories")
    cka = cache.get("cka")
    geometry = cache.get("geometry")
    intrinsic = cache.get("intrinsic_dimension")
    neighborhoods = cache.get("neighborhoods")
    dynamics = cache.get("training_dynamics")
    forgetting = cache.get("forgetting")
    transitions = cache.get("transitions")
    probing = cache.get("probing")
    parameters = cache.get("parameter_drift")
    attribution = cache.get("attribution")
    relationships = cache.get("performance_relationships")
    specialization = cache.get("layer_specialization")

    if not relationships.empty:
        joined = relationships[relationships["record_type"] == "joined_checkpoint"] if "record_type" in relationships else relationships
        metric = _first_numeric(joined, preferred=("span_strict_f1", "f1", "token_micro_f1"))
        if metric:
            emit("01_performance_training", "composites", "performance_relationships",
                 lambda: temporal_lines(joined, "checkpoint", metric, style, ylabel="NER performance"), "performance")
    if not projection.empty:
        method = str(projection["projection_method"].dropna().iloc[0])
        if method == "shared_pca":
            emit("02_shared_pca_evolution", "representations", "projection",
                 lambda: temporal_small_multiples(projection, style, palette=config.palette, background_limit=config.visual_sample_size, seed=config.visual_sample_seed),
                 "temporal_projection", representation_level="token", projection_method=method)
        if method == "aligned_umap":
            emit("03_aligned_umap_evolution", "representations", "projection",
                 lambda: temporal_small_multiples(projection, style, palette=config.palette, entity_only=True),
                 "temporal_projection", representation_level="token", projection_method=method)
        emit("04_temporal_embedding_small_multiples", "representations", "projection",
             lambda: temporal_small_multiples(projection, style, palette=config.palette, entity_only=True),
             "temporal_projection", representation_level="token", projection_method=method)
        emit("embedding_correctness_evolution", "representations", "projection",
             lambda: temporal_small_multiples(
                 projection.assign(correctness=projection["correct"].map(
                     {True: "correct", False: "incorrect"}).fillna("unlabelled")
                 ), style, palette=config.palette, background_limit=0,
                 color_by="correctness",
             ), "temporal_projection", representation_level="token", projection_method=method)
        emit("05_entity_centroid_trajectories", "trajectories", "projection",
             lambda: centroid_trajectories(projection, style, palette=config.palette), "centroid_dynamics")
        if "confidence" in projection:
            emit("embedding_uncertainty_confidence", "representations", "projection",
                 lambda: uncertainty_view(projection, style, "confidence"), "temporal_projection")
    if not trajectories.empty:
        observed = trajectories[trajectories["record_type"] == "observation"] if "record_type" in trajectories else trajectories
        emit("06_representation_drift", "trajectories", "trajectories",
             lambda: temporal_lines(observed, "checkpoint", "distance_pretrained", style, group="layer", ylabel="Distance from pretrained"), "trajectories")
        if not projection.empty:
            emit("selected_representation_trajectories", "trajectories", "trajectories",
                 lambda: selected_trajectories(projection, trajectories, style, palette=config.palette), "trajectories")
        for identifier, metric, label in (
            ("distance_from_final", "distance_final", "Distance from final"),
            ("cumulative_path_length", "cumulative_path_length", "Cumulative path length"),
        ):
            if metric in observed:
                emit(identifier, "trajectories", "trajectories",
                     lambda metric=metric, label=label: temporal_lines(
                         observed, "checkpoint", metric, style, group="layer", ylabel=label
                     ), "trajectories")
    if not cka.empty:
        baseline = cka[cka["checkpoint_a"] == "step_000_pretrained"]
        emit("07_cka_to_pretrained", "geometry", "cka",
             lambda: temporal_lines(baseline.rename(columns={"checkpoint_b": "checkpoint"}), "checkpoint", "cka_score", style, group="layer_b", ylabel="CKA to pretrained"), "cka")
        same_checkpoint = cka[cka["checkpoint_a"] == cka["checkpoint_b"]]
        checkpoint = same_checkpoint["checkpoint_a"].astype(str).iloc[-1] if not same_checkpoint.empty else None
        emit("08_cka_layer_similarity", "geometry", "cka",
             lambda: heatmap_figure(same_checkpoint[same_checkpoint["checkpoint_a"].astype(str) == checkpoint], "layer_a", "layer_b", "cka_score", style, xlabel="Layer B", ylabel="Layer A"), "cka")
        same_layer = cka[cka["layer_a"] == cka["layer_b"]]
        layer = same_layer["layer_a"].max() if not same_layer.empty else None
        emit("09_cka_checkpoint_similarity", "geometry", "cka",
             lambda: heatmap_figure(same_layer[same_layer["layer_a"] == layer], "checkpoint_a", "checkpoint_b", "cka_score", style, xlabel="Checkpoint B", ylabel="Checkpoint A"), "cka")
        checkpoint_order = list(dict.fromkeys(
            pd.concat([cka["checkpoint_a"], cka["checkpoint_b"]]).astype(str)
        ))
        order = {checkpoint: index for index, checkpoint in enumerate(checkpoint_order)}
        consecutive = cka[
            cka.apply(lambda row: order.get(str(row["checkpoint_b"]), -99) -
                      order.get(str(row["checkpoint_a"]), 99) == 1, axis=1)
        ].rename(columns={"checkpoint_b": "checkpoint"})
        if not consecutive.empty:
            emit("cka_consecutive_change", "geometry", "cka",
                 lambda: temporal_lines(consecutive, "checkpoint", "cka_score", style,
                                        group="layer_b", ylabel="Consecutive-checkpoint CKA"),
                 "cka")
    if not specialization.empty:
        metric = _first_numeric(specialization, preferred=("cka_to_pretrained", "probe_f1", "knn_purity"))
        emit("10_layer_adaptation_map", "geometry", "layer_specialization",
             lambda: heatmap_figure(specialization, "layer", "checkpoint", metric, style, xlabel="Checkpoint", ylabel="Layer"), "layer_specialization")
        for diagnostic in (
            "cka_to_pretrained", "probe_f1", "intrinsic_dimension", "anisotropy", "knn_purity"
        ):
            if diagnostic in specialization and specialization[diagnostic].notna().any():
                emit(f"layer_adaptation_{diagnostic}", "geometry", "layer_specialization",
                     lambda diagnostic=diagnostic: heatmap_figure(
                         specialization, "layer", "checkpoint", diagnostic, style,
                         xlabel="Checkpoint", ylabel="Layer"
                     ), "layer_specialization")
    if not intrinsic.empty:
        selected = intrinsic[intrinsic["estimator"] == "participation_ratio"]
        emit("11_intrinsic_dimension", "geometry", "intrinsic_dimension",
             lambda: temporal_lines(selected, "checkpoint", "estimate", style, group="layer", ylabel="Intrinsic dimension"), "intrinsic_dimension")
    for identifier, metric, label in (
        ("12_effective_rank", "effective_rank", "Effective rank"),
        ("13_anisotropy", "mean_pairwise_cosine", "Mean pairwise cosine"),
        ("16_knn_purity", "knn_label_purity", "kNN label purity"),
        ("geometry_silhouette", "silhouette", "Silhouette score"),
        ("geometry_fisher_ratio", "fisher_discriminant_ratio", "Fisher discriminant ratio"),
        ("geometry_participation_ratio", "participation_ratio", "Participation ratio"),
        ("geometry_intra_class_distance", "intra_class_euclidean", "Intra-class distance"),
        ("geometry_inter_class_distance", "inter_class_euclidean", "Inter-class distance"),
    ):
        if metric in geometry:
            emit(identifier, "geometry", "geometry",
                 lambda metric=metric, label=label: temporal_lines(geometry, "checkpoint", metric, style, group="layer", ylabel=label), "geometry")
    centroid = cache.get("centroid_dynamics")
    if not centroid.empty:
        summary = centroid[centroid["record_type"] == "class_summary"] if "record_type" in centroid else centroid
        emit("14_entity_compactness", "geometry", "centroid_dynamics",
             lambda: temporal_lines(summary, "checkpoint", "within_class_radius", style, group="class_label", ylabel="Within-class radius"), "centroid_dynamics")
        emit("15_entity_separation", "geometry", "centroid_dynamics",
             lambda: temporal_lines(summary, "checkpoint", "nearest_centroid_distance", style, group="class_label", ylabel="Nearest centroid distance"), "centroid_dynamics")
    if not neighborhoods.empty:
        summary = neighborhoods[neighborhoods["record_type"] == "summary"] if "record_type" in neighborhoods else neighborhoods
        emit("17_neighborhood_stability", "neighborhoods", "neighborhoods",
             lambda: temporal_lines(summary, "checkpoint", "jaccard_pretrained", style, ylabel="Neighborhood overlap with pretrained"), "neighborhoods")
        emit("semantic_neighborhood_change", "neighborhoods", "neighborhoods",
             lambda: neighborhood_panel(neighborhoods, style), "neighborhoods")
    if not dynamics.empty:
        emit("18_dataset_cartography", "dynamics", "training_dynamics",
             lambda: cartography_figure(dynamics, style, config.palette), "training_dynamics")
        if "record_type" in dynamics and (dynamics["record_type"] == "checkpoint_history").any():
            emit("selected_learning_timeline", "dynamics", "training_dynamics",
                 lambda: learning_timeline(dynamics, style), "training_dynamics")
    if not forgetting.empty:
        emit("19_forgetting_events", "dynamics", "forgetting",
             lambda: forgetting_distribution(forgetting, style), "forgetting")
        emit("20_first_learning", "dynamics", "forgetting",
             lambda: first_learning_distribution(forgetting, style), "forgetting")
    if not transitions.empty:
        emit("21_prediction_transitions", "dynamics", "transitions",
             lambda: transition_heatmap(transitions, style, True), "transitions")
    if not probing.empty:
        task = "entity_type" if "entity_type" in set(probing["probe_task"]) else str(probing["probe_task"].iloc[0])
        emit("22_probing_layer_epoch", "probing", "probing",
             lambda: probing_heatmap(probing, style, task), "probing")
        for probe_task in sorted(probing["probe_task"].dropna().astype(str).unique()):
            emit(f"probing_{probe_task}_layer_checkpoint", "probing", "probing",
                 lambda probe_task=probe_task: probing_heatmap(
                     probing, style, probe_task
                 ), "probing")
    if not parameters.empty:
        emit("23_parameter_drift", "parameters", "parameter_drift",
             lambda: parameter_heatmap(parameters, style), "parameter_drift")
    if not attribution.empty:
        case = attribution
        if "observation_id" in case:
            case = case[case["observation_id"] == case["observation_id"].iloc[0]]
        emit("attribution_selected_case", "attribution", "attribution",
             lambda: attribution_tokens(case, style), "attribution")
    series = []
    if not relationships.empty:
        joined = relationships[relationships["record_type"] == "joined_checkpoint"] if "record_type" in relationships else relationships
        metric = _first_numeric(joined, preferred=("span_strict_f1", "f1"))
        if metric: series.append(("NER performance", joined, metric))
    if not cka.empty: series.append(("CKA to pretrained", baseline.rename(columns={"checkpoint_b": "checkpoint"}), "cka_score"))
    if not trajectories.empty: series.append(("Representation drift", observed, "distance_pretrained"))
    if not geometry.empty and "silhouette" in geometry: series.append(("Entity separability", geometry, "silhouette"))
    if series:
        composite_source = "performance_relationships" if not relationships.empty else "cka"
        emit("24_performance_vs_internal_change", "composites", composite_source,
             lambda: adaptation_composite(series, style), "performance_relationships")
    hard = cache.get("hard_examples")
    if not hard.empty:
        emit("25_hard_example_panels", "dynamics", "hard_examples",
             lambda: _table_figure(hard.head(12), style, "Hard-example diagnostics"), "hard_examples")


def _table_figure(frame, style, heading):
    figure = SvgFigure(style)
    figure.text(30, 30, heading, weight="bold", size=style.font_size + 2)
    columns = [column for column in ("observation_id", "gold_label", "predicted_label", "forgetting_events", "hardness_rank_score") if column in frame]
    for row_index, record in enumerate(frame[columns].to_dict("records")):
        figure.text(35, 60 + row_index * 23, "  |  ".join(f"{key}: {str(value)[:28]}" for key, value in record.items()), size=style.font_size - 1)
    figure.axis_labels.extend(["Selected hard observations", "Ranked diagnostics"])
    return figure


def _first_numeric(frame, preferred=()):
    for column in preferred:
        if column in frame and pd.api.types.is_numeric_dtype(frame[column]) and frame[column].notna().any():
            return column
    excluded = {"analysis_seed", "configured_sample_size", "checkpoint_order"}
    return next((column for column in frame if column not in excluded and pd.api.types.is_numeric_dtype(frame[column]) and frame[column].notna().any()), None)


def _figure_manifest(records, config, manifest):
    rows = []
    for record in records:
        row = asdict(record)
        row.update({
            "run_id": manifest.get("run_id"), "model": manifest.get("model"),
            "split": manifest.get("split"), "palette": config.palette,
            "profile": config.profile, "seed": config.visual_sample_seed,
            "creation_configuration": json.dumps(asdict(config), sort_keys=True),
        })
        rows.append(row)
    return pd.DataFrame(rows)
