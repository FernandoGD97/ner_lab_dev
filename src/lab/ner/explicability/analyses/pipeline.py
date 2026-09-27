"""Transform Part 1 snapshots into compact Part 2 analytical Parquet datasets."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from lab.ner.explicability.analyses.data import SnapshotCollection, deterministic_sample
from lab.ner.explicability.analyses.dynamics import (
    forgetting_events, prediction_transitions, training_dynamics,
)
from lab.ner.explicability.analyses.geometry import geometry_metrics, intrinsic_dimension
from lab.ner.explicability.analyses.neighborhoods import neighborhood_evolution
from lab.ner.explicability.analyses.parameters import parameter_drift_files
from lab.ner.explicability.analyses.probing import mdl_probe, probe_targets, run_probes
from lab.ner.explicability.analyses.projection import temporal_projection
from lab.ner.explicability.analyses.relationships import (
    layer_specialization, performance_relationships,
)
from lab.ner.explicability.analyses.similarity import linear_cka, pwcca, svcca
from lab.ner.explicability.analyses.trajectories import (
    centroid_dynamics, representation_shifts, trajectory_metrics,
)
from lab.ner.explicability.config import AnalysisConfig, resolve_config
from lab.ner.explicability.finalize import explicability_root, read_manifest
from lab.ner.explicability.storage import write_table
from lab.ner.explicability.utils import atomic_json, directory_size


OUTPUTS = {
    "temporal_projection": "projection.parquet", "trajectories": "trajectories.parquet",
    "centroid_dynamics": "centroid_dynamics.parquet", "cka": "cka.parquet",
    "geometry": "geometry.parquet", "intrinsic_dimension": "intrinsic_dimension.parquet",
    "neighborhoods": "neighborhoods.parquet", "training_dynamics": "training_dynamics.parquet",
    "forgetting": "forgetting.parquet", "transitions": "transitions.parquet",
    "probing": "probing.parquet", "parameter_drift": "parameter_drift.parquet",
    "representation_shifts": "representation_shifts.parquet",
    "performance_relationships": "performance_relationships.parquet",
    "hard_examples": "hard_examples.parquet",
    "layer_specialization": "layer_specialization.parquet", "svcca": "svcca.parquet",
    "pwcca": "pwcca.parquet", "mdl_probing": "mdl_probing.parquet",
}


def run_analyses(
    run_dir: str | Path, config: AnalysisConfig | dict[str, Any] | None = None
) -> dict[str, Path]:
    """Run enabled analyses, persist every result, and update but do not finalize the run."""
    started = time.perf_counter()
    collection = SnapshotCollection(run_dir)
    root = collection.root
    if config is None:
        raw = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8")) or {}
        analysis_config = resolve_config(raw).analyses
    elif isinstance(config, AnalysisConfig):
        analysis_config = config
    else:
        analysis_config = resolve_config({"analyses": config}).analyses
    enabled = set(analysis_config.enabled())
    unsupported = enabled & {"attribution", "influence", "topology", "adaptation_phases"}
    if unsupported:
        raise ValueError(
            "Targeted/optional analyses require explicit case inputs and cannot run over all snapshots: "
            + ", ".join(sorted(unsupported))
        )
    checkpoints = collection.checkpoints
    common_ids = collection.common_observation_ids()
    if not len(common_ids):
        raise ValueError("Snapshots have no stable observations in common.")
    metadata = _longitudinal_metadata(collection, checkpoints, common_ids)
    base_metadata = metadata[metadata["checkpoint"] == checkpoints[0]].set_index("observation_id")
    eligible = base_metadata.loc[common_ids, "gold_label"].notna().to_numpy()
    common_ids = common_ids[eligible]
    labels_all = base_metadata.loc[common_ids, "gold_label"].astype(str).to_numpy()
    sample = deterministic_sample(
        common_ids, analysis_config.sample_size, analysis_config.sample_seed, labels_all
    )
    ids = common_ids[sample]
    labels = labels_all[sample]
    metadata = metadata[metadata["observation_id"].isin(ids)].copy()
    metadata["checkpoint_order"] = metadata["checkpoint"].map(
        {checkpoint: index for index, checkpoint in enumerate(checkpoints)}
    )
    paths: dict[str, Path] = {}
    tables: dict[str, list[pd.DataFrame]] = {name: [] for name in OUTPUTS}

    for layer in collection.layers:
        tensors = []
        for checkpoint in checkpoints:
            layer_ids, values = collection.load_layer(checkpoint, layer, set(ids))
            lookup = {item: index for index, item in enumerate(layer_ids)}
            tensors.append(values[[lookup[item] for item in ids]])

        if "temporal_projection" in enabled:
            projected = temporal_projection(
                {checkpoint: (ids, values) for checkpoint, values in zip(checkpoints, tensors)},
                analysis_config.projection_method, analysis_config.projection_dimensions,
                analysis_config.sample_seed,
            )
            projected["layer"] = layer
            projected["representation_level"] = "token"
            tables["temporal_projection"].append(projected.merge(
                metadata[[
                    "observation_id", "checkpoint", "gold_label", "predicted_label",
                    "correct", "confidence", "gold_confidence", "entropy",
                    "gold_vs_second_best_margin", "error_type",
                ]],
                on=["observation_id", "checkpoint"], how="left",
            ))
        if "trajectories" in enabled:
            frame = trajectory_metrics(checkpoints, ids, tensors)
            frame["layer"] = layer
            frame["representation_level"] = "token"
            frame = frame.merge(
                metadata[["observation_id", "checkpoint", "gold_label", "predicted_label", "correct", "error_type"]],
                on=["observation_id", "checkpoint"], how="left",
            )
            frame["record_type"] = "observation"
            tables["trajectories"].append(pd.concat([
                frame, _trajectory_aggregates(frame)
            ], ignore_index=True))
        if "centroid_dynamics" in enabled:
            summary, pairs = centroid_dynamics(checkpoints, tensors, labels, analysis_config.minimum_group_size)
            summary["record_type"] = "class_summary"
            if not pairs.empty:
                pairs["record_type"] = "centroid_pair"
            combined = pd.concat([summary, pairs], ignore_index=True)
            combined["layer"] = layer
            tables["centroid_dynamics"].append(combined)
        if "representation_shifts" in enabled:
            shifts = representation_shifts(checkpoints, tensors, labels)
            shifts["layer"] = layer
            tables["representation_shifts"].append(shifts)
        for method in ("cka", "svcca", "pwcca"):
            if method in enabled:
                tables[method].append(_similarity_table(method, checkpoints, tensors, layer, len(ids)))
        if "geometry" in enabled or "intrinsic_dimension" in enabled:
            for checkpoint, values in zip(checkpoints, tensors):
                if "geometry" in enabled:
                    row = geometry_metrics(
                        values, labels, analysis_config.neighbors_k,
                        analysis_config.minimum_group_size, analysis_config.block_size,
                    )
                    tables["geometry"].append(pd.DataFrame([{
                        "checkpoint": checkpoint, "layer": layer,
                        "representation_level": "token", **row,
                    }]))
                if "intrinsic_dimension" in enabled:
                    frame = pd.DataFrame(intrinsic_dimension(values))
                    frame["checkpoint"] = checkpoint
                    frame["layer"] = layer
                    frame["representation_level"] = "token"
                    tables["intrinsic_dimension"].append(frame)
        if "neighborhoods" in enabled:
            detail, summary = neighborhood_evolution(
                checkpoints, ids, tensors, labels, analysis_config.neighbors_k,
                analysis_config.block_size,
            )
            detail["record_type"] = "neighbor"
            summary["record_type"] = "summary"
            combined = pd.concat([detail, summary], ignore_index=True)
            combined["layer"] = layer
            tables["neighborhoods"].append(combined)
        if "probing" in enabled:
            for checkpoint, values in zip(checkpoints, tensors):
                tables["probing"].append(run_probes(
                    checkpoint, layer, ids, values, labels, analysis_config.probe_seed,
                    analysis_config.probe_train_fraction,
                    analysis_config.probe_validation_fraction,
                    analysis_config.probe_max_iterations,
                    analysis_config.probe_learning_rate,
                ))
        if "mdl_probing" in enabled:
            for checkpoint, values in zip(checkpoints, tensors):
                for task in ("entity_vs_non_entity", "bio_boundary", "entity_type"):
                    result = mdl_probe(
                        ids, values, probe_targets(labels, task), analysis_config.probe_seed
                    )
                    tables["mdl_probing"].append(pd.DataFrame([{
                        "checkpoint": checkpoint, "layer": layer,
                        "probe_task": task, "seed": analysis_config.probe_seed, **result,
                    }]))
        del tensors

    if "cka" in enabled and len(collection.layers) > 1:
        for checkpoint in checkpoints:
            layer_values = {}
            for layer in collection.layers:
                layer_ids, values = collection.load_layer(checkpoint, layer, set(ids))
                lookup = {item: index for index, item in enumerate(layer_ids)}
                layer_values[layer] = values[[lookup[item] for item in ids]]
            for left_index, left in enumerate(collection.layers):
                for right in collection.layers[left_index + 1:]:
                    tables["cka"].append(pd.DataFrame([{
                        "checkpoint_a": checkpoint, "checkpoint_b": checkpoint,
                        "layer_a": left, "layer_b": right,
                        "sample_population": "stable_common", "sample_count": len(ids),
                        "cka_score": linear_cka(layer_values[left], layer_values[right]),
                    }]))
            del layer_values

    for representation_level, observations in _metadata_levels(metadata):
        if "training_dynamics" in enabled:
            frame = training_dynamics(observations)
            frame["representation_level"] = representation_level
            frame["record_type"] = "summary"
            history_columns = [column for column in (
                "observation_id", "checkpoint", "checkpoint_order", "gold_label",
                "predicted_label", "correct", "gold_confidence", "confidence",
                "entropy", "gold_vs_second_best_margin",
            ) if column in observations]
            history = observations[history_columns].copy()
            history["representation_level"] = representation_level
            history["record_type"] = "checkpoint_history"
            tables["training_dynamics"].append(pd.concat([frame, history], ignore_index=True))
        if "forgetting" in enabled:
            frame = forgetting_events(observations)
            frame["representation_level"] = representation_level
            tables["forgetting"].append(frame)
        if "transitions" in enabled:
            frame = prediction_transitions(observations)
            frame["representation_level"] = representation_level
            tables["transitions"].append(frame)
    if "parameter_drift" in enabled:
        baseline = collection.parameter_path(checkpoints[0])
        if not baseline.exists():
            raise ValueError("parameter_drift enabled but snapshots contain no parameter states.")
        for checkpoint in checkpoints:
            state = collection.parameter_path(checkpoint)
            if not state.exists():
                raise ValueError(f"Missing parameter state for {checkpoint}.")
            tables["parameter_drift"].append(
                parameter_drift_files(baseline, state, checkpoint)
            )

    preliminary = _frames(tables)
    if "performance_relationships" in enabled:
        performance = _performance_table(Path(run_dir), checkpoints)
        internal = _internal_summary(preliminary)
        joined, correlations = performance_relationships(performance, internal)
        correlations["record_type"] = "correlation"
        joined["record_type"] = "joined_checkpoint"
        tables["performance_relationships"].append(pd.concat([joined, correlations], ignore_index=True))
    if "hard_examples" in enabled:
        tables["hard_examples"].append(_hard_examples(metadata, preliminary))
    if "layer_specialization" in enabled:
        tables["layer_specialization"].append(_specialization(preliminary))

    tables_dir = root / "tables"
    for name in sorted(enabled & set(OUTPUTS)):
        frame = pd.concat(tables[name], ignore_index=True) if tables[name] else pd.DataFrame()
        if frame.empty:
            raise ValueError(f"Enabled analysis {name!r} produced no results.")
        frame["analysis_seed"] = analysis_config.sample_seed
        frame["sample_population"] = (
            "full_gold_labeled" if analysis_config.sample_size is None
            else "explicit_stratified_gold_labeled_sample"
        )
        frame["configured_sample_size"] = analysis_config.sample_size
        paths[name] = write_table(frame, tables_dir / OUTPUTS[name], sort_by=_sort_columns(frame))

    _update_manifest(root, analysis_config, paths, time.perf_counter() - started, len(ids))
    return paths


def _longitudinal_metadata(collection, checkpoints, common_ids):
    frames = []
    for checkpoint in checkpoints:
        frame = collection.metadata(checkpoint)
        frame = frame[frame["observation_id"].isin(common_ids)].copy()
        frame["checkpoint"] = checkpoint
        frame["error_type"] = frame.apply(_error_type, axis=1)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _error_type(row) -> str:
    gold, predicted = row.get("gold_label"), row.get("predicted_label")
    if gold is None or pd.isna(gold):
        return "unlabelled"
    if gold == predicted:
        return "correct_non_entity" if gold == "O" else "true_positive"
    if gold == "O":
        return "false_positive"
    if predicted == "O":
        return "false_negative"
    gold_prefix, _, gold_type = str(gold).partition("-")
    predicted_prefix, _, predicted_type = str(predicted).partition("-")
    return "boundary_error" if gold_type == predicted_type and gold_prefix != predicted_prefix else "wrong_entity_type"


def _similarity_table(method, checkpoints, tensors, layer, count):
    function = {"cka": linear_cka, "svcca": svcca, "pwcca": pwcca}[method]
    pairs = {
        (left, right) for left in range(len(checkpoints))
        for right in range(left, len(checkpoints))
    }
    return pd.DataFrame([{
        "checkpoint_a": checkpoints[left], "checkpoint_b": checkpoints[right],
        "layer_a": layer, "layer_b": layer, "sample_population": "stable_common",
        "sample_count": count, f"{method}_score": function(tensors[left], tensors[right]),
    } for left, right in sorted(pairs)])


def _trajectory_aggregates(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "distance_previous", "cosine_distance_previous", "angular_change_radians",
        "distance_pretrained", "distance_final", "cumulative_path_length",
        "net_displacement", "path_displacement_ratio", "maximum_single_step",
    ]
    groups = ["checkpoint", "layer", "gold_label", "correct", "error_type"]
    aggregate = frame.groupby(groups, dropna=False, sort=True)[metrics].agg(["mean", "median"])
    aggregate.columns = [f"{metric}_{statistic}" for metric, statistic in aggregate.columns]
    aggregate = aggregate.reset_index()
    aggregate["sample_count"] = frame.groupby(groups, dropna=False).size().to_numpy()
    aggregate["record_type"] = "group_summary"
    aggregate["representation_level"] = "token"
    return aggregate


def _metadata_levels(metadata: pd.DataFrame):
    token = metadata.copy()
    yield "token", token
    for level, column in (("word", "word_observation_id"), ("document", "document_id")):
        if column not in metadata:
            continue
        selected = metadata[metadata[column].notna()].copy()
        if selected.empty:
            continue
        selected["observation_id"] = selected[column].astype(str)
        keys = ["observation_id", "checkpoint", "checkpoint_order"]
        rows = []
        for group_key, group in selected.groupby(keys, sort=True):
            predicted = group["predicted_label"].mode().iloc[0]
            gold = group["gold_label"].mode().iloc[0]
            rows.append({
                "observation_id": group_key[0], "checkpoint": group_key[1],
                "checkpoint_order": group_key[2], "predicted_label": predicted,
                "gold_label": gold, "correct": bool((group["predicted_label"] == group["gold_label"]).all()),
                "confidence": float(group["confidence"].mean()),
                "gold_confidence": float(group["gold_confidence"].mean()),
                "entropy": float(group["entropy"].mean()),
                "gold_vs_second_best_margin": float(group["gold_vs_second_best_margin"].mean()),
            })
        yield level, pd.DataFrame(rows)


def _frames(tables):
    return {name: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame() for name, parts in tables.items()}


def _performance_table(run_dir: Path, checkpoints: list[str]) -> pd.DataFrame:
    candidates = [run_dir / "epoch_metrics.parquet", *sorted(run_dir.glob("fold_*/epoch_metrics.parquet"))]
    frames = [pd.read_parquet(path) for path in candidates if path.exists()]
    if not frames:
        return pd.DataFrame({"checkpoint": checkpoints, "evaluation_order": range(len(checkpoints))})
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame.get("split", "eval") == "eval"] if "split" in frame else frame
    intermediate = [checkpoint for checkpoint in checkpoints if checkpoint not in {"step_000_pretrained", "step_final"}]
    frame = frame.head(len(intermediate)).copy()
    frame["checkpoint"] = intermediate[:len(frame)]
    return frame


def _internal_summary(tables):
    frames = []
    trajectories = tables.get("trajectories", pd.DataFrame())
    if not trajectories.empty:
        frames.append(trajectories.groupby("checkpoint", as_index=False).agg(
            mean_representation_displacement=("distance_pretrained", "mean"),
            mean_representation_velocity=("distance_previous", "mean")))
    geometry = tables.get("geometry", pd.DataFrame())
    if not geometry.empty:
        frames.append(geometry.groupby("checkpoint", as_index=False).agg(
            silhouette=("silhouette", "mean"), knn_purity=("knn_label_purity", "mean"),
            anisotropy=("mean_pairwise_cosine", "mean")))
    dimensions = tables.get("intrinsic_dimension", pd.DataFrame())
    if not dimensions.empty:
        selected = dimensions[dimensions["estimator"] == "participation_ratio"]
        frames.append(selected.groupby("checkpoint", as_index=False).agg(
            intrinsic_dimension=("estimate", "mean")))
    probes = tables.get("probing", pd.DataFrame())
    if not probes.empty:
        frames.append(probes.groupby("checkpoint", as_index=False).agg(probe_f1=("f1", "mean")))
    if not frames:
        return pd.DataFrame(columns=["checkpoint"])
    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on="checkpoint", how="outer")
    return result


def _hard_examples(metadata, tables, limit=100):
    dynamics = tables.get("training_dynamics", pd.DataFrame())
    forgetting = tables.get("forgetting", pd.DataFrame())
    trajectories = tables.get("trajectories", pd.DataFrame())
    neighborhoods = tables.get("neighborhoods", pd.DataFrame())
    if "representation_level" in dynamics:
        dynamics = dynamics[dynamics["representation_level"] == "token"]
    if "record_type" in dynamics:
        dynamics = dynamics[dynamics["record_type"] == "summary"]
    if "representation_level" in forgetting:
        forgetting = forgetting[forgetting["representation_level"] == "token"]
    base = dynamics.merge(forgetting, on="observation_id", how="outer")
    if not trajectories.empty:
        if "record_type" in trajectories:
            trajectories = trajectories[trajectories["record_type"] == "observation"]
        final = trajectories.sort_values("checkpoint").groupby("observation_id").tail(1)
        base = base.merge(final[["observation_id", "cumulative_path_length", "maximum_single_step"]], on="observation_id", how="outer")
    if not neighborhoods.empty and "neighborhood_turnover" in neighborhoods:
        turnover = neighborhoods.groupby("observation_id", as_index=False)["neighborhood_turnover"].mean()
        base = base.merge(turnover, on="observation_id", how="outer")
    final_meta = metadata.sort_values("checkpoint_order").groupby("observation_id").tail(1)
    base = base.merge(final_meta[["observation_id", "document_id", "token_position", "gold_label", "predicted_label"]], on="observation_id", how="left")
    columns = [column for column in ("cumulative_path_length", "maximum_single_step", "mean_entropy", "forgetting_events", "confidence_variability", "neighborhood_turnover") if column in base]
    score = sum(base[column].rank(pct=True).fillna(0) for column in columns)
    base["hardness_rank_score"] = score
    return base.nlargest(min(limit, len(base)), "hardness_rank_score")


def _specialization(tables):
    frames = []
    cka = tables.get("cka", pd.DataFrame())
    if not cka.empty:
        baseline = cka[cka["checkpoint_a"] == "step_000_pretrained"].rename(columns={"checkpoint_b": "checkpoint", "layer_b": "layer", "cka_score": "cka_to_pretrained"})
        frames.append(baseline[["checkpoint", "layer", "cka_to_pretrained"]])
    for name, columns in (("probing", {"f1": "probe_f1"}), ("geometry", {"mean_pairwise_cosine": "anisotropy", "knn_label_purity": "knn_purity"}), ("intrinsic_dimension", {"estimate": "intrinsic_dimension"})):
        frame = tables.get(name, pd.DataFrame())
        if not frame.empty:
            if name == "intrinsic_dimension":
                frame = frame[frame["estimator"] == "participation_ratio"]
            frames.append(frame.groupby(["checkpoint", "layer"], as_index=False).agg({source: "mean" for source in columns}).rename(columns=columns))
    return layer_specialization(*frames)


def _sort_columns(frame):
    preferred = ["checkpoint", "checkpoint_a", "layer", "layer_a", "observation_id", "record_type", "rank"]
    return [column for column in preferred if column in frame]


def _update_manifest(root, config, paths, runtime, sample_count):
    manifest = read_manifest(root)
    complete = sorted(set(manifest.get("completed_analyses", [])) | set(paths))
    required = sorted(set(manifest.get("required_outputs", [])) | {
        str(path.relative_to(root)) for path in paths.values()
    })
    manifest.update({
        "completed_analyses": complete, "mandatory_analyses": sorted(config.enabled()),
        "required_outputs": required, "analysis_parameters": config.__dict__,
        "analysis_runtime_seconds": runtime, "analysis_sample_count": sample_count,
        "permanent_storage_bytes": directory_size(root / "tables"),
        "status": "analyses_complete",
    })
    atomic_json(manifest, root / "manifest.json")
