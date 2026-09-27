"""NER dataset cartography, forgetting events, and label transitions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def training_dynamics(
    observations: pd.DataFrame, easy_quantile: float = 0.67, hard_quantile: float = 0.33
) -> pd.DataFrame:
    """Summarize raw longitudinal uncertainty and attach data-derived cartography regions."""
    confidence_column = "gold_confidence" if "gold_confidence" in observations else "confidence"
    required = {"observation_id", "checkpoint", confidence_column, "entropy", "correct",
                "gold_vs_second_best_margin", "predicted_label"}
    missing = required - set(observations)
    if missing:
        raise ValueError(f"Training dynamics missing columns: {sorted(missing)}")
    rows = []
    for observation_id, group in observations.groupby("observation_id", sort=True):
        group = group.sort_values("checkpoint_order", kind="stable") if "checkpoint_order" in group else group
        correct = group["correct"].fillna(False).astype(bool).to_numpy()
        labels = group["predicted_label"].astype(str).to_numpy()
        rows.append({
            "observation_id": str(observation_id),
            "mean_confidence": float(group[confidence_column].mean()),
            "confidence_variability": float(group[confidence_column].std(ddof=0)),
            "correctness_frequency": float(correct.mean()),
            "mean_entropy": float(group["entropy"].mean()),
            "mean_margin": float(group["gold_vs_second_best_margin"].mean()),
            "prediction_stability": float(np.mean(labels[1:] == labels[:-1])) if len(labels) > 1 else 1.0,
            "sample_count": len(group),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    low = frame["mean_confidence"].quantile(hard_quantile)
    high = frame["mean_confidence"].quantile(easy_quantile)
    variability = frame["confidence_variability"].median()
    frame["cartography_region"] = np.select(
        [
            (frame["mean_confidence"] >= high) & (frame["correctness_frequency"] >= easy_quantile),
            (frame["mean_confidence"] <= low) & (frame["correctness_frequency"] <= hard_quantile),
            frame["confidence_variability"] >= variability,
        ], ["easy_to_learn", "hard_to_learn", "ambiguous"], default="middle"
    )
    frame["easy_threshold"] = high
    frame["hard_threshold"] = low
    return frame


def forgetting_events(observations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for observation_id, group in observations.groupby("observation_id", sort=True):
        group = group.sort_values("checkpoint_order", kind="stable")
        correct = group["correct"].fillna(False).astype(bool).to_numpy()
        checkpoints = group["checkpoint"].astype(str).to_numpy()
        predicted = group["predicted_label"].astype(str).to_numpy()
        transitions = np.flatnonzero(correct[:-1] & ~correct[1:]) + 1 if len(correct) > 1 else np.array([], int)
        correct_positions = np.flatnonzero(correct)
        stable = next((i for i in correct_positions if correct[i:].all()), None)
        rows.append({
            "observation_id": str(observation_id),
            "first_correct_checkpoint": checkpoints[correct_positions[0]] if len(correct_positions) else None,
            "first_stable_learning_checkpoint": checkpoints[stable] if stable is not None else None,
            "forgetting_events": len(transitions),
            "last_forgetting_checkpoint": checkpoints[transitions[-1]] if len(transitions) else None,
            "final_correct": bool(correct[-1]),
            "label_changes": int(np.sum(predicted[1:] != predicted[:-1])),
            "stability_after_first_learning": (
                float(correct[correct_positions[0]:].mean()) if len(correct_positions) else 0.0
            ),
        })
    return pd.DataFrame(rows)


def prediction_transitions(observations: pd.DataFrame) -> pd.DataFrame:
    events = []
    for observation_id, group in observations.groupby("observation_id", sort=True):
        group = group.sort_values("checkpoint_order", kind="stable")
        records = list(group.to_dict("records"))
        for previous, current in zip(records, records[1:]):
            before_correct = bool(previous.get("correct") or False)
            after_correct = bool(current.get("correct") or False)
            events.append({
                "checkpoint_from": previous["checkpoint"], "checkpoint_to": current["checkpoint"],
                "label_from": str(previous["predicted_label"]),
                "label_to": str(current["predicted_label"]),
                "correction": not before_correct and after_correct,
                "degradation": before_correct and not after_correct,
                "persistent": str(previous["predicted_label"]) == str(current["predicted_label"]),
                "observation_id": str(observation_id),
            })
    detail = pd.DataFrame(events)
    if detail.empty:
        return detail
    keys = ["checkpoint_from", "checkpoint_to", "label_from", "label_to"]
    aggregate = detail.groupby(keys, sort=True, dropna=False).agg(
        transition_count=("observation_id", "size"), corrections=("correction", "sum"),
        degradations=("degradation", "sum"), persistence_count=("persistent", "sum"),
    ).reset_index()
    totals = aggregate.groupby(keys[:-1])["transition_count"].transform("sum")
    aggregate["transition_probability"] = aggregate["transition_count"] / totals
    aggregate["persistence_rate"] = aggregate["persistence_count"] / aggregate["transition_count"]
    return aggregate
