"""STRICT diagnostic oracle upper bounds derived from canonical events."""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from lab.core.scoring import safe_f1


@dataclass(frozen=True)
class OracleSelection:
    """Disjoint strict contributions changed by one counterfactual scenario."""

    pair_ids: frozenset[int] = frozenset()
    missed_event_ids: frozenset[int] = frozenset()
    spurious_event_ids: frozenset[int] = frozenset()

    def union(self, *others: "OracleSelection") -> "OracleSelection":
        return OracleSelection(
            self.pair_ids.union(*(other.pair_ids for other in others)),
            self.missed_event_ids.union(*(other.missed_event_ids for other in others)),
            self.spurious_event_ids.union(*(other.spurious_event_ids for other in others)),
        )


def oracle_error_budget(events: pd.DataFrame, minimum_confusion_support: int = 1) -> pd.DataFrame:
    """Build all documented single-family and combined STRICT upper bounds."""
    selections = oracle_selections(events, minimum_confusion_support)
    rows = [oracle_result(events, name, selection, kind) for name, selection, kind in selections]
    return pd.DataFrame(rows).sort_values(
        ["scenario_kind", "delta_f1", "error_family"], ascending=[True, False, True]
    ).reset_index(drop=True)


def oracle_selections(events: pd.DataFrame, minimum_confusion_support: int = 1):
    gold_pairs = events[events["error_pair_role"] == "GOLD"]
    pair = lambda categories: _pair_selection(gold_pairs[gold_pairs["error_primary"].isin(categories)])
    boundary = pair({"BOUNDARY_ERROR"})
    label = pair({"LABEL_ERROR"})
    both = pair({"BOUNDARY_AND_LABEL_ERROR"})
    misses = _miss_selection(events[events["error_primary"] == "MISSED"])
    exact_misses = _miss_selection(events[(events["error_primary"] == "MISSED") & events["train_exact_seen"].fillna(False)])
    normalized_misses = _miss_selection(events[(events["error_primary"] == "MISSED") & events["train_normalized_seen"].fillna(False)])
    fuzzy_seen_misses = _miss_selection(events[(events["error_primary"] == "MISSED") & ~events["zero_shot_fuzzy"].fillna(True)])
    zero_misses = _miss_selection(events[(events["error_primary"] == "MISSED") & events["zero_shot_fuzzy"].fillna(False)])
    spurious = _spurious_selection(events[events["error_primary"] == "SPURIOUS"])
    rows = [
        ("Boundary correction", boundary, "single_family"),
        ("Label correction", label, "single_family"),
        ("Boundary + label correction", both, "single_family"),
        ("All missed entities", misses, "single_family"),
        ("Exact-seen miss recovery", exact_misses, "single_family"),
        ("Normalized-seen miss recovery", normalized_misses, "single_family"),
        ("Fuzzy-seen miss recovery", fuzzy_seen_misses, "single_family"),
        ("Fuzzy-zero-shot miss recovery", zero_misses, "single_family"),
        ("Spurious-prediction removal", spurious, "single_family"),
        ("All boundary-related errors fixed", boundary.union(both), "combined"),
        ("All classification-related errors fixed", label.union(both), "combined"),
        ("All seen misses recovered", exact_misses.union(normalized_misses, fuzzy_seen_misses), "combined"),
        ("All generalization misses recovered", zero_misses, "combined"),
        ("All diagnostic errors corrected", boundary.union(label, both, misses, spurious), "combined"),
    ]
    labels = sorted(set(events["gold_label"].dropna().astype(str)) | set(events["pred_label"].dropna().astype(str)))
    for entity_label in labels:
        matching_pairs = gold_pairs[gold_pairs["gold_label"] == entity_label]
        selection = _pair_selection(matching_pairs).union(
            _miss_selection(events[(events["error_primary"] == "MISSED") & (events["gold_label"] == entity_label)]),
            _spurious_selection(events[(events["error_primary"] == "SPURIOUS") & (events["pred_label"] == entity_label)]),
        )
        rows.append((f"Label: {entity_label}", selection, "per_label"))
    label_errors = gold_pairs[gold_pairs["error_primary"] == "LABEL_ERROR"]
    for gold_label in sorted(label_errors["gold_label"].dropna().astype(str).unique()):
        subset = label_errors[label_errors["gold_label"] == gold_label]
        for pred_label in sorted({_paired_label(events, row) for row in subset.itertuples()}):
            selected = subset[subset.apply(lambda row: _paired_label(events, row) == pred_label, axis=1)]
            if len(selected) >= minimum_confusion_support:
                rows.append((f"Confusion: {gold_label} -> {pred_label}", _pair_selection(selected), "confusion_pair"))
    return rows


def oracle_result(
    events: pd.DataFrame,
    name: str,
    selection: OracleSelection,
    scenario_kind: str = "single_family",
) -> dict:
    """Apply a selection to official contributions and report the upper bound."""
    current = _counts(events)
    corrected_pairs = len(selection.pair_ids)
    recovered_misses = len(selection.missed_event_ids)
    removed_spurious = len(selection.spurious_event_ids)
    oracle = {
        "tp": current["tp"] + corrected_pairs + recovered_misses,
        "fp": current["fp"] - corrected_pairs - removed_spurious,
        "fn": current["fn"] - corrected_pairs - recovered_misses,
    }
    if min(oracle.values()) < 0:
        raise ValueError(f"Oracle {name!r} removes more contributions than exist.")
    baseline_score, oracle_score = safe_f1(**current), safe_f1(**oracle)
    affected = _affected(events, selection)
    baseline_error = 1.0 - float(baseline_score["f1"])
    relative = (
        (float(oracle_score["f1"]) - float(baseline_score["f1"])) / baseline_error
        if baseline_error else 0.0
    )
    return {
        "error_family": name,
        "scenario_kind": scenario_kind,
        "event_count": corrected_pairs * 2 + recovered_misses + removed_spurious,
        "affected_events": corrected_pairs + recovered_misses + removed_spurious,
        "support": int(affected["gold_exists"].sum()),
        "percentage_of_all_errors": _percentage_of_errors(events, selection),
        "percentage_of_label_errors": _percentage_of_label_errors(events, selection),
        "gold_entities_affected": int(affected["gold_exists"].sum()),
        "prediction_entities_affected": int(affected["pred_exists"].sum()),
        **{f"current_{key}": value for key, value in baseline_score.items()},
        **{f"oracle_{key}": value for key, value in oracle_score.items()},
        "delta_precision": round(float(oracle_score["precision"] - baseline_score["precision"]), 4),
        "delta_recall": round(float(oracle_score["recall"] - baseline_score["recall"]), 4),
        "delta_f1": round(float(oracle_score["f1"] - baseline_score["f1"]), 4),
        "relative_error_reduction": round(float(relative), 4),
        "main_labels": _main_labels(affected),
        "seen_fraction": _fraction(affected, ~affected["zero_shot_default"].fillna(True)),
        "zero_shot_fraction": _fraction(affected, affected["zero_shot_default"].fillna(False)),
        "mean_confidence": _mean_confidence(affected),
    }


def _counts(events):
    return {
        "tp": int(events["strict_is_tp"].sum()),
        "fp": int(events["strict_is_fp"].sum()),
        "fn": int(events["strict_is_fn"].sum()),
    }


def _pair_selection(rows):
    return OracleSelection(pair_ids=frozenset(int(value) for value in rows["error_pair_id"].dropna()))


def _miss_selection(rows):
    return OracleSelection(missed_event_ids=frozenset(int(value) for value in rows["event_id"]))


def _spurious_selection(rows):
    return OracleSelection(spurious_event_ids=frozenset(int(value) for value in rows["event_id"]))


def _affected(events, selection):
    return events[
        events["error_pair_id"].isin(selection.pair_ids)
        | events["event_id"].isin(selection.missed_event_ids | selection.spurious_event_ids)
    ]


def _percentage_of_errors(events, selection):
    total = int((events["strict_is_fp"] | events["strict_is_fn"]).sum())
    affected = len(_affected(events, selection))
    return float(affected / total) if total else 0.0


def _percentage_of_label_errors(events, selection):
    label_pairs = set(
        int(value) for value in events.loc[
            (events["error_pair_role"] == "GOLD") & (events["error_primary"] == "LABEL_ERROR"),
            "error_pair_id",
        ].dropna()
    )
    selected = len(selection.pair_ids & label_pairs)
    return float(selected / len(label_pairs)) if label_pairs else 0.0


def _paired_label(events, row):
    match = events[(events["error_pair_id"] == row.error_pair_id) & (events["error_pair_role"] == "PRED")]
    return str(match.iloc[0]["pred_label"])


def _main_labels(events):
    labels = events.apply(lambda row: row.gold_label if row.gold_exists else row.pred_label, axis=1)
    return [str(value) for value in labels.value_counts().head(3).index]


def _fraction(events, mask):
    gold = events[events["gold_exists"]]
    return float(mask.loc[gold.index].mean()) if len(gold) else None


def _mean_confidence(events):
    values = pd.to_numeric(events["pred_confidence"], errors="coerce").dropna()
    return float(values.mean()) if len(values) else None
