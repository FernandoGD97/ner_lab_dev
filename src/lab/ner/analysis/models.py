"""Data contracts for persistent NER evaluation analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


STRICT_TP = "TP"
STRICT_FP = "FP"
STRICT_FN = "FN"
STRICT_OUTCOMES = (STRICT_TP, STRICT_FP, STRICT_FN)


@dataclass(frozen=True)
class AnalysisResult:
    """The canonical event table, official metrics, and persisted artifact."""

    path: Path
    events: pd.DataFrame
    metrics: dict[str, float | int]
    metadata: dict[str, Any]

    @property
    def strict_counts(self) -> dict[str, int]:
        """Return TP/FP/FN totals represented by the event rows."""
        return {
            outcome.lower(): int((self.events["strict_outcome"] == outcome).sum())
            for outcome in STRICT_OUTCOMES
        }

    def zero_shot(self) -> pd.DataFrame:
        """Gold-bearing events under the configured zero-shot definition."""
        return self._gold(self.events["zero_shot_default"].fillna(False))

    def zero_shot_correct(self) -> pd.DataFrame:
        return self._gold(self.events["zero_shot_default"].fillna(False) & self.events["strict_correct"])

    def zero_shot_errors(self) -> pd.DataFrame:
        return self._gold(self.events["zero_shot_default"].fillna(False) & ~self.events["strict_correct"])

    def boundary_errors(self) -> pd.DataFrame:
        return self._diagnostic({"BOUNDARY_ERROR", "BOUNDARY_AND_LABEL_ERROR"})

    def label_errors(self) -> pd.DataFrame:
        return self._diagnostic({"LABEL_ERROR", "BOUNDARY_AND_LABEL_ERROR"})

    def by_label(self, label: str) -> pd.DataFrame:
        mask = (self.events["gold_label"] == label) | (self.events["pred_label"] == label)
        return self.events.loc[mask].copy()

    def by_similarity(self, similarity_bin: str) -> pd.DataFrame:
        return self._gold(self.events["similarity_bin"] == similarity_bin)

    def by_training_frequency(self, frequency_bin: str) -> pd.DataFrame:
        return self._gold(self.events["train_frequency_bin"] == frequency_bin)

    def overlapping_entities(self) -> pd.DataFrame:
        return self._gold(self.events["gold_overlaps_other_gold"].fillna(False))

    def acronyms(self) -> pd.DataFrame:
        return self._gold(self.events["entity_acronym_like"].fillna(False))

    def high_confidence_errors(self) -> pd.DataFrame:
        return self.events.loc[self.events["confidence_high_error"].fillna(False)].copy()

    def subgroup_metrics(self, family: str | None = None) -> pd.DataFrame:
        rows = self.metadata.get("aggregates", {}).get("subgroups", [])
        frame = pd.DataFrame(rows)
        return frame if family is None or frame.empty else frame[frame["family"] == family].reset_index(drop=True)

    def summary(self) -> dict[str, Any]:
        return self.metadata.get("aggregates", {}).get("summary", {})

    def oracle_error_budget(self, minimum_confusion_support: int = 1) -> pd.DataFrame:
        """Return STRICT diagnostic upper bounds without rerunning inference."""
        from lab.ner.analysis.oracles import oracle_error_budget

        return oracle_error_budget(self.events, minimum_confusion_support)

    def _gold(self, mask) -> pd.DataFrame:
        return self.events.loc[mask & self.events["gold_exists"]].copy()

    def _diagnostic(self, categories: set[str]) -> pd.DataFrame:
        mask = self.events["error_primary"].isin(categories)
        # One canonical gold-side row per relationship avoids double-counting.
        return self.events.loc[mask & (self.events["error_pair_role"] == "GOLD")].copy()
