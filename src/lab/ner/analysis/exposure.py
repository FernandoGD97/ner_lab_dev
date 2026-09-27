"""Training exposure and efficient nearest-neighbour mention similarity."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable

import pandas as pd

from lab.ner.analysis.config import AnalysisConfig, frequency_bin, similarity_bin


@dataclass(frozen=True)
class Neighbour:
    text: str | None
    label: str | None
    similarity: float | None


class ExposureIndex:
    """Deduplicated training mention index with cached RapidFuzz lookups."""

    def __init__(self, training: pd.DataFrame, config: AnalysisConfig) -> None:
        self.config = config
        self.exact_counts = Counter(str(value) for value in training["text"])
        self.exact_labels: dict[str, set[str]] = defaultdict(set)
        self.normalized_counts: Counter[str] = Counter()
        self.normalized_labels: dict[str, set[str]] = defaultdict(set)
        self.entries: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str]] = set()

        for row in training.itertuples(index=False):
            text, label = str(row.text), str(row.label)
            normalized = config.normalization.normalize(text)
            self.exact_labels[text].add(label)
            self.normalized_counts[normalized] += 1
            self.normalized_labels[normalized].add(label)
            key = (normalized, label)
            if key not in seen:
                self.entries.append((normalized, text, label))
                seen.add(key)

        self.entries.sort(key=lambda item: (item[0], item[1], item[2]))
        self._cache: dict[tuple[str, str | None, bool], Neighbour] = {}
        self.cache_hits = 0
        self.backend, self._score = _resolve_scorer(config.fuzzy_scorer)

    @property
    def unique_training_mentions(self) -> int:
        return len(self.entries)

    def nearest(self, text: str, label: str | None = None, *, other: bool = False) -> Neighbour:
        normalized = self.config.normalization.normalize(text)
        key = (normalized, label, other)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]

        candidates = [
            entry for entry in self.entries
            if label is None or ((entry[2] != label) if other else (entry[2] == label))
        ]
        if not candidates:
            result = Neighbour(None, None, None)
        else:
            scored = [
                (round(float(self._score(normalized, entry[0])) / 100.0, 6), entry)
                for entry in candidates
            ]
            score, (_, original, candidate_label) = min(
                scored, key=lambda item: (-item[0], item[1][1], item[1][2], item[1][0])
            )
            result = Neighbour(original, candidate_label, score)

        self._cache[key] = result
        return result

    def describe(self, text: str, label: str) -> dict:
        normalized = self.config.normalization.normalize(text)
        exact_labels = sorted(self.exact_labels.get(text, set()))
        normalized_labels = sorted(self.normalized_labels.get(normalized, set()))
        nearest = self.nearest(text)
        same = self.nearest(text, label)
        other = self.nearest(text, label, other=True)
        threshold = threshold_on_rapidfuzz_scale(self.config.fuzzy_threshold)
        exact_seen = bool(exact_labels)
        normalized_seen = bool(normalized_labels)
        fuzzy_seen = nearest.similarity is not None and nearest.similarity * 100.0 >= threshold

        if label in exact_labels:
            exposure = "EXACT_SEEN_SAME_LABEL"
        elif exact_seen:
            exposure = "EXACT_SEEN_OTHER_LABEL"
        elif label in normalized_labels:
            exposure = "NORMALIZED_SEEN_SAME_LABEL"
        elif normalized_seen:
            exposure = "NORMALIZED_SEEN_OTHER_LABEL"
        elif same.similarity is not None and same.similarity * 100.0 >= threshold:
            exposure = "FUZZY_SEEN_SAME_LABEL"
        elif fuzzy_seen:
            exposure = "FUZZY_SEEN_OTHER_LABEL"
        else:
            exposure = "UNSEEN"

        zero_exact = not exact_seen
        zero_normalized = not normalized_seen
        zero_fuzzy = not fuzzy_seen
        zero_default = {
            "exact_unseen": zero_exact,
            "normalized_unseen": zero_normalized,
            "fuzzy_unseen": zero_fuzzy,
        }[self.config.zero_shot_definition]

        return {
            "train_exact_seen": exact_seen,
            "train_exact_seen_same_label": label in exact_labels,
            "train_exact_seen_other_label": any(value != label for value in exact_labels),
            "train_exact_frequency": int(self.exact_counts[text]),
            "train_exact_labels": exact_labels,
            "train_exact_label_ambiguous": len(exact_labels) > 1,
            "train_normalized_text": normalized,
            "train_normalized_seen": normalized_seen,
            "train_normalized_seen_same_label": label in normalized_labels,
            "train_normalized_seen_other_label": any(value != label for value in normalized_labels),
            "train_normalized_frequency": int(self.normalized_counts[normalized]),
            "train_normalized_labels": normalized_labels,
            "train_nearest_text": nearest.text,
            "train_nearest_label": nearest.label,
            "train_nearest_similarity": nearest.similarity,
            "train_nearest_same_label": nearest.label == label if nearest.label is not None else None,
            "train_nearest_same_label_text": same.text,
            "train_nearest_same_label_similarity": same.similarity,
            "train_nearest_other_label_text": other.text,
            "train_nearest_other_label": other.label,
            "train_nearest_other_label_similarity": other.similarity,
            "exposure_class": exposure,
            "similarity_bin": similarity_bin(nearest.similarity, self.config.similarity_bins),
            "train_frequency_bin": frequency_bin(int(self.exact_counts[text]), self.config.frequency_bins),
            "zero_shot_exact": zero_exact,
            "zero_shot_normalized": zero_normalized,
            "zero_shot_fuzzy": zero_fuzzy,
            "zero_shot_default": zero_default,
        }


def threshold_on_rapidfuzz_scale(threshold: float) -> float:
    """Convert the sole public 0-1 threshold representation to RapidFuzz's scale."""
    if not 0 <= threshold <= 1:
        raise ValueError("fuzzy threshold must be on the 0.0-1.0 scale.")
    return threshold * 100.0


def _resolve_scorer(name: str) -> tuple[str, Callable[[str, str], float]]:
    try:
        from rapidfuzz import fuzz
    except ImportError:
        # Packaging declares RapidFuzz. This deterministic compatibility path
        # keeps base-install artifact loading usable in deliberately minimal
        # environments; metadata makes the backend explicit.
        return "difflib-compatibility", lambda first, second: SequenceMatcher(
            None, first, second, autojunk=False
        ).ratio() * 100.0

    return "rapidfuzz", getattr(fuzz, name)
