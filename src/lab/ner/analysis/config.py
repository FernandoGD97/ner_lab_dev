"""Configuration and text normalization for diagnostic analysis."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class NormalizationConfig:
    """Explicit mention normalization; destructive options are disabled by default."""

    unicode_form: str | None = "NFKC"
    strip: bool = True
    casefold: bool = True
    collapse_whitespace: bool = True
    normalize_punctuation: bool = False
    remove_diacritics: bool = False

    def normalize(self, text: str) -> str:
        value = str(text)

        if self.unicode_form:
            value = unicodedata.normalize(self.unicode_form, value)

        if self.remove_diacritics:
            value = "".join(
                character
                for character in unicodedata.normalize("NFD", value)
                if unicodedata.category(character) != "Mn"
            )

        if self.normalize_punctuation:
            value = "".join(" " if unicodedata.category(c).startswith("P") else c for c in value)

        if self.casefold:
            value = value.casefold()

        if self.collapse_whitespace:
            value = re.sub(r"\s+", " ", value)

        return value.strip() if self.strip else value


DEFAULT_SIMILARITY_BINS = (
    ("EXACT", 1.0, 1.0),
    ("0.95-<1.00", 0.95, 1.0),
    ("0.90-<0.95", 0.90, 0.95),
    ("0.85-<0.90", 0.85, 0.90),
    ("0.80-<0.85", 0.80, 0.85),
    ("<0.80", 0.0, 0.80),
)


@dataclass(frozen=True)
class AnalysisConfig:
    """Stable public settings for Part-2 analysis."""

    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    fuzzy_threshold: float = 0.80
    fuzzy_scorer: str = "WRatio"
    zero_shot_definition: str = "fuzzy_unseen"
    similarity_bins: tuple[tuple[str, float, float], ...] = DEFAULT_SIMILARITY_BINS
    frequency_bins: tuple[tuple[str, int, int | None], ...] = (
        ("0", 0, 0), ("1", 1, 1), ("2-5", 2, 5),
        ("6-10", 6, 10), ("11-20", 11, 20), (">20", 21, None),
    )
    confidence_thresholds: tuple[float, ...] = ()
    high_confidence_threshold: float = 0.90
    low_confidence_threshold: float = 0.50
    acronym_max_length: int = 12
    acronym_min_uppercase_ratio: float = 0.60
    minimum_intersection_support: int = 2

    def __post_init__(self) -> None:
        if not 0 <= self.fuzzy_threshold <= 1:
            raise ValueError("fuzzy_threshold must be on the 0.0-1.0 scale.")
        if self.fuzzy_scorer not in {"ratio", "WRatio", "partial_ratio", "token_sort_ratio", "token_set_ratio"}:
            raise ValueError(f"Unsupported fuzzy_scorer: {self.fuzzy_scorer}")
        if self.zero_shot_definition not in {"exact_unseen", "normalized_unseen", "fuzzy_unseen"}:
            raise ValueError(f"Unsupported zero_shot_definition: {self.zero_shot_definition}")

    def to_dict(self) -> dict:
        return asdict(self)


def similarity_bin(value: float | None, bins=DEFAULT_SIMILARITY_BINS) -> str | None:
    if value is None:
        return None
    for name, lower, upper in bins:
        if name == "EXACT" and value == 1.0:
            return name
        if name != "EXACT" and lower <= value < upper:
            return name
    return None


def frequency_bin(value: int, bins: tuple[tuple[str, int, int | None], ...]) -> str:
    for name, lower, upper in bins:
        if value >= lower and (upper is None or value <= upper):
            return name
    raise ValueError(f"No training-frequency bin covers {value}.")
