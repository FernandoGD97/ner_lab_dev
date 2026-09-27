"""Annotation integrity, entity morphology, overlap, and optional tokenization features."""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


def resolve_documents(value: pd.DataFrame | dict[str, str] | str | Path | None) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(key): str(text) for key, text in value.items()}
    frame = value if isinstance(value, pd.DataFrame) else pd.read_parquet(value, columns=["doc_id", "text"])
    if not {"doc_id", "text"}.issubset(frame.columns):
        raise ValueError("Document data requires doc_id and text columns.")
    return {str(row.doc_id): str(row.text) for row in frame.itertuples(index=False)}


def audit_annotations(spans: pd.DataFrame, documents: dict[str, str]) -> pd.DataFrame:
    """Audit annotations without changing them or classifying them as model errors."""
    duplicate_keys = Counter()
    labels_by_boundary: dict[tuple, set[str]] = defaultdict(set)
    for row in spans.itertuples(index=False):
        key = (str(row.filename), int(row.start_span), int(row.end_span), str(row.label), str(row.text))
        duplicate_keys[key] += 1
        labels_by_boundary[key[:3]].add(str(row.label))
    records = []
    for row in spans.itertuples(index=False):
        doc_id, start, end, surface = str(row.filename), int(row.start_span), int(row.end_span), str(row.text)
        issues = []
        if start < 0:
            issues.append("NEGATIVE_OFFSET")
        if end <= start:
            issues.append("ZERO_OR_NEGATIVE_LENGTH")
        document = documents.get(doc_id)
        if document is None:
            issues.append("DOCUMENT_UNAVAILABLE")
        elif start > len(document) or end > len(document):
            issues.append("OUT_OF_RANGE")
        else:
            actual = document[start:end]
            if actual != surface:
                issues.append("SURFACE_MISMATCH")
                if any(document[max(0, start + shift):min(len(document), end + shift)] == surface for shift in (-1, 1)):
                    issues.append("OFF_BY_ONE")
                if actual.replace("\r\n", "\n") == surface.replace("\r\n", "\n"):
                    issues.append("CRLF_LF_MISMATCH")
                elif actual.replace("\n", "") == surface.replace("\n", ""):
                    issues.append("NEWLINE_MISMATCH")
                if unicodedata.normalize("NFKC", actual) == unicodedata.normalize("NFKC", surface):
                    issues.append("UNICODE_NORMALIZATION_MISMATCH")
                if actual.replace("\u00a0", " ") == surface.replace("\u00a0", " "):
                    issues.append("NON_BREAKING_SPACE")
        if surface[:1].isspace():
            issues.append("LEADING_WHITESPACE")
        if surface[-1:].isspace():
            issues.append("TRAILING_WHITESPACE")
        full_key = (doc_id, start, end, str(row.label), surface)
        if duplicate_keys[full_key] > 1:
            issues.append("DUPLICATE_ANNOTATION")
        if len(labels_by_boundary[(doc_id, start, end)]) > 1:
            issues.append("CONFLICTING_LABELS")
        substantive = [issue for issue in issues if issue != "DOCUMENT_UNAVAILABLE"]
        records.append({"annotation_valid": not substantive, "annotation_issues": sorted(set(issues))})
    return pd.DataFrame(records, columns=["annotation_valid", "annotation_issues"])


def enrich_features(
    events: pd.DataFrame,
    gold: pd.DataFrame,
    predicted: pd.DataFrame,
    *,
    evaluation_documents: dict[str, str],
    tokenizer=None,
    acronym_max_length: int = 12,
    acronym_min_uppercase_ratio: float = 0.60,
    high_confidence_threshold: float = 0.90,
    low_confidence_threshold: float = 0.50,
) -> pd.DataFrame:
    enriched = events.copy()
    gold_audit = audit_annotations(gold, evaluation_documents)
    pred_audit = audit_annotations(predicted, evaluation_documents)
    for name in ("annotation_gold_valid", "annotation_pred_valid"):
        enriched[name] = None
    for name in ("annotation_gold_issues", "annotation_pred_issues"):
        enriched[name] = None
    feature_names = list(entity_features("", tokenizer, acronym_max_length, acronym_min_uppercase_ratio))
    for name in feature_names:
        enriched[name] = None
    for index, event in enriched.iterrows():
        if event.gold_exists:
            audit = gold_audit.iloc[int(event.gold_id)]
            enriched.at[index, "annotation_gold_valid"] = bool(audit.annotation_valid)
            enriched.at[index, "annotation_gold_issues"] = audit.annotation_issues
        if event.pred_exists:
            audit = pred_audit.iloc[int(event.pred_id)]
            enriched.at[index, "annotation_pred_valid"] = bool(audit.annotation_valid)
            enriched.at[index, "annotation_pred_issues"] = audit.annotation_issues
        text = event.gold_text if event.gold_exists else event.pred_text
        for name, value in entity_features(
            str(text), tokenizer, acronym_max_length, acronym_min_uppercase_ratio
        ).items():
            enriched.at[index, name] = value
    confidence = pd.to_numeric(enriched["pred_confidence"], errors="coerce")
    enriched["confidence_high_error"] = (
        confidence.ge(high_confidence_threshold) & ~enriched["strict_correct"]
    )
    enriched["confidence_low_correct"] = confidence.le(low_confidence_threshold) & enriched["strict_correct"]
    return enriched


def entity_features(text: str, tokenizer, acronym_max_length: int, uppercase_ratio: float) -> dict[str, Any]:
    words = text.split()
    letters = [character for character in text if character.isalpha()]
    upper_ratio = sum(character.isupper() for character in letters) / len(letters) if letters else 0.0
    acronym = bool(
        letters and len(text) <= acronym_max_length and len(words) <= 3
        and upper_ratio >= uppercase_ratio and not text.islower()
    )
    subtokens = None
    if tokenizer is not None:
        encoded = tokenizer(text, add_special_tokens=False)
        subtokens = len(encoded["input_ids"])
    ratio = subtokens / max(1, len(words)) if subtokens is not None else None
    return {
        "entity_character_length": len(text),
        "entity_whitespace_token_count": len(words),
        "entity_word_length_bin": str(len(words)) if len(words) < 5 else "5+",
        "entity_contains_digit": any(c.isdigit() for c in text),
        "entity_contains_decimal": bool(re.search(r"\d+[.,]\d+", text)),
        "entity_contains_percentage": "%" in text,
        "entity_contains_hyphen": bool(re.search(r"[-‐‑‒–—]", text)),
        "entity_contains_slash": "/" in text,
        "entity_contains_parentheses": any(c in "()[]{}" for c in text),
        "entity_contains_punctuation": any(c in string.punctuation for c in text),
        "entity_contains_greek": any("GREEK" in unicodedata.name(c, "") for c in text),
        "entity_contains_special_symbol": any(unicodedata.category(c).startswith("S") for c in text),
        "entity_casing": _casing(text),
        "entity_acronym_like": acronym,
        "tokenization_subtoken_count": subtokens,
        "tokenization_subtoken_to_word_ratio": ratio,
        "tokenization_fragmentation_category": _fragmentation_bin(subtokens),
        "tokenization_available": tokenizer is not None,
    }


def _casing(text: str) -> str:
    if text.isupper():
        return "UPPERCASE"
    if text.islower():
        return "LOWERCASE"
    if text.istitle():
        return "TITLECASE"
    return "MIXEDCASE" if any(c.isalpha() for c in text) else "UNCASED"


def _fragmentation_bin(count: int | None) -> str | None:
    if count is None:
        return None
    if count <= 2:
        return str(count)
    if count <= 4:
        return "3-4"
    if count <= 8:
        return "5-8"
    return "9+"
