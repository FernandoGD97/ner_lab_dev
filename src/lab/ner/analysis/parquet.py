"""Compact Arrow schema and metadata for canonical evaluation artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = "2.0.0"
SUPPORTED_SCHEMA_VERSIONS = {"1.0.0", SCHEMA_VERSION}
METADATA_KEY = b"lab.ner.analysis"


def evaluation_schema() -> pa.Schema:
    """Return the versioned, extensible schema of ``evaluation.parquet``."""
    dictionary = pa.dictionary(pa.int16(), pa.string())
    label_list = pa.list_(pa.string())

    fields = [
            pa.field("run_id", dictionary, nullable=False),
            pa.field("event_id", pa.uint64(), nullable=False),
            pa.field("document_id", dictionary, nullable=False),
            pa.field("gold_id", pa.uint64()),
            pa.field("gold_exists", pa.bool_(), nullable=False),
            pa.field("gold_start", pa.int64()),
            pa.field("gold_end", pa.int64()),
            pa.field("gold_text", pa.string()),
            pa.field("gold_label", dictionary),
            pa.field("gold_labels", label_list),
            pa.field("pred_id", pa.uint64()),
            pa.field("pred_exists", pa.bool_(), nullable=False),
            pa.field("pred_start", pa.int64()),
            pa.field("pred_end", pa.int64()),
            pa.field("pred_text", pa.string()),
            pa.field("pred_label", dictionary),
            pa.field("pred_labels", label_list),
            pa.field("pred_confidence", pa.float32()),
            pa.field("strict_outcome", dictionary, nullable=False),
            pa.field("strict_is_tp", pa.bool_(), nullable=False),
            pa.field("strict_is_fp", pa.bool_(), nullable=False),
            pa.field("strict_is_fn", pa.bool_(), nullable=False),
            pa.field("strict_correct", pa.bool_(), nullable=False),
            pa.field("error_pair_id", pa.uint64()),
            pa.field("error_type", dictionary),
        ]
    fields.extend(pa.field(name, dictionary) for name in (
        "error_primary", "error_pair_role", "span_boundary_error", "span_relation",
        "exposure_class", "similarity_bin", "train_frequency_bin", "zero_shot_outcome",
        "entity_word_length_bin", "entity_casing", "tokenization_fragmentation_category",
        "train_nearest_label", "train_nearest_other_label",
    ))
    fields.extend(pa.field(name, pa.bool_()) for name in (
        "error_fragmentation", "error_merging", "error_duplicate_prediction",
        "error_nested_entity", "error_overlapping_entity", "error_label_agreement",
        "span_one_character_offset", "span_leading_whitespace", "span_trailing_whitespace",
        "span_punctuation_included", "span_punctuation_excluded",
        "gold_overlaps_other_gold", "gold_nested", "gold_contains_other_gold",
        "gold_contained_by_other_gold", "annotation_gold_valid", "annotation_pred_valid",
        "train_exact_seen", "train_exact_seen_same_label", "train_exact_seen_other_label",
        "train_exact_label_ambiguous", "train_normalized_seen",
        "train_normalized_seen_same_label", "train_normalized_seen_other_label",
        "train_nearest_same_label", "zero_shot_exact", "zero_shot_normalized",
        "zero_shot_fuzzy", "zero_shot_default", "train_exact_seen_with_pred_label",
        "train_normalized_seen_with_pred_label", "train_nearest_has_pred_label",
        "entity_contains_digit", "entity_contains_decimal", "entity_contains_percentage",
        "entity_contains_hyphen", "entity_contains_slash", "entity_contains_parentheses",
        "entity_contains_punctuation", "entity_contains_greek", "entity_contains_special_symbol",
        "entity_acronym_like", "tokenization_available", "confidence_high_error",
        "confidence_low_correct",
    ))
    fields.extend(pa.field(name, pa.int64()) for name in (
        "diagnostic_gold_id", "diagnostic_pred_id", "span_intersection", "span_start_delta",
        "span_end_delta", "span_length_delta", "span_abs_start_delta", "span_abs_end_delta",
        "gold_overlap_depth", "train_exact_frequency", "train_normalized_frequency",
        "entity_character_length", "entity_whitespace_token_count", "tokenization_subtoken_count",
    ))
    fields.extend(pa.field(name, pa.float32()) for name in (
        "span_iou", "train_nearest_similarity", "train_nearest_same_label_similarity",
        "train_nearest_other_label_similarity", "tokenization_subtoken_to_word_ratio",
    ))
    fields.extend(pa.field(name, pa.string()) for name in (
        "train_normalized_text", "train_nearest_text", "train_nearest_same_label_text",
        "train_nearest_other_label_text",
    ))
    fields.extend(pa.field(name, label_list) for name in (
        "annotation_gold_issues", "annotation_pred_issues", "train_exact_labels",
        "train_normalized_labels",
    ))
    return pa.schema(fields)


def write_evaluation_parquet(
    events: pd.DataFrame,
    path: str | Path,
    metadata: dict[str, Any],
) -> Path:
    """Atomically write canonical events with ZSTD and dictionary encoding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = evaluation_schema().with_metadata(
        {METADATA_KEY: json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()}
    )
    table = pa.Table.from_pandas(events, schema=schema, preserve_index=False, safe=True)
    temporary = path.with_name(f".{path.name}.tmp")

    try:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

    return path


def read_evaluation_parquet(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read canonical events and their run-level metadata."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Evaluation artifact does not exist: {path}")

    table = pq.read_table(path)
    encoded = (table.schema.metadata or {}).get(METADATA_KEY)

    if encoded is None:
        raise ValueError(f"{path} is not a lab NER analysis artifact (metadata is missing).")

    metadata = json.loads(encoded.decode())

    if metadata.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported evaluation schema {metadata.get('schema_version')!r}; "
            f"supported versions are {sorted(SUPPORTED_SCHEMA_VERSIONS)}."
        )

    return table.to_pandas(), metadata
