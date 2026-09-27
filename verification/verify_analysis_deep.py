"""Verify Part-2 diagnostics, exposure, features, and full subgroup evaluation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from _harness import Checks, run
from lab.ner.analysis import (
    AnalysisConfig,
    ExposureIndex,
    NormalizationConfig,
    analyze_evaluation,
    audit_annotations,
    threshold_on_rapidfuzz_scale,
)

C = ["filename", "label", "start_span", "end_span", "text"]


def frame(rows, scored=False):
    return pd.DataFrame(rows, columns=[*C, "score"] if scored else C)


class TinyTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.replace("-", " ").split()) * 2))}


def verify_exposure(checks: Checks) -> None:
    training = frame([
        ["t", "D", 0, 5, "Alpha"], ["t", "D", 6, 11, "Alpha"],
        ["t", "X", 12, 17, "Alpha"], ["t", "D", 18, 31, "heart  failure"],
    ])
    index = ExposureIndex(training, AnalysisConfig())
    exact = index.describe("Alpha", "D")
    checks.equal("same exact surface and label is seen", exact["exposure_class"], "EXACT_SEEN_SAME_LABEL")
    checks.equal("exact frequencies retain duplicates", exact["train_exact_frequency"], 3)
    checks.check("training label ambiguity is retained", exact["train_exact_label_ambiguous"])
    checks.check("same surface under another label is retained", exact["train_exact_seen_other_label"])
    normalized = index.describe("  HEART failure ", "D")
    checks.equal("normalization is independently classified", normalized["exposure_class"], "NORMALIZED_SEEN_SAME_LABEL")
    checks.equal("default normalization is explicit", NormalizationConfig().normalize("  HéART\t Failure "), "héart failure")
    checks.equal("public threshold converts once", threshold_on_rapidfuzz_scale(0.80), 80.0)
    checks.raises("threshold scale is enforced", ValueError, threshold_on_rapidfuzz_scale, 80.0, match="0.0-1.0")

    simple = ExposureIndex(frame([["t", "D", 0, 1, "candidate"]]), AnalysisConfig(fuzzy_threshold=0.8))
    for raw, expected in ((80.0, False), (79.999, True), (80.001, False)):
        simple._cache.clear()
        simple._score = lambda first, second, value=raw: value
        result = simple.describe("query", "D")
        checks.equal(f"threshold behavior at score {raw}", result["zero_shot_fuzzy"], expected)
    simple._cache.clear()
    simple._score = lambda first, second: 81.0
    simple.describe("repeat", "D")
    simple.describe("repeat", "D")
    checks.check("repeated mentions reuse cached neighbours", simple.cache_hits > 0)
    checks.equal("training candidates are deduplicated", index.unique_training_mentions, 3)


def verify_diagnostics(checks: Checks, root: Path) -> None:
    gold = frame([
        ["d", "D", 0, 10, "abcdefghij"],
        ["d", "X", 20, 25, "klmno"],
        ["d", "N", 40, 50, "qrstuvwxyz"],
        ["d", "N", 42, 46, "stuv"],
    ])
    predicted = frame([
        ["d", "D", 0, 8, "abcdefgh", .8],
        ["d", "Y", 20, 25, "klmno", .95],
        ["d", "M", 39, 50, "pqrstuvwxyz", .7],
        ["d", "N", 42, 46, "stuv", .4],
    ], scored=True)
    result = analyze_evaluation(
        predicted, gold, frame([]), root / "taxonomy", run_id="taxonomy",
        tokenizer=TinyTokenizer(), config=AnalysisConfig(high_confidence_threshold=.9),
    )
    primary = set(result.events["error_primary"])
    checks.check("boundary errors are classified", "BOUNDARY_ERROR" in primary)
    checks.check("label errors are classified", "LABEL_ERROR" in primary)
    checks.check("boundary plus label errors are classified", "BOUNDARY_AND_LABEL_ERROR" in primary)
    boundary = result.boundary_errors()
    checks.check("boundary deltas are persisted", boundary["span_end_delta"].notna().all())
    checks.check("one-character offsets are identified", boundary["span_one_character_offset"].any())
    checks.check("nested gold is queryable", result.events["gold_nested"].fillna(False).any())
    checks.check("high-confidence label errors are queryable", not result.high_confidence_errors().empty)
    checks.check("tokenization is optional but populated when supplied", result.events["tokenization_available"].all())
    confusion = result.metadata["aggregates"]["confusion"]["exact_boundary_label_confusion"]
    checks.equal("single-label confusion isolates equal boundaries", confusion[0]["pred_label"], "Y")

    fragmented = analyze_evaluation(
        frame([["d", "D", 0, 5, "abcde"], ["d", "D", 5, 10, "fghij"]]),
        frame([["d", "D", 0, 10, "abcdefghij"]]), frame([]), root / "fragmented", run_id="f",
    )
    checks.check("fragmentation is an independent flag", fragmented.events["error_fragmentation"].any())
    merged = analyze_evaluation(
        frame([["d", "D", 0, 10, "abcdefghij"]]),
        frame([["d", "D", 0, 5, "abcde"], ["d", "D", 5, 10, "fghij"]]),
        frame([]), root / "merged", run_id="m",
    )
    checks.check("merging is an independent flag", merged.events["error_merging"].any())


def verify_subgroups(checks: Checks, root: Path) -> None:
    training = frame([["t", "D", 0, 5, "alpha"]])
    gold = frame([["d", "D", 0, 5, "alpha"], ["d", "D", 10, 14, "beta"]])
    predicted = frame([["d", "D", 0, 5, "alpha"], ["d", "D", 20, 25, "gamma"]])
    result = analyze_evaluation(predicted, gold, training, root, run_id="groups")
    exact = result.subgroup_metrics("exact_exposure").set_index("value")
    checks.equal("seen subgroup has full official accounting", tuple(exact.loc["true", ["strict_tp", "strict_fp", "strict_fn"]]), (1, 0, 0))
    checks.equal("unseen subgroup includes its own FP denominator", tuple(exact.loc["false", ["strict_tp", "strict_fp", "strict_fn"]]), (0, 1, 1))
    checks.equal("unseen subgroup F1 is official zero", exact.loc["false", "strict_f1"], 0.0)
    checks.check("similarity bins are evaluated", not result.subgroup_metrics("similarity").empty)
    checks.check("frequency groups are evaluated", not result.subgroup_metrics("training_frequency").empty)
    checks.check("entity labels are evaluated", not result.subgroup_metrics("label").empty)
    checks.check("acronym groups are evaluated", not result.subgroup_metrics("acronym").empty)
    checks.check("overlap groups are evaluated", not result.subgroup_metrics("overlapping").empty)
    checks.equal("missing tokenizer stays explicitly unavailable", bool(result.events["tokenization_available"].any()), False)
    checks.equal("missing confidence stays nullable", int(result.events["pred_confidence"].notna().sum()), 0)


def verify_annotation_audit(checks: Checks) -> None:
    bad = frame([
        ["d", "D", -1, 2, "ab"], ["d", "D", 0, 0, ""],
        ["d", "D", 0, 2, "zz"], ["d", "X", 0, 2, "ab"], ["d", "X", 0, 2, "ab"],
    ])
    audit = audit_annotations(bad, {"d": "ab\r\ncd"})
    issues = {item for row in audit["annotation_issues"] for item in row}
    checks.check("negative offsets are audited", "NEGATIVE_OFFSET" in issues)
    checks.check("zero lengths are audited", "ZERO_OR_NEGATIVE_LENGTH" in issues)
    checks.check("surface mismatches are audited", "SURFACE_MISMATCH" in issues)
    checks.check("duplicates are audited", "DUPLICATE_ANNOTATION" in issues)
    checks.check("conflicting labels are audited", "CONFLICTING_LABELS" in issues)


def main() -> int:
    checks = Checks("deep analysis")
    verify_exposure(checks)
    verify_annotation_audit(checks)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        verify_diagnostics(checks, root)
        verify_subgroups(checks, root / "groups")
    return checks.report()


if __name__ == "__main__":
    run(main)
