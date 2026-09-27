# NER analysis: researcher and developer guide

`lab.ner.analysis` turns NER predictions, evaluation gold spans, and training
gold spans into a persistent diagnostic artifact and, optionally, a scientific
report. It is intended for error analysis and generalization studies after
inference has already been run.

> **Current command-line status:** the local implementation has **no Typer CLI
> and no analysis command tree**. There is no `lab.ner.analysis.cli`, no
> `lab.ner.analysis.__main__`, and no analysis task registered with the
> project-wide `lab` executable. Therefore commands such as
> `python -m lab.ner.analysis --help`, `lab analysis`, and
> `lab run` with an analysis task are not supported. This guide deliberately
> provides no invented CLI syntax. All runnable workflows below use the actual
> Python API.

The project-wide executable is an `argparse` application implemented in
`src/lab/cli.py`; it supports `lab run <config.yaml>` and `lab tasks`. Its NER
task registry (`src/lab/ner/tasks.py`) contains training, hyperparameter search,
and prediction only. It does not expose this analysis package.

## Contents

1. [Overview and architecture](#overview-and-architecture)
2. [Installation and input contracts](#installation-and-input-contracts)
3. [Quick start](#quick-start)
4. [Python API reference](#python-api-reference)
5. [Configuration](#configuration)
6. [Official evaluation and diagnostic association](#official-evaluation-and-diagnostic-association)
7. [Training exposure, RapidFuzz, and zero-shot definitions](#training-exposure-rapidfuzz-and-zero-shot-definitions)
8. [`evaluation.parquet`](#evaluationparquet)
9. [Parquet data dictionary](#parquet-data-dictionary)
10. [Inspecting and querying an artifact](#inspecting-and-querying-an-artifact)
11. [Subgroup metrics and aggregate metadata](#subgroup-metrics-and-aggregate-metadata)
12. [Oracle analysis](#oracle-analysis)
13. [Document-level bootstrap](#document-level-bootstrap)
14. [Publication report and figures](#publication-report-and-figures)
15. [Interpretation guide](#interpretation-guide)
16. [Troubleshooting](#troubleshooting)
17. [Performance and reproducibility](#performance-and-reproducibility)
18. [Developer extension guide](#developer-extension-guide)

## Overview and architecture

The analysis layer adds diagnostics to the existing evaluation; it does not
replace evaluation and it never reruns inference.

```text
prediction span table
evaluation-gold span table
training-gold span table
        │
        ├─ span validation and normalization of dtypes
        │      src/lab/ner/analysis/evaluation.py
        │
        ├─ existing span_metrics evaluator
        │      src/lab/ner/evaluation/scoring.py
        │      STRICT + existing character-level metrics
        │
        ├─ canonical STRICT TP/FP/FN events
        │      src/lab/ner/analysis/evaluation.py
        │
        ├─ diagnostic FP/FN association and error taxonomy
        │      src/lab/ner/analysis/diagnostics.py
        │
        ├─ annotation/entity/tokenization features
        │      src/lab/ner/analysis/features.py
        │
        ├─ training exposure and nearest training mentions
        │      src/lab/ner/analysis/exposure.py
        │
        ├─ complete subgroup evaluation with span_metrics
        │      src/lab/ner/analysis/subgroups.py
        │
        └─ evaluation.parquet + aggregate metadata
               src/lab/ner/analysis/parquet.py
                       │
                       ├─ STRICT oracle upper bounds
                       │      oracles.py
                       ├─ document bootstrap intervals
                       │      statistics.py
                       └─ Parquet tables + scientific SVG report
                              publication.py / plotting.py
```

The public orchestration function is `analyze_evaluation` in `api.py`. It calls
the same local `span_metrics` implementation used elsewhere in the project.
There is no second STRICT scorer and no analysis-specific partial F1.

## Installation and input contracts

Install the local package according to the repository's normal installation
instructions. `rapidfuzz>=3.0` is a declared base dependency. Pandas and
PyArrow are also base dependencies.

### Span tables

The three required inputs are:

1. **predictions** — model-predicted spans;
2. **evaluation gold** — the spans against which predictions are evaluated;
3. **training gold** — training annotations used for exposure and similarity.

Each may be a pandas `DataFrame`, a `.parquet` path, or a tab-separated file
path. Non-Parquet paths are always read with `pandas.read_csv(..., sep="\t")`.
They are span tables, not canonical corpus Parquets containing
`entities_json`.

Required columns:

| Column | Required for | Type accepted | Meaning |
|---|---|---|---|
| `filename` | all inputs | string-like | Document identifier. |
| `label` | all inputs | string-like | Entity label. |
| `start_span` | all inputs | integer-like | Inclusive, zero-based start offset. |
| `end_span` | all inputs | integer-like | Exclusive end offset; spans follow `text[start:end]`. |
| `text` | all inputs | string-like | Stored entity surface. |
| `score` | predictions only, optional | numeric in `[0, 1]` | Prediction score used for confidence diagnostics and threshold sweeps. It is not called a calibrated probability. |

The current canonical span model is single-label. The Parquet schema includes
list-valued `gold_labels` and `pred_labels` for extensibility, but the current
evaluator and confusion analysis use the scalar `label` value.

### Optional document text

`evaluation_documents` and `training_documents` may be:

* a `{doc_id: text}` mapping;
* a DataFrame with `doc_id` and `text` columns; or
* a Parquet path containing those two columns.

Document text enables surface/offset integrity checks. Its absence does not
stop analysis; the audit records `DOCUMENT_UNAVAILABLE`. A tokenizer may be
passed as a Python object accepting `tokenizer(text,
add_special_tokens=False)` and returning an `input_ids` sequence. There is no
CLI tokenizer loader.

## Quick start

This is the shortest complete executable workflow:

```python
from lab.ner.analysis import analyze_evaluation, create_publication_report

analysis = analyze_evaluation(
    predictions="runs/model-1/predictions.tsv",
    evaluation_gold="corpora/eval/gold.tsv",
    training_gold="corpora/train/gold.tsv",
    output_dir="analysis_output",
    run_id="model-1-eval",
)

report = create_publication_report(analysis.path)

print(analysis.path)                         # analysis_output/evaluation.parquet
print(analysis.metrics["span_strict_f1"])
print(report.tables["oracle_error_budget"])
print(report.figures["generalization_strict_f1"])
```

This performs the following actual steps:

1. loads and validates all three span tables;
2. calls the existing evaluator for overall STRICT and character metrics;
3. creates canonical TP/FP/FN contribution rows;
4. associates unmatched FP/FN rows diagnostically without changing evaluation;
5. computes exposure, similarity, zero-shot, structural, and entity features;
6. calls the existing evaluator on supported subgroup gold and prediction sets;
7. writes `analysis_output/evaluation.parquet` with ZSTD compression;
8. reloads that canonical artifact to compute oracle and bootstrap tables; and
9. writes vector SVG figures, captions, findings, and a report manifest.

## Command-line interface status

### Actual command hierarchy

```text
Analysis Typer hierarchy: none
Analysis callbacks:       none
Analysis aliases:         none
Analysis environment vars:none
Analysis CLI config files:none
```

The following invocations do **not** exist and must not be used:

```text
lab analysis ...
lab run <analysis-config.yaml>
python -m lab.ner.analysis ...
```

There are consequently no CLI options such as `--fuzzy-threshold` or
`--bootstrap-samples`. The equivalent real controls are Python dataclass
fields, shown below. Defaults come from the dataclasses; there is no precedence
among CLI flags, environment variables, and configuration files because none
of those mechanisms exists for analysis.

## Python API reference

All names below are exported from `lab.ner.analysis`.

### `analyze_evaluation(...) -> AnalysisResult`

```python
analyze_evaluation(
    predictions,
    evaluation_gold,
    training_gold,
    output_dir,
    *,
    run_id,
    tags=None,
    min_overlap_percentage=40.0,
    config=None,
    evaluation_documents=None,
    training_documents=None,
    tokenizer=None,
)
```

| Parameter | Required | Default | Behavior |
|---|---:|---:|---|
| `predictions` | yes | — | Prediction DataFrame or span-table path. Optional `score` is retained as `pred_confidence`. |
| `evaluation_gold` | yes | — | Evaluation-gold DataFrame or span-table path. |
| `training_gold` | yes | — | Training-gold DataFrame or span-table path. Empty training is accepted and makes all mentions lexically unseen. |
| `output_dir` | yes | — | Created if necessary; receives `evaluation.parquet`. |
| `run_id` | yes, keyword-only | — | Non-empty run identifier stored on every event and in metadata. |
| `tags` | no | union of gold/predicted labels | Labels passed to the existing evaluator. Supplying a restricted list changes what that evaluator includes. |
| `min_overlap_percentage` | no | `40.0` | Passed to the existing evaluator and its STRICT-index extraction. It does not create a partial analysis metric. |
| `config` | no | `AnalysisConfig()` | Exposure, bins, confidence, acronym, and intersection settings. |
| `evaluation_documents` | no | `None` | Optional source text for gold/prediction annotation audits and document length. |
| `training_documents` | no | `None` | Optional source text for aggregate training annotation audits. |
| `tokenizer` | no | `None` | Optional tokenizer object for subtoken features. Missing tokenization remains explicit and nullable. |

The return object exposes `path`, `events`, `metrics`, and `metadata`.

### `load_evaluation(path) -> AnalysisResult`

Loads an existing analysis Parquet without inference. The file must contain the
`lab.ner.analysis` schema metadata and a supported schema version (`1.0.0` or
the current `2.0.0`).

```python
from lab.ner.analysis import load_evaluation

analysis = load_evaluation("analysis_output/evaluation.parquet")
print(analysis.strict_counts)
print(analysis.summary())
```

### `AnalysisResult` query methods

| Method | Returned rows/data |
|---|---|
| `zero_shot()` | Gold-bearing rows where configured `zero_shot_default` is true. |
| `zero_shot_correct()` | Correct gold-bearing default-zero-shot rows. |
| `zero_shot_errors()` | Non-correct gold-bearing default-zero-shot rows. |
| `boundary_errors()` | One gold-side row per `BOUNDARY_ERROR` or `BOUNDARY_AND_LABEL_ERROR` relationship. |
| `label_errors()` | One gold-side row per `LABEL_ERROR` or `BOUNDARY_AND_LABEL_ERROR` relationship. |
| `by_label(label)` | Rows whose gold or prediction label matches. |
| `by_similarity(bin_name)` | Gold-bearing rows in one configured similarity bin. |
| `by_training_frequency(bin_name)` | Gold-bearing rows in one frequency bin. |
| `overlapping_entities()` | Gold-bearing rows overlapping another gold entity. |
| `acronyms()` | Gold-bearing acronym-like rows. |
| `high_confidence_errors()` | Error contributions whose prediction score meets the configured high threshold. |
| `subgroup_metrics(family=None)` | Aggregate subgroup table, optionally restricted by family. |
| `summary()` | Factual aggregate summary stored in metadata. |
| `oracle_error_budget(minimum_confusion_support=1)` | STRICT diagnostic upper-bound DataFrame. |

### Publication/statistics functions

The public functions `oracle_error_budget`, `bootstrap_intervals`,
`paired_seen_zero_shot_difference`, and `create_publication_report` consume an
existing `AnalysisResult` or its `evaluation.parquet`; none reruns inference.

```python
from lab.ner.analysis import (
    BootstrapConfig,
    PublicationConfig,
    bootstrap_intervals,
    create_publication_report,
    oracle_error_budget,
    paired_seen_zero_shot_difference,
)

oracles = oracle_error_budget(analysis.events, minimum_confusion_support=2)
intervals = bootstrap_intervals(
    analysis.events,
    BootstrapConfig(samples=2000, confidence_level=0.95, seed=17),
)
paired = paired_seen_zero_shot_difference(
    analysis.events,
    BootstrapConfig(samples=2000, seed=17),
)
report = create_publication_report(
    analysis.path,
    config=PublicationConfig(
        bootstrap=BootstrapConfig(samples=2000, seed=17),
        minimum_label_support=5,
        minimum_confusion_support=2,
        boundary_display_quantile=0.99,
    ),
)
```

## Configuration

Configuration is Python-only. Complete constructors and defaults follow.

### `NormalizationConfig`

```python
NormalizationConfig(
    unicode_form="NFKC",
    strip=True,
    casefold=True,
    collapse_whitespace=True,
    normalize_punctuation=False,
    remove_diacritics=False,
)
```

The first four transformations are defaults. Punctuation replacement and
diacritic removal are potentially destructive and therefore opt-in.

### `AnalysisConfig`

```python
from lab.ner.analysis import AnalysisConfig, NormalizationConfig

config = AnalysisConfig(
    normalization=NormalizationConfig(),
    fuzzy_threshold=0.80,
    fuzzy_scorer="WRatio",
    zero_shot_definition="fuzzy_unseen",
    similarity_bins=(
        ("EXACT", 1.00, 1.00),
        ("0.95-<1.00", 0.95, 1.00),
        ("0.90-<0.95", 0.90, 0.95),
        ("0.85-<0.90", 0.85, 0.90),
        ("0.80-<0.85", 0.80, 0.85),
        ("<0.80", 0.00, 0.80),
    ),
    frequency_bins=(
        ("0", 0, 0),
        ("1", 1, 1),
        ("2-5", 2, 5),
        ("6-10", 6, 10),
        ("11-20", 11, 20),
        (">20", 21, None),
    ),
    confidence_thresholds=(),
    high_confidence_threshold=0.90,
    low_confidence_threshold=0.50,
    acronym_max_length=12,
    acronym_min_uppercase_ratio=0.60,
    minimum_intersection_support=2,
)
```

Accepted fuzzy scorers are exactly `ratio`, `WRatio`, `partial_ratio`,
`token_sort_ratio`, and `token_set_ratio`. `fuzzy_threshold` must be within
`0.0–1.0`. Accepted `zero_shot_definition` values are `exact_unseen`,
`normalized_unseen`, and `fuzzy_unseen`.

Set confidence thresholds explicitly to obtain a non-empty threshold sweep:

```python
config = AnalysisConfig(confidence_thresholds=(0.50, 0.70, 0.80, 0.90))
```

### `BootstrapConfig` and `PublicationConfig`

```python
BootstrapConfig(
    enabled=True,
    samples=2000,
    confidence_level=0.95,
    seed=20250927,
    minimum_documents=2,
)

PublicationConfig(
    bootstrap=BootstrapConfig(),
    minimum_label_support=1,
    minimum_confusion_support=1,
    boundary_display_quantile=0.99,
)
```

To disable bootstrap work while still producing tables and figures:

```python
report = create_publication_report(
    analysis.path,
    config=PublicationConfig(bootstrap=BootstrapConfig(enabled=False)),
)
```

## Official evaluation and diagnostic association

### Existing evaluator reuse

Overall evaluation and every supported full subgroup call
`lab.ner.evaluation.span_metrics`. The evaluator returns the repository's
official `span_strict_*` metrics and `char_*` character-coverage metrics. The
publication code calls the latter “character” metrics; filenames retain
`chrf` where established by the reporting interface. They are not a new scorer.

Subgroups independently select gold and prediction populations before calling
the evaluator. Implemented families are exact, normalized, and fuzzy exposure;
configured zero-shot status; similarity and frequency bins; labels;
seen/zero-shot within labels; word length; acronym status; overlap/nesting;
optional tokenization groups; and configured confidence thresholds. This is
methodologically different from filtering a table of already-correct matches,
because both gold and prediction denominators are evaluated.

### STRICT versus diagnostic association

Suppose gold is `myocardial infarction` and prediction is
`acute myocardial infarction`. Official STRICT evaluation still contributes
one FP and one FN. Diagnostic association may link those two contribution rows
and classify them as `BOUNDARY_ERROR`; it never converts them into evaluation
credit. There is no partial precision, partial recall, partial F1, or fractional
TP in this package.

Unmatched spans are associated per document by deterministic maximum-weight
bipartite assignment. Positive candidate weights use exact boundary equality,
intersection/IoU, label agreement, containment, start/end distance, and
confidence. Non-overlapping, non-equal spans are not paired.

### Error taxonomy

`error_primary` is mutually exclusive per event:

| Value | Definition |
|---|---|
| `CORRECT` | Official STRICT TP containing both gold and prediction. |
| `MISSED` | Unpaired official FN. |
| `SPURIOUS` | Unpaired official FP. |
| `LABEL_ERROR` | Diagnostically paired FP/FN with equal boundaries and different labels. |
| `BOUNDARY_ERROR` | Paired FP/FN with matching label and unequal boundaries. |
| `BOUNDARY_AND_LABEL_ERROR` | Paired FP/FN with both boundary and label disagreement. |

A paired error exists as two canonical rows because official accounting must
retain its FP and FN. `error_pair_id` joins the rows; `error_pair_role` is
`GOLD` or `PRED`. Query helpers return the gold-side row to avoid double-counting.

Independent booleans are not mutually exclusive:

* `error_fragmentation`: one unmatched gold overlaps at least two unmatched predictions;
* `error_merging`: one unmatched prediction overlaps at least two unmatched gold spans;
* `error_duplicate_prediction`: the prediction table repeats the same document, label, and boundaries;
* `error_nested_entity`: the gold span contains or is contained by another gold span;
* `error_overlapping_entity`: the gold span intersects another gold span.

### Boundary fields

For associated spans:

```text
span_start_delta  = pred_start - gold_start
span_end_delta    = pred_end - gold_end
span_length_delta = predicted length - gold length
```

Thus `span_start_delta = -1` means the prediction starts one character before
gold; `span_end_delta = 2` means it ends two characters after gold. Absolute
versions remove direction. `span_boundary_error` is
`LEFT_BOUNDARY_ERROR`, `RIGHT_BOUNDARY_ERROR`, or `BOTH_BOUNDARIES_ERROR`.
`span_relation` is `EXACT_BOUNDARY`, `PRED_CONTAINS_GOLD`,
`GOLD_CONTAINS_PRED`, `LEFT_OVERLAP`, or `RIGHT_OVERLAP`.

## Training exposure, RapidFuzz, and zero-shot definitions

### Exposure precedence

Each mention receives the first applicable `exposure_class`:

1. `EXACT_SEEN_SAME_LABEL`
2. `EXACT_SEEN_OTHER_LABEL`
3. `NORMALIZED_SEEN_SAME_LABEL`
4. `NORMALIZED_SEEN_OTHER_LABEL`
5. `FUZZY_SEEN_SAME_LABEL`
6. `FUZZY_SEEN_OTHER_LABEL`
7. `UNSEEN`

Exact frequency counts all training rows with the raw surface. Normalized
frequency counts all rows with the normalized surface. Label-list columns
preserve every observed training label; `train_exact_label_ambiguous` is true
when an exact surface has multiple labels.

### RapidFuzz behavior

The default scorer is `WRatio`. Training `(normalized surface, label)` pairs
are deduplicated, entries are stably sorted, and repeated evaluation lookup
requests are cached. The implementation retains only nearest all-label,
same-label, and other-label neighbors; it never materializes a dense
evaluation-by-training matrix.

Similarity is persisted on `0.0–1.0`. The public default threshold `0.80`
means 80% under the selected scorer and is converted internally to `80.0` for
comparison with RapidFuzz's scale. Ties are deterministic by original text,
label, and normalized text.

RapidFuzz is declared as a required dependency. If it is nevertheless absent
from a deliberately minimal environment, `_resolve_scorer` uses a deterministic
`difflib.SequenceMatcher` compatibility backend and records
`similarity_backend="difflib-compatibility"` in metadata. This fallback does
not implement the five distinct RapidFuzz scorer semantics; install the declared
dependency for research runs that specify a RapidFuzz scorer.

### Separate zero-shot definitions

| Column | True when |
|---|---|
| `zero_shot_exact` | Raw evaluation surface never occurs in training. |
| `zero_shot_normalized` | Normalized surface never occurs in training. |
| `zero_shot_fuzzy` | Nearest normalized training similarity is below `fuzzy_threshold`. Equality with `0.80` is fuzzy-seen at the default threshold. |
| `zero_shot_default` | The definition selected by `zero_shot_definition`; default is fuzzy unseen. |

These are lexical definitions. They do not imply an unseen semantic concept,
entity type, or context.

## `evaluation.parquet`

This is the canonical analytical artifact. Each row contributes exactly one
official STRICT TP, FP, or FN. A TP preserves both sides; an FP has a prediction
side; an FN has a gold side. Diagnostic pairs remain two rows linked by
`error_pair_id`.

The current schema version is `2.0.0`; the loader also accepts `1.0.0`.
Writing uses ZSTD, dictionary encoding, statistics, and an atomic temporary-file
replacement. Full document text and training tables are not duplicated.

Namespaced Arrow metadata under `lab.ner.analysis` stores the schema/run ID,
official metric dictionary, exact analysis configuration, similarity backend,
training provenance SHA-256 and annotation summary, and compact aggregates for
subgroups, confusion, documents, intersections, concentrations, labels,
dataset shift, and the factual summary.

Official STRICT counts reconstruct as:

```python
tp = int(df["strict_is_tp"].sum())
fp = int(df["strict_is_fp"].sum())
fn = int(df["strict_is_fn"].sum())
```

Character metrics are aggregate evaluator results in metadata; fake event-level
character contributions are intentionally not stored.

## Parquet data dictionary

Arrow `float` below is float32. Dictionary columns use int16 indices and string
values. All columns marked nullable may be null when their side, diagnostic
pair, tokenizer, source document, training neighbor, or confidence is absent.

### Identity, gold, prediction, and official STRICT

| Column | Arrow type | Nullable | Description |
|---|---|---:|---|
| `run_id` | dictionary/string | no | User-supplied run ID. |
| `event_id` | uint64 | no | Stable row ID within the artifact. |
| `document_id` | dictionary/string | no | Document identifier. |
| `gold_id` | uint64 | yes | Input gold-row index. |
| `gold_exists` | bool | no | Whether this official contribution has a gold side. |
| `gold_start`, `gold_end` | int64 | yes | Half-open gold offsets. |
| `gold_text` | string | yes | Gold surface. |
| `gold_label` | dictionary/string | yes | Scalar gold label. |
| `gold_labels` | list<string> | yes | Current single-label value represented as a list. |
| `pred_id` | uint64 | yes | Input prediction-row index. |
| `pred_exists` | bool | no | Whether this contribution has a prediction side. |
| `pred_start`, `pred_end` | int64 | yes | Half-open prediction offsets. |
| `pred_text` | string | yes | Predicted surface. |
| `pred_label` | dictionary/string | yes | Scalar predicted label. |
| `pred_labels` | list<string> | yes | Current single-label value represented as a list. |
| `pred_confidence` | float32 | yes | Optional input `score`; no calibration claim. |
| `strict_outcome` | dictionary/string | no | `TP`, `FP`, or `FN`. |
| `strict_is_tp`, `strict_is_fp`, `strict_is_fn` | bool | no | One-hot official accounting flags. |
| `strict_correct` | bool | no | True exactly for official STRICT TP rows. |

### Diagnostic errors and boundary association

| Column | Arrow type | Nullable | Description |
|---|---|---:|---|
| `error_pair_id` | uint64 | yes | Identifier shared by associated FP/FN rows. |
| `error_type` | dictionary/string | yes | Compatibility alias of `error_primary`. |
| `error_primary` | dictionary/string | yes | Primary taxonomy described above. |
| `error_pair_role` | dictionary/string | yes | `GOLD` or `PRED` within a diagnostic pair. |
| `diagnostic_gold_id`, `diagnostic_pred_id` | int64 | yes | Input indices of associated entities. |
| `error_label_agreement` | bool | yes | Whether associated labels agree. |
| `error_fragmentation`, `error_merging` | bool | yes | Independent structural relationship flags. |
| `error_duplicate_prediction` | bool | yes | Repeated prediction identity. |
| `error_nested_entity`, `error_overlapping_entity` | bool | yes | Gold structural flags mirrored into error features. |
| `span_intersection` | int64 | yes | Character intersection length. |
| `span_iou` | float32 | yes | Span intersection over union. |
| `span_start_delta`, `span_end_delta` | int64 | yes | Signed prediction-minus-gold boundary offsets. |
| `span_length_delta` | int64 | yes | Signed length difference. |
| `span_abs_start_delta`, `span_abs_end_delta` | int64 | yes | Absolute boundary errors. |
| `span_boundary_error` | dictionary/string | yes | Left, right, or both boundary category. |
| `span_relation` | dictionary/string | yes | Exact, containment, or overlap direction. |
| `span_one_character_offset` | bool | yes | At least one absolute delta equals one. |
| `span_leading_whitespace`, `span_trailing_whitespace` | bool | yes | Predicted surface whitespace flags. |
| `span_punctuation_included`, `span_punctuation_excluded` | bool | yes | Simple punctuation-only surface difference flags. |

### Gold overlap and annotation integrity

| Column | Arrow type | Nullable | Description |
|---|---|---:|---|
| `gold_overlaps_other_gold` | bool | yes | Gold intersects another gold span. |
| `gold_nested` | bool | yes | Gold contains or is contained by another gold span. |
| `gold_contains_other_gold` | bool | yes | Gold contains another gold span. |
| `gold_contained_by_other_gold` | bool | yes | Another gold span contains this one. |
| `gold_overlap_depth` | int64 | yes | Number of overlapping gold spans including self. |
| `annotation_gold_valid`, `annotation_pred_valid` | bool | yes | Integrity status independent of model-error taxonomy. |
| `annotation_gold_issues`, `annotation_pred_issues` | list<string> | yes | Issues such as unavailable document, range/surface/whitespace/newline/Unicode problems, duplicates, or conflicting labels. |

### Training exposure, neighbors, and zero-shot

| Column | Arrow type | Nullable | Description |
|---|---|---:|---|
| `exposure_class` | dictionary/string | yes | Seven-level exposure class. |
| `train_exact_seen` | bool | yes | Raw surface occurs in training. |
| `train_exact_seen_same_label`, `train_exact_seen_other_label` | bool | yes | Exact label relationships. |
| `train_exact_frequency` | int64 | yes | Raw training surface count. |
| `train_exact_labels` | list<string> | yes | Sorted training labels for raw surface. |
| `train_exact_label_ambiguous` | bool | yes | Exact surface has multiple labels. |
| `train_normalized_text` | string | yes | Configured normalized surface. |
| `train_normalized_seen` | bool | yes | Normalized surface occurs in training. |
| `train_normalized_seen_same_label`, `train_normalized_seen_other_label` | bool | yes | Normalized label relationships. |
| `train_normalized_frequency` | int64 | yes | Normalized training surface count. |
| `train_normalized_labels` | list<string> | yes | Sorted labels for normalized surface. |
| `train_nearest_text`, `train_nearest_label` | string/dictionary | yes | Deterministic overall nearest training mention. |
| `train_nearest_similarity` | float32 | yes | Overall nearest similarity on `0–1`. |
| `train_nearest_same_label` | bool | yes | Whether overall nearest label equals evaluation label. |
| `train_nearest_same_label_text` | string | yes | Nearest same-label training text. |
| `train_nearest_same_label_similarity` | float32 | yes | Same-label similarity on `0–1`. |
| `train_nearest_other_label_text`, `train_nearest_other_label` | string/dictionary | yes | Nearest other-label training mention. |
| `train_nearest_other_label_similarity` | float32 | yes | Other-label similarity on `0–1`. |
| `train_exact_seen_with_pred_label` | bool | yes | For associated errors, exact training labels include predicted label. |
| `train_normalized_seen_with_pred_label` | bool | yes | Normalized training labels include predicted label. |
| `train_nearest_has_pred_label` | bool | yes | Overall nearest training label equals associated prediction label. |
| `similarity_bin` | dictionary/string | yes | Configured nearest-similarity stratum. |
| `train_frequency_bin` | dictionary/string | yes | Configured exact-frequency group. |
| `zero_shot_exact`, `zero_shot_normalized`, `zero_shot_fuzzy` | bool | yes | Independent lexical zero-shot flags. |
| `zero_shot_default` | bool | yes | Selected zero-shot definition. |
| `zero_shot_outcome` | dictionary/string | yes | `ZERO_SHOT_` plus gold-side primary outcome. |

### Entity, tokenization, and confidence features

| Column | Arrow type | Nullable | Description |
|---|---|---:|---|
| `entity_character_length` | int64 | yes | Surface character length. |
| `entity_whitespace_token_count` | int64 | yes | `text.split()` count. |
| `entity_word_length_bin` | dictionary/string | yes | `1`, `2`, `3`, `4`, or `5+` (empty text can produce `0`). |
| `entity_contains_digit`, `entity_contains_decimal`, `entity_contains_percentage` | bool | yes | Numeric orthographic flags. |
| `entity_contains_hyphen`, `entity_contains_slash`, `entity_contains_parentheses` | bool | yes | Delimiter flags. |
| `entity_contains_punctuation`, `entity_contains_greek`, `entity_contains_special_symbol` | bool | yes | Additional orthographic flags. |
| `entity_casing` | dictionary/string | yes | `UPPERCASE`, `LOWERCASE`, `TITLECASE`, `MIXEDCASE`, or `UNCASED`. |
| `entity_acronym_like` | bool | yes | Configurable length/uppercase-ratio heuristic. |
| `tokenization_available` | bool | yes | Whether a tokenizer was supplied. |
| `tokenization_subtoken_count` | int64 | yes | Model-token count without special tokens. |
| `tokenization_subtoken_to_word_ratio` | float32 | yes | Subtokens divided by at least one whitespace token. |
| `tokenization_fragmentation_category` | dictionary/string | yes | `1`, `2`, `3-4`, `5-8`, or `9+`. |
| `confidence_high_error` | bool | yes | Non-TP with score at least configured high threshold. |
| `confidence_low_correct` | bool | yes | TP with score at most configured low threshold. |

## Inspecting and querying an artifact

```python
import pandas as pd

df = pd.read_parquet("analysis_output/evaluation.parquet")

# Gold-side fuzzy-zero-shot misses (one row per missed gold entity).
zero_shot_misses = df[
    df["gold_exists"]
    & df["zero_shot_fuzzy"].fillna(False)
    & (df["error_primary"] == "MISSED")
]

# One row per diagnostic boundary relationship, avoiding paired-row duplication.
boundary_errors = df[
    (df["error_pair_role"] == "GOLD")
    & df["error_primary"].isin(["BOUNDARY_ERROR", "BOUNDARY_AND_LABEL_ERROR"])
]

# All contributions involving a DISEASE label.
disease = df[(df["gold_label"] == "DISEASE") | (df["pred_label"] == "DISEASE")]

# True, unpaired, high-score spurious predictions.
high_score_spurious = df[
    (df["error_primary"] == "SPURIOUS")
    & df["pred_confidence"].ge(0.90)
]
```

Prefer `load_evaluation` when metadata or query helpers are needed; plain
`pandas.read_parquet` reads event columns but does not decode the namespaced
JSON metadata into an `AnalysisResult`.

To inspect metadata directly:

```python
import json
import pyarrow.parquet as pq

schema = pq.read_schema("analysis_output/evaluation.parquet")
metadata = json.loads(schema.metadata[b"lab.ner.analysis"].decode("utf-8"))
print(metadata["schema_version"])
print(metadata["analysis_config"])
print(metadata["official_metrics"])
```

## Subgroup metrics and aggregate metadata

`analysis.subgroup_metrics()` returns columns including `family`, `value`,
optional `label`, gold/predicted support, STRICT TP/FP/FN/P/R/F1, and character
TP/FP/FN/P/R/F1. Families currently emitted are:

* `exact_exposure`, `normalized_exposure`, `fuzzy_exposure`;
* `zero_shot_default`;
* `similarity`, `training_frequency`;
* `label`, `label_zero_shot`;
* `word_length`, `acronym`, `overlapping`, `nested`, `tokenization`; and
* `confidence_threshold` when thresholds were configured and scores exist.

Metadata also contains exact-boundary label confusion, `[MISSED]`/`[SPURIOUS]`
flows, document summaries, supported error intersections, error concentration,
per-label analysis, descriptive training/evaluation shift, and a factual
summary. These are descriptive diagnostics, not additional official metrics.

## Oracle analysis

An oracle is a **counterfactual diagnostic upper bound**, not an expected model
improvement. Baseline counts always come from canonical official contributions.

Implemented scenarios:

| Scenario | Counterfactual operation |
|---|---|
| Boundary correction | Pure `BOUNDARY_ERROR` FP/FN pair becomes one TP. |
| Label correction | Exact-boundary `LABEL_ERROR` pair becomes one TP. |
| Boundary + label correction | Corresponding combined-error pair becomes one TP. |
| All missed entities | Every unpaired `MISSED` FN becomes a TP. |
| Exact-/normalized-/fuzzy-seen miss recovery | Selected unpaired seen FN becomes a TP. |
| Fuzzy-zero-shot miss recovery | Selected unpaired fuzzy-unseen FN becomes a TP. |
| Spurious-prediction removal | Every true unpaired `SPURIOUS` FP is removed. |
| Per-label | Paired errors and unpaired misses/spurious contributions associated with one label are corrected. |
| Confusion pair | Supported exact-boundary `gold → predicted` label pairs are corrected. |
| Combined scenarios | Explicit unions for boundary, classification, seen misses, generalization misses, or all diagnostic errors. |

Rows report baseline and oracle counts/P/R/F1, absolute deltas, relative error
reduction, affected events/entities, support, exposure mix, main labels, and
mean available confidence.

Oracle gains are not additive. F1 is nonlinear; scenarios can overlap; and one
diagnostic relation shares an FP and FN. Therefore
`ΔF1(boundary) + ΔF1(label) + ΔF1(zero-shot)` is not the total recoverable F1.
Use an explicitly computed combined row. No character-level oracle is invented,
because strict correction does not uniquely specify counterfactual character
coverage.

## Document-level bootstrap

Bootstrap uses documents rather than entities as the resampling unit, retaining
within-document dependence. The implementation first calls the existing
evaluator for each complete document/stratum, then sums its additive official
counts according to each resample and derives metrics with the existing count
helper.

Targets include overall STRICT and character precision/recall/F1, configured
zero-shot groups, similarity bins, labels, training-frequency groups,
overlap/non-overlap, and acronym/non-acronym. The paired comparison reports
seen-minus-zero-shot differences from the same sampled documents.

Defaults are 2,000 samples, 95% intervals, seed `20250927`, and at least two
documents. A smaller dataset receives `insufficient_documents` with null bounds
instead of misleading uncertainty. No automatic multiple hypothesis-testing
suite is performed.

## Publication report and figures

`create_publication_report(evaluation_parquet, output_dir=None, config=None)`
loads the canonical artifact. With `output_dir=None`, outputs are placed beside
`evaluation.parquet`.

```text
analysis_output/
├── evaluation.parquet
├── figures/
│   └── *.svg
├── tables/
│   ├── overall_metrics.parquet
│   ├── metrics_by_exposure.parquet
│   ├── metrics_by_similarity.parquet
│   ├── metrics_by_label.parquet
│   ├── major_label_confusions.parquet
│   ├── oracle_error_budget.parquet
│   ├── bootstrap_intervals.parquet
│   └── paired_comparisons.parquet
├── captions.md
├── findings.md
└── report_manifest.json
```

### Figure reference

| Filename | Scientific content and axes |
|---|---|
| `generalization_strict_recall.svg` | Similarity bin (x) versus STRICT recall (y), with support and bootstrap CI when available. |
| `generalization_strict_f1.svg` | Similarity bin versus STRICT F1, support, and available CI. |
| `generalization_chrf_f1.svg` | Similarity bin versus existing character F1. |
| `rapidfuzz_vs_strict_f1.svg` | Alternate explicitly named similarity-versus-STRICT-F1 view with configured threshold marker. |
| `seen_vs_zeroshot_by_label.svg` | Grouped fuzzy-seen/zero-shot STRICT F1 by label. |
| `seen_vs_zeroshot_by_label_precision.svg` | Grouped STRICT precision by label. |
| `seen_vs_zeroshot_by_label_recall.svg` | Grouped STRICT recall by label. |
| `seen_vs_zeroshot_by_label_chrf_f1.svg` | Grouped existing character F1 by label. |
| `error_landscape_counts.svg` | Exposure class × diagnostic outcome raw-count heatmap. |
| `error_landscape_normalized.svg` | Same heatmap row-normalized. |
| `oracle_error_budget.svg` | Single-family STRICT absolute ΔF1 upper bounds. Bars are not additive. |
| `label_confusion_counts.svg` | Exact-boundary wrong-label count matrix. |
| `label_confusion_normalized.svg` | Same confusion matrix normalized within gold-label row. |
| `training_frequency_vs_strict_f1.svg` | Exact training-frequency bin versus STRICT F1 and support. |
| `boundary_start_delta.svg` | Histogram of `pred_start - gold_start`. |
| `boundary_end_delta.svg` | Histogram of `pred_end - gold_end`. |
| `boundary_absolute_delta.svg` | Combined absolute start/end deviations. Display clipping uses the configured quantile and is annotated without altering data. |
| `boundary_direction_by_label.svg` | Gold label × boundary-category count heatmap. |
| `entity_length_vs_performance.svg` | Word-length bin versus STRICT F1. |
| `tokenization_vs_performance.svg` | Fragmentation category versus STRICT F1; can show no eligible data without a tokenizer. |
| `confidence_by_error_type.svg` | Deterministic prediction-score strip distribution by outcome; it is not a calibration plot. |
| `confidence_threshold_sweep.svg` | Configured threshold versus STRICT F1; empty when no thresholds/scores exist. |
| `confidence_coverage_vs_threshold.svg` | Threshold versus retained-prediction coverage. |
| `error_intersections.svg` | Supported feature/error intersections above configured minimum support. |
| `error_pareto.svg` | Rank versus cumulative proportion of errors concentrated by surface. |
| `dataset_shift_labels.svg` | Descriptive training/evaluation label proportions. |
| `dataset_shift_characteristics.svg` | Descriptive acronym, digit, and punctuation prevalence. |

### Plot style and SVG semantics

`plotting.py` centralizes `PlotTheme`, `OKABE_ITO`, and `SIZE_PRESETS`.
Figures use a white background, Okabe–Ito colors, no default grids, no top or
right Cartesian spines, stable ordering, editable `<text>`, and legends above
the data. Presets are `single-column` (3.5 × 2.7 in), `1.5-column` (5.2 × 3.4
in), `double-column` (7.2 × 4.4 in), and `presentation` (10 × 5.625 in).

> SVG is vectorial and therefore does not itself have a DPI resolution.

The theme's `savefig_dpi=2100` is recorded for rasterized artists or optional
raster exports. The current report writes SVG only and does not rasterize its
lines, bars, markers, axes, annotations, or text.

## Common workflows

### Change the fuzzy threshold and scorer

```python
from lab.ner.analysis import AnalysisConfig, analyze_evaluation

analysis = analyze_evaluation(
    "predictions.tsv", "evaluation_gold.tsv", "training_gold.tsv",
    "analysis_output", run_id="threshold-090",
    config=AnalysisConfig(fuzzy_threshold=0.90, fuzzy_scorer="token_sort_ratio"),
)
```

### Analyze an existing artifact without inference

```python
from lab.ner.analysis import load_evaluation

analysis = load_evaluation("analysis_output/evaluation.parquet")
print(analysis.zero_shot_errors()[["gold_text", "gold_label", "error_primary"]])
```

### Regenerate figures with bootstrap disabled

```python
from lab.ner.analysis import BootstrapConfig, PublicationConfig, create_publication_report

report = create_publication_report(
    "analysis_output/evaluation.parquet",
    config=PublicationConfig(bootstrap=BootstrapConfig(enabled=False)),
)
```

### Run fixed-seed bootstrap intervals only

```python
from lab.ner.analysis import BootstrapConfig, bootstrap_intervals, load_evaluation

analysis = load_evaluation("analysis_output/evaluation.parquet")
intervals = bootstrap_intervals(
    analysis.events,
    BootstrapConfig(samples=5000, confidence_level=0.95, seed=42),
)
intervals.to_parquet("analysis_output/tables/custom_bootstrap.parquet", index=False)
```

## How to interpret the analysis

* **High seen and lower zero-shot performance:** observed performance is
  associated with lexical exposure. This is not proof that the model memorizes.
* **Character F1 substantially above STRICT F1:** predictions often cover much
  of the correct labeled character region but fail exact span/label matching.
  Inspect boundary categories; character overlap never changes STRICT credit.
* **Many exact-boundary label errors:** span localization often agrees while
  label assignment contributes to remaining strict errors. Inspect the raw and
  row-normalized confusion tables and training-label ambiguity.
* **Many fuzzy-zero-shot misses:** unpaired misses are enriched among mentions
  below the configured lexical-similarity threshold. This does not establish
  that novelty caused the misses.
* **Large boundary oracle ΔF1:** exact localization represents a large measured
  counterfactual ceiling and is a candidate for targeted investigation.
* **Large zero-shot-miss oracle ΔF1:** lexically novel misses account for a
  substantial portion of the available strict error budget; this remains a
  diagnostic association, not a forecast of an intervention.
* **High-confidence errors:** high model scores co-occur with strict errors.
  Scores are not assumed calibrated; prioritize manual inspection rather than
  interpreting them as probabilities.
* **Overlapping/nested degradation:** compare complete official subgroup metrics
  and support. The association does not prove overlap caused the errors.

## Troubleshooting

| Symptom | Actual cause and action |
|---|---|
| `python -m lab.ner.analysis --help` fails | Expected: no analysis `__main__` or Typer CLI exists. Use the Python API. |
| `lab tasks` does not list analysis | Expected: no analysis task is registered in `src/lab/ner/tasks.py`. |
| `Span input does not exist` | Check the path passed to an API function. Relative paths resolve from the process working directory. |
| `Span frame is missing columns ...` | Supply `filename`, `label`, `start_span`, `end_span`, and `text`. A corpus Parquet with `entities_json` is not a span table. |
| `end_span <= start_span` | The existing span validator rejects zero/inverted spans before official evaluation. Use `audit_annotations` separately to inspect raw invalid annotations. |
| Prediction score error | Scores, when present, must be numeric, non-null, finite in practice, and within `[0, 1]`. Remove the column if confidence is unavailable. |
| Empty predictions | Supported when the frame still has the required columns; all gold entities become FNs. |
| Empty training gold | Supported with required columns; mentions become unseen and nearest-neighbor fields are null. |
| Unknown label appears | With `tags=None`, labels are inferred from gold/predictions. If passing `tags`, include every label intended for evaluation. |
| Invalid fuzzy scorer | Use one of `ratio`, `WRatio`, `partial_ratio`, `token_sort_ratio`, or `token_set_ratio`. |
| Threshold rejected | `fuzzy_threshold` must be in `[0, 1]`, not `[0, 100]`; use `0.80`, not `80`. |
| Tokenization columns are null | Expected when no tokenizer object was passed. `tokenization_available` is false. |
| Confidence figures/sweep have no data | Predictions lacked `score`, or `confidence_thresholds` was empty. |
| Annotation issues say `DOCUMENT_UNAVAILABLE` | Pass document mappings/DataFrames/Parquets to the corresponding document parameters. This availability issue does not mark the annotation invalid by itself. |
| Duplicate predictions | They remain detectable through `error_duplicate_prediction`; review upstream serialization if duplicates were unintended. |
| Artifact metadata missing | The file is an ordinary Parquet, not an analysis artifact produced by this package. |
| Unsupported schema | Loader accepts only declared supported versions (`1.0.0`, `2.0.0`). Regenerate with the current analysis code or write a deliberate migration. |
| Figure reports “No eligible data” | The source group is genuinely unavailable, often because tokenization, confidence thresholds, or confusion observations are absent. |

Python APIs raise `ValueError`, `FileNotFoundError`, `RuntimeError`, or PyArrow/
pandas errors directly. There is no CLI exception translation, logging setup,
or exit-code contract for analysis.

## Performance and reproducibility

Efficiency decisions include deduplicated normalized training candidates,
cached repeated nearest-neighbor lookups, no dense evaluation × training
matrix, compact nullable Arrow types, dictionary encoding, and ZSTD. For large
corpora, reuse `evaluation.parquet` for report regeneration rather than rerun
exposure and diagnostic construction. Bootstrap precomputes official counts per
document/stratum and resamples those counts rather than invoking the evaluator
for every sample.

The analysis Parquet records schema version, run ID, official evaluator name and
metrics, exact analysis configuration (including RapidFuzz scorer/threshold and
normalization), similarity backend/cache statistics, and training SHA-256 and
counts. The publication manifest additionally records the installed `lab`
version, current Git commit when available, bootstrap and publication settings,
plot theme version, canonical source path, figure/table names, and optional
`model_id`, `dataset`, and `split` values if those keys already exist in artifact
metadata. The current `analyze_evaluation` signature does not accept model,
dataset, or split parameters, so those manifest values are normally null unless
metadata was enriched elsewhere.

## Developer extension guide

### CLI extension status

There is no analysis Typer layer to extend. Do not add documentation for a
command before implementing and registering it. If a future Typer interface is
introduced, keep it under `src/lab/ner/analysis/`, make command functions thin,
construct the existing dataclasses explicitly, and delegate all work to public
analysis functions. Decide separately whether it should be a module entry point
or integrate with the project's current `argparse` task runner; those mechanisms
do not currently compose automatically.

Conceptual future pattern (not runnable current code):

```python
# cli.py — only if a future CLI is deliberately added
def command(...):
    config = AnalysisConfig(...)
    return analyze_evaluation(..., config=config)
```

Business logic belongs in `api.py`, `subgroups.py`, `oracles.py`,
`statistics.py`, or another analysis service—not in an interface callback.
Future CLI tests should invoke root and per-command `--help`, check important
option names structurally, and run one temporary-directory workflow without
snapshotting full help text.

### Adding an analytical slice

1. Define how the property applies independently to gold and prediction spans.
2. Add the property in `_properties` in `subgroups.py` (or a reusable feature
   module) and stable values/bins in `config.py` if configurable.
3. In `subgroup_analyses`, create both masks and call `score_subgroup`.
4. Never calculate subgroup F1 by filtering only canonical TP/FN rows:
   `score_subgroup` must receive the subgroup's complete gold and prediction
   frames so it calls `span_metrics` with valid denominators.
5. Add a synthetic test with known TP/FP/FN/P/R/F1 and an FP whose property
   demonstrates the prediction denominator is included.

This pattern applies to dosage expressions, document-start entities, or other
linguistic characteristics.

### Adding a figure

1. Reuse `PlotTheme`, `OKABE_ITO`, `SIZE_PRESETS`, and `SVGFigure` from
   `plotting.py` or add a generic primitive there.
2. Keep stable category ordering; do not rely on set/hash iteration.
3. Preserve white background, no default grid, omitted top/right spines, and
   legends outside the data region.
4. Write deterministic `.svg` output with editable text. Do not describe SVG as
   2100 DPI.
5. Register the deterministic filename in `_figures` in `publication.py` and a
   matching factual caption in `_captions`.
6. Test SVG structure and theme properties, not pixels.

### Documentation drift guard

`verification/verify_analysis_documentation.py` checks that this guide states
the absence of a Typer CLI, names every exported analysis API and every current
Parquet schema column, and does not claim a runnable module entry point. If a
CLI is added later, update both implementation and this guide together.
