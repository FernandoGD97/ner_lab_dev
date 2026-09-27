# Controlled experiments, result files, resume, and comparison

[Home](README.md) · [Configuration](configuration.md) · [Measurements](measurements.md) · [Analysis](analysis.md)

## Matrix labels

The validator recognizes these **experiment labels**, not algorithm implementations:

| IDs | Intended study cells |
|---|---|
| A0–A7 | FP32, FP16, BF16, dynamic/static INT8, INT4, vocabulary pruning, vocabulary+INT8 |
| B1–B8 | depth/width/FFN students, structured pruning, distillation |
| C1–C6 | combined distilled/pruned/vocabulary/INT8 variants |

An ID does not create an artifact or prove that its intended method was used. Metadata comes from
the path/manifest. A4/A5 and most B/C recipes are experimental, unsupported, or require training.

## Controlled versus maximum-throughput

Controlled mode requires A0, one runtime/provider, and the same YAML inference settings for all
models. Per-model batch size is forbidden. Document order comes from the same dataset read;
preprocessing comes from each artifact's saved `encoding.json`, whose `max_length` must match YAML.
Padding, device, warm-up, repetitions, canonical prediction/postprocessing, and subprocess model
are fixed.

Maximum-throughput mode permits `models[].batch_size`. It does not tune batches/runtimes, and its
rows remain labeled `maximum_throughput`; do not aggregate them with controlled rows.

## Execution order and isolation

Model IDs are seeded-shuffled once, then rotated per repetition to counterbalance first-position
thermal/DVFS bias. The actual order is stored. Each model × repetition receives a payload and fresh
Python subprocess. The parent captures stdout/stderr. OOM never triggers a smaller-batch retry.

## Directory layout

```text
RESULTS_PARENT/EXPERIMENT_ID/
├── resolved_experiment.yaml
├── environment.json
├── execution_order.json
├── study_manifest.json
├── benchmark_long.tsv
├── benchmark_summary.tsv
├── comparison.tsv
├── models/MODEL_ID/
│   ├── predictions/{predictions.tsv,gold.tsv}
│   ├── model_metadata.json
│   ├── metrics.json
│   ├── tokenizer_statistics.json
│   └── runs/run_001/
│       ├── payload.json
│       ├── run.json                 # source of truth
│       ├── timing.json
│       ├── memory.json
│       ├── energy.json
│       ├── energy.csv               # only when energy produced data
│       ├── stdout.log
│       └── stderr.log
└── analysis/                        # created by Phase 3 commands
```

Predictions/metrics are model-level and are rewritten by repetitions; per-run `run.json` is the
source of truth for benchmark observations. Long/summary/comparison tables are derived and can be
regenerated.

## File reference

| File | Granularity / purpose |
|---|---|
| `resolved_experiment.yaml` | Fully default-expanded Pydantic configuration. |
| `environment.json` | One study snapshot: timestamp, host/OS/kernel/CPU/RAM, GPU count/models/driver when available, CUDA/cuDNN, Python/package versions, Git commit, runtime/provider, selected env vars. |
| `execution_order.json` | One ordered entry per repetition × model, with position. |
| `study_manifest.json` | Study lifecycle, protocol, readiness, output paths, terminal status counts. |
| `models/*/runs/*/run.json` | Authoritative terminal or in-progress run record. Includes flattened long metrics plus tracebacks/errors where relevant. |
| `timing.json` | One run's timing fields. |
| `memory.json` | One run's RSS and current/peak CUDA allocator measurements. |
| `energy.json` | Separate `end_to_end`/`model_only` dictionaries. |
| `energy.csv` | Same available mode rows in tabular form. Not created when energy is disabled/empty. |
| `metrics.json` | Existing NER span metrics for a model prediction. |
| `benchmark_long.tsv` | One row per terminal model × repetition × split. Stable wide schema below. |
| `benchmark_summary.tsv` | One row per model × metric: count, mean, median, sample std, p50/p95, normal 95% CI. SUCCESS only. |
| `comparison.tsv` | One row per model relative to baseline means. SUCCESS only. |

## Complete `benchmark_long.tsv` column dictionary

| Columns | Meaning |
|---|---|
| `experiment_id`, `run_id`, `model_id`, `artifact_id` | Study/run/model identity; artifact ID currently equals model ID. |
| `source_model`, `method_chain`, `precision` | Manifest-derived source, JSON method list, dtype/precision description. |
| `model_representation`, `runtime`, `backend`, `execution_provider`, `protocol` | Explicit runtime factors; never collapse them into “compression”. |
| `task`, `dataset`, `split` | Task and data identity. |
| `batch_size`, `max_length` | Effective run controls. |
| `n_documents`, `n_sentences`, `n_tokens`, `n_padded_tokens` | Workload counts; tokens include model input rows. |
| `parameters`, `nonzero_parameters`, `checkpoint_bytes` | Logical count, observable dense nonzeros (NA for packed specialized weights), recursive artifact bytes. |
| `vocab_size`, `hidden_size`, `intermediate_size`, `layers`, `attention_heads` | Architecture/config dimensions. |
| `precision_score`, `recall_score`, `f1_score` | Existing strict span metrics; NA without gold. |
| `load_time_s` | Model/tokenizer/encoding load. |
| `preprocessing_time_s` | Corpus reading/validation. |
| `tokenization_time_s` | Saved encoder's corpus encoding. |
| `h2d_time_s` | Pre-collated tensor transfer; NA on CPU. |
| `model_time_s` | Direct cached-input forward over all batches. |
| `pipeline_inference_time_s` | Canonical `predict_logits` interval, including Trainer/data handling. |
| `postprocessing_time_s` | Span decoding. |
| `end_to_end_time_s` | Preprocessing + tokenization + canonical prediction + postprocessing; excludes load/write/scoring. |
| `cold_start_time_s` | Load + the preceding end-to-end sum. |
| `documents_per_s`, `sentences_per_s`, `tokens_per_s` | Counts divided by end-to-end time. |
| `cpu_rss_mb` | Process RSS (or platform peak-RSS fallback). |
| `gpu_allocated_mb`, `gpu_reserved_mb` | Current allocator state after measured work. |
| `gpu_peak_allocated_mb`, `gpu_peak_reserved_mb` | Peaks after warm-up reset. |
| `cpu_energy_kwh`, `gpu_energy_kwh`, `ram_energy_kwh`, `total_energy_kwh` | CodeCarbon observations when available. |
| `co2_kg`, `energy_mode` | Derived emissions and selected flattened primary mode (`end_to_end` preferred). |
| `joules_per_1000_documents`, `joules_per_1000_tokens`, `wh_per_1000_documents` | Work-normalized observed total energy. |
| `status` | Terminal status represented in the row. |

## Comparison formulas

`reduction_pct = (baseline - candidate) / baseline × 100` and
`speedup = baseline_time / candidate_time`. Forward and end-to-end speedup are separate.
Precision/recall/F1 deltas are candidate minus baseline. Empty/zero denominators yield missing data,
not infinity.

## Resume and statuses

Statuses are `PENDING`, `RUNNING`, `SUCCESS`, `FAILED`, `OOM`, `SKIPPED`, and `UNSUPPORTED`.
Workers actually write `RUNNING`, then a terminal state; `PENDING` is defined but not currently
persisted by orchestration. Resume considers a run complete only when its readable `run.json`
contains exactly `"status": "SUCCESS"`. Failed, OOM, incomplete, and corrupt runs are eligible to
rerun. Use:

```bash
lab quantisation experiment status results/study
lab quantisation experiment run study.yaml --resume
```

If configuration changed materially, prefer a new experiment ID or `--force`; resume does not
compare old payload/config hashes. Save failed logs before force, because force deletes the entire
result root.
