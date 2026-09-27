# Controlled compression benchmarking

[Home](README.md) · [Experiments](experiments.md) · [Measurements](measurements.md)

## Scientific contract

The primary protocol compares A0 and compressed artifacts with one fixed dataset and split,
document order, preprocessing and postprocessing, batch size, maximum sequence length, padding,
device, runtime family, worker count, warm-up, repetitions, and canonical `lab.ner.inference`
functions. A model that cannot satisfy that contract is recorded as failed, unsupported, or OOM;
the runner never lowers its batch size. `maximum_throughput` is a separate deployment protocol
that permits per-model batch sizes, and its rows carry a different `protocol` value. Never pool
the two protocols.

Parameter reduction is not checkpoint reduction, memory reduction, latency reduction, or energy
reduction. Each is observed independently. In particular, a sparse dense tensor can contain fewer
nonzero values without changing its parameter count or speeding up its kernel, and a smaller
vocabulary can create longer sequences.

## Configuration and readiness

The experiment YAML is validated with Pydantic. Model entries contain only identity, artifact path,
and explicit runtime factors; compression metadata is read from `compression_manifest.json` and
`compression_recipe.yaml`. The complete A0–A7, B1–B8, and C1–C6 ID vocabulary is recognized.
Before execution each artifact is classified `READY`, `MISSING`, `INVALID`, or `UNSUPPORTED`.
Normal NER artifacts must contain `encoding.json`, because it is the canonical pipeline's source
of preprocessing settings.

```yaml
experiment: {id: biomedical_ner_compression_v1, seed: 42}
dataset: {path: assets/test.parquet, split: test}
inference:
  batch_size: 16
  max_length: 256
  device: cuda:0
  warmup_batches: 10
  repetitions: 5
  pad_to_multiple_of: 8
  number_of_workers: 0
models:
  - {id: A0, path: models/original, runtime: pytorch}
  - {id: A1, path: models/fp16, runtime: pytorch}
energy:
  enabled: true
  modes: [end_to_end, model_only]
  min_duration_seconds: 60
output: {directory: results}
protocol: controlled
```

```bash
lab quantisation experiment validate experiment.yaml
lab quantisation experiment run experiment.yaml --dry-run
lab quantisation experiment run experiment.yaml
lab quantisation experiment run experiment.yaml --resume
lab quantisation experiment run experiment.yaml --force
lab quantisation experiment status results/biomedical_ner_compression_v1
lab quantisation experiment summarize results/biomedical_ner_compression_v1
lab quantisation experiment compare results/biomedical_ner_compression_v1
```

`--resume` skips only valid `SUCCESS` run records. `--force` deletes and recreates the experiment
directory. The options are mutually exclusive. Every model × repetition runs in a fresh Python
subprocess, so allocator state, failures, and cleanup are isolated. The parent stores stdout,
stderr, payload, and traceback. Statuses are `PENDING`, `RUNNING`, `SUCCESS`, `FAILED`, `OOM`,
`SKIPPED`, and `UNSUPPORTED`.

## Timing

The worker composes the existing inference primitives: `load_model`, `read_documents`, the saved
encoder, `predict_logits`, `decode_spans`, and the existing strict span scorer. It does not carry a
second NER implementation. It measures load, corpus preprocessing, encoding/tokenization, model
pure cached-input model forward, canonical prediction, and postprocessing separately.
`end_to_end_time_s` is the steady-state pipeline sum
without model loading; `cold_start_time_s` adds load. Host-to-device time remains null where the
Transformers Trainer owns transfers and cannot expose them honestly.

Warm-up batches execute before peak reset and measured prediction. Warm-up is never included in
steady-state latency. Collation happens before a separately timed host-to-device transfer. CUDA
wall-clock regions synchronize before and after. A CUDA Event pair also
records GPU elapsed prediction time; wall time is still retained because event time and complete
pipeline time answer different questions. Compilation/engine initialization belongs in load or an
explicit future runtime adapter, never in repeated forward latency.

Execution order starts from a seeded shuffle and rotates it on successive repetitions. This is
deterministic and counterbalances first-position thermal, DVFS, and clock-drift effects. Every
repetition remains an individual source-of-truth record.

## Memory and throughput

CPU RSS is read from the process (with a portable peak-RSS fallback). CUDA allocated, reserved,
peak allocated, and peak reserved bytes come from the PyTorch allocator. CUDA peak statistics are
reset after warm-up. Parameter count, nonzero parameter count, and serialized checkpoint bytes are
stored separately; actual VRAM is never estimated from parameters.

The runner records documents, sentence segments, model input tokens, padded input tokens, and
throughput in documents/s, sentences/s, and tokens/s. Padding follows the fixed batch size and
`pad_to_multiple_of`. Tokenizer statistics include mean, median, and p95 tokens/document, mean
subwords/word, mean subwords/entity, and the entity-fragmentation distribution.

## Quality

Strict precision, recall, and F1 are read from the repository's existing `span_metrics` output.
The full existing exact, partial, entity-type, and character metrics are preserved in each model's
`metrics.json`. Predictions and gold spans use the normal NER TSV writers.

## Energy and CodeCarbon

The adapter uses the same current `EmissionsTracker` process mode and visible-GPU handling as NER
training. End-to-end mode repeats encoding, inference, and decoding; model-only mode keeps encoded
rows fixed and repeats prediction. Modes are stored separately and are never mixed. Model loading
and downloading occur before tracking.

An energy loop repeats the fixed evaluation workload until `min_duration_seconds`; this does not
change predictive evaluation. It records passes, processed documents/tokens, duration, CPU/GPU/RAM
and total kWh, and CO2e when CodeCarbon supplies them. Energy is primary. CO2e is secondary because
it depends on carbon intensity. Normalizations are measured kWh converted to J/1000 documents,
J/1000 tokens, and Wh/1000 documents. Missing hardware/API observations stay null; they are never
fabricated.

## Runtime confounding

Every row names representation, precision, runtime, backend, execution provider, and protocol.
A PyTorch FP32/ONNX INT8 pair cannot isolate precision from runtime. Make within-runtime comparisons
(FP32 PyTorch vs INT8 PyTorch; FP32 ONNX vs INT8 ONNX) before attributing a change to compression.
ONNX is currently classified unsupported by the canonical PyTorch NER path rather than silently
using a different inference implementation.

## Outputs and statistics

The result root contains resolved configuration, environment, execution order, study manifest,
per-model metadata/metrics/tokenizer statistics/predictions, and per-run timing, memory, energy,
logs, and `run.json`. Individual run JSON is the source of truth.

`benchmark_long.tsv` has one stable row per model × repetition × split. `benchmark_summary.tsv`
has one model/metric row with count, mean, median, sample standard deviation, p50, p95, and a normal
95% confidence interval. `comparison.tsv` defaults to A0 and independently reports parameter,
checkpoint, peak VRAM, and energy reductions; strict quality deltas; and model-forward and complete
pipeline speedups. Speedup is `baseline_time / candidate_time`.

## Reproducibility

`environment.json` captures UTC timestamp, host, OS/kernel, CPU/RAM, visible GPUs, driver when
available, CUDA/cuDNN, Python, PyTorch, Transformers, tokenizers, CodeCarbon, Git commit, runtime,
provider, and defined `CUDA_VISIBLE_DEVICES`, `OMP_NUM_THREADS`, and `MKL_NUM_THREADS`. The resolved
YAML and actual seeded execution order make the run reconstructable. Hardware load and ambient
conditions remain external variables and should be reported by the study author.
