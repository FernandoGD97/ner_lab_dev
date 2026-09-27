# Phase 2 implementation report

## Files added and modified

`experiment/` adds validated configuration, matrix readiness, subprocess orchestration, worker
instrumentation, result schemas/statistics, environment capture, tokenizer analysis, and a
CodeCarbon adapter. The quantisation CLI gained a nested experiment group. Canonical NER model
loading now recognizes Phase 1 dynamic-INT8 and SVD runtime artifacts. Artifact creation copies the
source `encoding.json`, retaining the canonical inference contract. Documentation and CPU-only
verification checks were added.

## CLI and configuration

`lab quantisation experiment` provides `validate`, `run`, `status`, `summarize`, and `compare`.
Run supports `--dry-run`, `--resume`, and `--force`. Pydantic models validate controlled versus
maximum-throughput protocols, fixed inference conditions, energy modes, unique matrix IDs, and
positive sizing/repetition values. Manifest and recipe paths are discovered automatically.

## Timing and CUDA methodology

Each model × repetition executes in a fresh subprocess. Load, preprocessing, encoding/tokenization,
prediction, postprocessing, steady-state end-to-end, and cold-start durations are distinct.
Configurable warm-up is excluded. Inputs are collated on the host before a separately timed
transfer. CUDA wall timers synchronize on both boundaries; CUDA Events also measure cached-input
model-forward GPU elapsed time.

## Memory, throughput, and tokenizer effects

Workers observe process RSS and current/peak CUDA allocated and reserved memory after resetting
peaks following warm-up. They independently count logical parameters, observable nonzero dense
parameters, serialized bytes, documents, sentences, input/padded tokens, and three throughput
rates. Tokenizer statistics expose document quantiles, subwords/word, subwords/entity, and entity
fragmentation.

## CodeCarbon and energy methodology

The adapter follows the repository's existing current CodeCarbon `EmissionsTracker` API. Separate
end-to-end and cached-input/model-only regions repeat fixed work until the minimum duration. Raw
component/total energy, duration, emissions, passes, and processed workload are retained. Joule and
Wh normalizations derive only from observed energy; unavailable values remain null.

## Results and baseline comparisons

Per-run JSON remains authoritative. The stable long TSV contains identity, runtime factors,
workload, model, quality, timing, throughput, memory, energy, and status fields. Summary rows carry
mean, median, standard deviation, p50/p95, and 95% confidence bounds. A0 comparisons independently
calculate parameter/checkpoint/VRAM/energy reduction, strict metric deltas, model prediction
speedup, and end-to-end speedup.

## Tests executed

CPU-only verification covers timing/memory schemas, energy normalization, OOM classification,
tokenizer statistics, aggregation, confidence columns, result files, speedups/reductions, baseline
deltas, environment metadata, CLI nesting, registry metadata, and repository layering. Pydantic
configuration and actual model execution need the declared dependencies.

## Small validation experiment

The complete A0/A1/A3 experiment could not be executed in this checkout: its pre-existing virtual
environment does not contain PyTorch, Pydantic, datasets, or CodeCarbon, and package download was
unavailable. No GPU, memory, latency, energy, or CO2 values were fabricated. Output generation,
metric calculations, and subprocess CLI wiring are exercised by CPU-only tests; an environment with
`pip install -e '.[ner]'` is required for the representative model run.

## Known limitations and Phase 3

Host-to-device timing covers pre-collated tensors, while canonical Trainer prediction is retained
as a separate complete-prediction interval. Dynamic INT8 is CPU only; ONNX remains an explicitly
unsupported runtime in the canonical PyTorch pipeline. Energy
component availability depends on CodeCarbon and hardware. Normal confidence intervals are a
simple descriptive summary for five repetitions. Phase 3 may add runtime-specific adapters,
advanced uncertainty, long-tail/entity analysis, Pareto analysis, deployment batch searches, and
Entity Linking experiments; none are implemented here.
