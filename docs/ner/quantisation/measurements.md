# Timing, memory, throughput, and energy methodology

[Home](README.md) · [Experiments/results](experiments.md) · [Scientific analysis](analysis.md)

## Timing regions

| Field | Region actually measured |
|---|---|
| `load_time_s` | Canonical model/tokenizer/`encoding.json` loading and device placement. |
| `preprocessing_time_s` | `read_documents`; parquet/text loading and canonical corpus validation. |
| `tokenization_time_s` | `encoder_from_description(...).encode(corpus)`, including segmentation/windowing/tokenization. |
| `h2d_time_s` | Moving already-collated tensor batches to CUDA. NA on CPU. |
| `model_time_s` | One no-grad direct forward pass over cached device batches, excluding collator and transfer. |
| `cuda_event_model_time_s` | CUDA Event elapsed seconds around the same direct forward; stored in `timing.json`/`run.json` but not a `benchmark_long.tsv` column. |
| `pipeline_inference_time_s` | Existing `predict_logits`, including Trainer creation/data path and returning logits. |
| `postprocessing_time_s` | Existing `decode_spans`. |
| `end_to_end_time_s` | preprocessing + tokenization + pipeline inference + postprocessing. It excludes model load, prediction file writing, and metric scoring. |
| `cold_start_time_s` | load + the end-to-end sum. |

“Steady state” here means the measured model forward follows warm-up. The complete pipeline runs
once per repetition and includes its normal Trainer initialization; do not interpret
`end_to_end_time_s` as a per-batch steady-state kernel statistic.

## Warm-up and CUDA correctness

The worker collates/transfers fixed rows, executes `warmup_batches` direct batches (cycling if
needed), synchronizes, then resets CUDA peak statistics. Warm-up is excluded from measured forward.

CUDA launches asynchronously. The shared timer calls `torch.cuda.synchronize(device)` immediately
before and after timed functions. Direct forward additionally records CUDA Events. Wall time answers
host-observed completion; Event time answers device-stream elapsed execution. Neither should be
replaced by unsynchronized Python launch time.

## Throughput and workload

Documents/s, sentence segments/s, and input tokens/s all divide by measured end-to-end time.
`n_tokens` sums encoded row lengths (including each row's special tokens). `n_padded_tokens` models
per-batch maximum padding rounded to `pad_to_multiple_of`. Vocabulary changes can increase these
counts, so compare tokens/s as well as documents/s.

## Memory

| Metric | Source |
|---|---|
| CPU RSS | `psutil.Process().memory_info().rss`; `resource.ru_maxrss` fallback when psutil is absent. |
| CUDA allocated | Live tensor bytes known to PyTorch allocator. |
| CUDA reserved | Allocator-managed reserved pool, including unused blocks. |
| Peak allocated/reserved | Maximum since reset immediately after warm-up. |
| Checkpoint bytes | Recursive serialized artifact-directory bytes measured independently. |

PyTorch allocator values do not include every driver/runtime allocation, and CPU fallback peak RSS
is not instantaneous RSS. Parameter count does not determine optimizer state (none in inference),
dtype, activations, allocator fragmentation, tokenizer buffers, runtime engine memory, or sequence
length; it therefore does not determine RSS/VRAM.

## CodeCarbon

Energy is optional and instantiated only when `energy.enabled: true`. The adapter uses
`EmissionsTracker` with process tracking, visible GPU IDs, one-second default sampling, logger-only
output, and multiple-run support.

### `model_only`

Rows are tokenized/collated/transferred once. The worker repeatedly executes direct model forward
on identical cached batches. Loading, tokenization, transfer, decoding, and file output are outside
the region.

### `end_to_end`

For each pass, the worker rereads the dataset, encodes it, runs canonical prediction, and decodes
spans. Model loading/downloading is outside. Prediction/gold file writing and quality scoring are
also outside.

Each mode executes at least once and repeats until `min_duration_seconds`. The predictive result is
not recomputed/changed by these energy passes. Raw mode dictionaries record:

* `cpu_energy_kwh`, `gpu_energy_kwh`, `ram_energy_kwh`, `total_energy_kwh` when CodeCarbon exposes them;
* `duration_s`, `co2_kg`;
* passes, documents, and tokens processed;
* `joules_per_1000_documents`, `joules_per_1000_tokens`, and `wh_per_1000_documents`.

Normalization converts measured total kWh (`1 kWh = 3,600,000 J`). Missing measurements remain
null. Energy is the primary hardware/workload observation. CO2e additionally depends on geographic
carbon-intensity assumptions and is secondary for model comparison.

When both modes exist, the flattened long row prefers end-to-end values and retains both complete
mode records in `run.json`/`energy.json`; never mix them in one claim.

## Initialization and special runtimes

Dynamic INT8 is CPU-only. ONNX export is not accepted by the PyTorch worker. The code has no
OpenVINO, TensorRT, or `torch.compile` experiment adapter, so no compilation/engine timing is
reported. EL energy and component limitations are documented separately in
[Entity Linking](entity-linking.md).
