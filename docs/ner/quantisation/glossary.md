# Glossary

[Home](README.md) · [Methods](methods/README.md) · [Measurements](measurements.md)

* **Artifact** — New directory containing transformed representation, tokenizer/config, provenance,
  and optionally copied NER encoding.
* **Backend** — Implementation/storage executor named in a manifest, e.g. `transformers`,
  `pytorch_dynamic_int8`, `pytorch_low_rank`, `onnx`.
* **Calibration** — Representative-data collection used to choose quantization scales. Not
  implemented for static INT8 here.
* **Cold start** — Model load plus the measured end-to-end pipeline sum.
* **Compression category** — Internal taxonomy: model compression, computational compression, or
  runtime optimization.
* **Distillation** — Student training from teacher outputs/states. Loss/spec scaffolding exists;
  execution does not.
* **Execution provider** — Concrete runtime device/provider (for example ONNX CPU/CUDA provider).
  Recorded as a factor; no ONNX provider adapter exists.
* **Manifest** — `compression_manifest.json`, recording source hash, method/backend, dimensions,
  versions, references, and details.
* **Pareto front** — Non-dominated F1/resource points; no combined score.
* **Per-channel quantization** — Separate scale per output/input channel. Not user-configurable here.
* **Per-tensor quantization** — One scale for a tensor. Not user-configurable here.
* **PTQ** — Post-training quantization. Dynamic INT8 is the implemented PTQ form.
* **QAT** — Quantization-aware training. Not implemented.
* **Runtime** — Execution family recorded in experiment rows, normally `pytorch` in implemented NER runs.
* **Steady state** — Measured execution after warm-up; exact region is documented per timing field.
* **Structured pruning** — Removal of shape-aligned structures. Metadata/scaffold only here.
* **Throughput** — Documents, sentence segments, or encoded input tokens per measured end-to-end second.
* **Unstructured pruning** — Individual weight values set to zero; implemented magnitude pruning remains dense.
* **Vocabulary pruning** — Removal of existing tokens/embedding rows. Implemented only as safe WordPiece tail removal.
* **Warm-up** — Unmeasured direct batches before timing/peak reset.
* **W8A8** — Eight-bit weights and activations. The current dynamic INT8 command is not exposed as a configurable W8A8 scheme.
* **Weight-only quantization** — Quantized weights with non-quantized activation computation. No configurable weight-only backend is exposed.
