# Quantisation hardening report

## Executable additions

* Deterministic calibration-based CPU reference W8A8 stores INT8 weights and activation ranges, records calibration provenance, and loads through canonical NER inference.
* Portable CPU weight-only INT4 uses genuine nibble packing, per-group scales and optional zero points, a specialized state-dictionary loader, and excludes the classifier by default.
* Structural pruning physically invokes Hugging Face head pruning and slices BERT/RoBERTa FFN channels.
* Student construction builds a valid token-classification architecture with deterministic evenly spaced depth mapping and a tensor initialization report.
* Distillation now provides masked, temperature-correct logit KL, hidden-state projection support, attention matching, a frozen teacher, and an executable batch training loop.

These reference quantization backends prioritize artifact correctness and offline portability. They dequantise for floating Linear compute and therefore make no optimized latency claim.

## Contracts and integration

Manifest schema 2 records an artifact ID, timestamps, source/result counts and bytes, representation, supported devices, runtime requirements, seed, detailed backend metadata, and retains legacy fields. Canonical `lab.ner.inference.load_model` dispatches dynamic INT8, reference static INT8, reference INT4, and low-rank artifacts. New recipe fields are explicitly typed and unknown fields are rejected.

Results retain compatibility TSV and now also write canonical ZSTD Parquet. Plot data are ZSTD Parquet and rendered figures are white-background, grid-free, Okabe–Ito-blue SVG. Pareto exports have both TSV and Parquet representations.

## Explicit limitations

There is no complete corpus-to-artifact distillation CLI, recovery fine-tuning orchestrator, QAT, optimized INT4/W8A8 kernel, or C-cell pipeline runner. Initialized students must not be reported as distilled models. Specialized quantized loaders support the standard linear token-classification path, not the custom CRF architecture. The definitive executable matrix in `docs/ner/quantisation/matrix.md` records these limits.

## Verification environment

Sources compile and dependency-layer checks run without optional ML dependencies. Full torch/Transformers tests could not run in this container because the locked PyTorch/CUDA wheels and Pydantic core were not present in the offline cache. No model or dependency was downloaded.
