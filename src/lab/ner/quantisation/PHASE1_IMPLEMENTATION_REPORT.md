# Phase 1 implementation report

## Architecture

The framework lives inside `lab.ner.quantisation`. A unified `CompressionMethod` API separates
compatibility, application, and validation; a registry supplies structured scientific metadata
to Python and CLI callers. Model inspection, artifact provenance, recipe validation, and reload
validation are independent modules. Technique families are split into precision, quantization,
pruning, architecture, distillation, low-rank, and runtime packages. Imports of heavy optional
runtime libraries remain local.

## Files added

Core modules: `metadata.py`, `base.py`, `registry.py`, `model.py`, `artifacts.py`, `validation.py`,
`config.py`, and `cli.py`. Family modules implement or describe precision, INT8/INT4,
weight/vocabulary pruning, depth/width/FFN changes, distillation losses/students/trainer, SVD,
and ONNX. `references/bibliography.yaml` is the reference registry; `recipes/` contains A0–A7,
B1–B8, and C1–C6. Documentation is under `docs/ner/quantisation/`; checks are under
`verification/quantisation/`.

## CLI commands

The repository's central `lab` argparse application now owns `quantisation` (and US spelling
alias `quantization`) with `inspect`, `methods`, `method`, `fp16`, `bf16`, `int8`,
`prune-vocabulary`, `prune`, `reduce-depth`, `svd`, `export-onnx`, `run-recipe`, and `validate`.
A Typer `app` exposes the same API for embedding by Typer users. Compression logic is not
implemented in command functions.

## Method status

Implemented standard HF artifacts: FP16, BF16, unstructured magnitude pruning, safe tail-only
WordPiece vocabulary pruning, and experimental depth reduction. Implemented specialized-runtime
artifacts: CPU dynamic INT8, SVD low-rank factorization, and ONNX export. FP32 is baseline
metadata. Static INT8, QAT, INT4, structured pruning, width/FFN students, and combined pipelines
are partial scaffolds or recipes. Distillation provides student validation and modular logits,
hidden-state, and attention losses, but deliberately does not start expensive training.

## Scientific references

Each registered method has publications and fidelity metadata. The bibliography covers KD,
DistilBERT, TinyBERT, MobileBERT, MiniLM, ALBERT, Q-BERT, TernaryBERT, I-BERT, BinaryBERT,
ZeroQuant, Movement/Block Pruning, CoFi, PoWER-BERT, DeeBERT, PABEE, KroneckerBERT,
multilingual vocabulary work, compact biomedical Transformers, GPTQ, SmoothQuant, and AWQ.
Generic methods are not presented as reproductions of named papers.

## Tests

Dependency-light verification covers registry metadata, reference links, unsupported method
errors, Pydantic loading of every study recipe, and integration with the central CLI. The
validation implementation additionally performs AutoConfig/tokenizer/model reload, task and
label preservation, vocabulary/embedding checks, forward smoke inference, finite values, and
shape checks. Transformation checks require PyTorch and are designed for tiny local HF fixtures;
this checkout's base environment does not install the `ner` extra.

## Known limitations

XLM-R/SentencePiece vocabulary mutation is analysis-only; arbitrary remapping cannot safely
preserve IDs. Dynamic INT8 and SVD need their documented specialized loader; ONNX needs an ONNX
runtime. Depth/SVD/pruning need task-specific quality recovery and evaluation. Dense zero weights
do not imply sparse speedups. No metric is inferred from parameter count. Remote source hashing
covers the source identifier rather than Hub cache content. CRF wrappers saved outside normal
HF `PreTrainedModel` conventions are not Phase 1 transformation targets.

## Next steps

Phase 2 should add calibrated INT8 and QAT datasets, backend-tested INT4, corpus-driven safe
SentencePiece reconstruction where ID compatibility can be explicitly relaxed, trained
structured/student methods, chained recipe execution, benchmark matrices, CodeCarbon energy
collection, repeated latency trials, confidence intervals, and downstream NER equivalence and
quality gates.
