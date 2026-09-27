# Compression methods: support and scientific interpretation

[Home](../README.md) · [CLI](../cli.md) · [References](../references.md) · [Tutorials](../tutorials/README.md)

The registry exposes nine metadata records; eight are transform implementations. Architecture/task
lists are metadata defaults, not dynamic compatibility guarantees. Always inspect, transform a copy,
validate, and benchmark.

| Registry ID | CLI | Support | Artifact/backend | Training declaration |
|---|---|---|---|---|
| `fp32` | none | baseline metadata only | source | none |
| `fp16` | `fp16` | implemented | normal HF / `transformers` | none |
| `bf16` | `bf16` | implemented | normal HF / `transformers` | none |
| `int8-dynamic` | `int8` | implemented, CPU | specialized / `pytorch_dynamic_int8` | none |
| `magnitude-pruning` | `prune` | implemented | normal HF / `transformers` | fine-tuning recommended/declared |
| `vocabulary-pruning` | `prune-vocabulary` | implemented only for tail WordPiece mutation | normal HF / `transformers` | none |
| `depth-reduction` | `reduce-depth` | experimental | normal HF / `transformers` | fine-tuning |
| `svd` | `svd` | experimental | specialized / `pytorch_low_rank` | fine-tuning |
| `onnx` | `export-onnx` | export implemented | `onnx` runtime artifact | none |

## Fidelity labels

* `EXACT`: the implementation does exactly the named operation, not exact prediction equivalence.
* `APPROXIMATION`: a lossy approximation (SVD).
* `PAPER_INSPIRED`: uses an idea associated with literature but does not reproduce the complete paper.
* `GENERIC_EQUIVALENT`: a generic algorithm, deliberately not branded as a named paper method.
* `EXTERNAL_ADAPTER`: hands representation/execution to another format/backend.

Generic dynamic INT8 is not Q-BERT, I-BERT, TernaryBERT, BinaryBERT, or ZeroQuant. Metadata links
papers for scientific context; it does not claim reproduction.

## Family pages

* [Precision: FP32/FP16/BF16](precision.md)
* [Quantization: dynamic INT8 and unavailable INT4/static/QAT](quantization.md)
* [Magnitude and structured pruning](pruning.md)
* [Vocabulary pruning](vocabulary.md)
* [Depth, width, and FFN reduction](architecture-reduction.md)
* [Distillation scaffolding](distillation.md)
* [Low-rank SVD](low-rank.md)
* [Runtime/ONNX](runtime.md)

## References in code

A method owns immutable `MethodMetadata.publications`, a tuple of `Publication(key, title, url)`.
There is no class named `PaperReference`. `methods`/`method` serialize publications in JSON;
artifact creation copies publication URLs into `scientific_references`. The broader, hand-maintained
bibliography is `src/lab/ner/quantisation/references/bibliography.yaml`; it is loaded by
`load_references()` and is not generated from method metadata. See [references](../references.md).

See the [definitive executable matrix](../matrix.md) for artifact/runtime and linear-NER/CRF support.
