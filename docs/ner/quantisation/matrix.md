# Definitive executable matrix

This table describes code that exists in this revision, not aspirational recipe names. “Reference” means a portable PyTorch representation whose storage is genuinely quantised but whose Linear operation dequantises to floating point; it must not be used to claim optimized-kernel speed-up. CRF artifacts are unsupported by the specialized quantised loaders.

| ID | Method | Executable artifact | Calibration | Recovery training | Distillation | CPU | CUDA | Artifact/runtime | Linear NER | CRF |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| A0 | FP32 baseline | yes (source) | no | no | no | yes | yes | Transformers | yes | yes |
| A1 | FP16 | yes | no | no | no | limited | yes | Transformers | yes | no verified support |
| A2 | BF16 | yes | no | no | no | hardware-dependent | hardware-dependent | Transformers | yes | no verified support |
| A3 | dynamic INT8 PTQ | yes | no | no | no | yes | no | `pytorch_dynamic_int8` | yes | no |
| A4 | calibrated W8A8 | yes | **yes** | no | no | yes | no | `pytorch_reference_static_int8` | yes | no |
| A5 | weight-only INT4 | yes | no | no | no | yes | no | `pytorch_int4_reference` | yes | no |
| A6 | tail-pruned WordPiece vocabulary | yes | analysis corpus | recommended | no | yes | yes | Transformers | yes | no verified support |
| A7 | vocabulary then dynamic INT8 | executable as two explicit commands | analysis corpus | recommended | no | yes | no | chained artifacts | yes | no |
| B1–B3 | reduced depth | yes, raw initialization | no | **required for a scientific model** | recommended | yes | yes | Transformers | yes | no |
| B4–B6 | configurable student width/FFN | yes, raw initialization | no | **required** | recommended | yes | yes | Transformers | yes | no |
| B7 | head/FFN structural pruning | yes, BERT/RoBERTa constraints | no | recommended | no | yes | yes | Transformers | yes | no |
| B8 | fully trained distilled artifact | **not exposed as an end-to-end command** | no | yes | yes | — | — | trainer API only | pending | no |
| C1–C6 | trained combinations | **not yet orchestrated** | varies | yes | varies | — | — | recipes remain non-executable | pending | no |

The `student` command produces a structurally valid initialized checkpoint, not a trained or distilled result. The distillation loss and training-loop APIs are executable, but corpus-to-training orchestration and artifact publication are not yet a CLI operation; therefore B8 and dependent C cells remain unsupported as study results.

See [quantization](methods/quantization.md), [pruning](methods/pruning.md), and [architecture reduction](methods/architecture-reduction.md).
