# Quantization

[Methods](README.md) · [INT8 tutorial](../tutorials/int8.md) · [Troubleshooting](../troubleshooting.md)

## Implemented: dynamic INT8 PTQ

PyTorch `quantize_dynamic` replaces eligible CPU `nn.Linear` operations with dynamically quantized
INT8 modules. Linear weights are packed; activations are quantized dynamically around operations.
Embeddings and unsupported modules remain floating-point. No calibration dataset or training is
used. It is computational compression and may reduce packed weight memory/storage, but parameter
count, FLOPs, latency, and energy are not inferred.

The artifact is specialized (`quantized_model.pt`) and CPU-only. Use the canonical manifest-aware
NER loader or `load_dynamic_int8`; plain `AutoModel.from_pretrained` is not supported.

Scientific context: [Quantization and Training of Neural Networks](https://arxiv.org/abs/1712.05877)
— **PREPRINT VERSION LINKED**. Fidelity is `GENERIC_EQUIVALENT`, not a named encoder quantization
paper reproduction.

## Not implemented

* Static/calibrated INT8 and QAT have no method class or CLI command.
* INT4 has metadata/scaffold only (`int4.py`) and no registry/command.
* W8A8, weight-only settings, per-channel/per-tensor controls, GPTQ, SmoothQuant, and AWQ are not
  selectable. The bibliography notes decoder-only evidence for the latter methods.

Do not use A4/A5 recipe labels as evidence these transformations exist.
