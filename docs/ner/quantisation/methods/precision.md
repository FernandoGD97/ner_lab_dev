# Precision: FP32, FP16, BF16

[Methods](README.md) · [FP16 tutorial](../tutorials/fp16.md) · [Measurements](../measurements.md)

| Variant | Physical change | Parameters/FLOPs | Likely storage/memory | Training | Main limitation |
|---|---|---|---|---|---|
| FP32 | None; registry baseline metadata only | unchanged | baseline | none | no transformation command |
| FP16 | Every model parameter cast with `model.half()` | unchanged | weight bytes often ~half | none | CPU kernels may fail/be slower |
| BF16 | Every model parameter cast to `torch.bfloat16` | unchanged | weight bytes often ~half | none | efficient hardware support required |

The output remains a normal Hugging Face artifact and supports base, token-classification, and
sequence-classification AutoClasses when the source does. Neither conversion proves lower latency,
VRAM, or energy; activation/runtime behavior must be measured.

Primary context links stored in metadata:

* FP16: [Mixed Precision Training](https://arxiv.org/abs/1710.03740) — **PREPRINT / NON-PEER-REVIEWED VERSION LINKED**.
* BF16: [A Study of BFLOAT16 for Deep Learning Training](https://arxiv.org/abs/1905.12322) — **PREPRINT / NON-PEER-REVIEWED VERSION LINKED**.
* FP32 baseline context: [BERT](https://arxiv.org/abs/1810.04805) — **PREPRINT VERSION LINKED**.
