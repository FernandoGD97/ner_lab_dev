# Architectural reduction

[Methods](README.md) · [Distillation](distillation.md)

`reduce-depth` recognizes conventional `base_model.encoder.layer` or
`base_model.transformer.layer`, retains the first N blocks, and updates the depth config. This
physically removes parameters and dense layer work, but proportional storage/memory/latency/energy
changes are not assumed. The transformation is `PAPER_INSPIRED` and needs fine-tuning/distillation.

Context: [LayerDrop](https://arxiv.org/abs/1909.11556) — **PREPRINT VERSION LINKED**. This is prefix
truncation, not a reproduction of LayerDrop training.

Width and FFN modules intentionally raise `NotImplementedError`; unsafe matrix slicing is not
provided. Hidden-size/FFN reductions require a trained student. No attention-head reduction command
exists.
