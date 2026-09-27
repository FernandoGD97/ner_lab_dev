# Depth, width, FFN, and distillation

Depth reduction keeps a prefix of a conventional encoder layer list and updates configuration.
It reduces parameters and dense layer FLOPs but does not guarantee proportional bytes, memory,
latency, or energy. Fine-tuning/distillation is required for credible quality, so generated
artifacts are experimental.

```bash
lab quantisation reduce-depth MODEL --layers 6 --output artifacts/6l
```

Hidden-size and FFN reduction never slice a trained model. They require a newly configured and
trained student. `StudentSpec` supports arbitrary layer/hidden/FFN/head settings, including
9L/768H, 6L/768H, 6L/512H, 6L/384H, and 4L/384H. Loss modules provide supervised-hard-label
integration points, logits/soft-target KD, hidden-state and attention matching; actual expensive
training is deferred.

References: [LayerDrop](https://arxiv.org/abs/1909.11556), [knowledge distillation](https://arxiv.org/abs/1503.02531),
[DistilBERT](https://arxiv.org/abs/1910.01108), [TinyBERT](https://arxiv.org/abs/1909.10351),
[MobileBERT](https://arxiv.org/abs/2004.02984), and [MiniLM](https://arxiv.org/abs/2002.10957).
The generic facilities are not claimed as those named algorithms.
