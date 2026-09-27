# Architectural reduction

[Methods](README.md) · [Distillation](distillation.md) · [Matrix](../matrix.md)

`reduce-depth` retains a prefix of a recognized encoder stack. `student` instead constructs a valid configurable token-classification architecture and uses deterministic evenly spaced teacher-layer mapping for shape-compatible initialization. Hidden/FFN/head settings must form a valid configuration; an initialization report records copied and random tensors.

```bash
lab quantisation student TEACHER --output student-init --layers 6 \
  --hidden-size 384 --intermediate-size 1536 --attention-heads 6 --seed 42
```

The student artifact is executable but **untrained**. It is not B8 and should not be benchmarked as a scientific distilled result until trained. Context: [LayerDrop](https://arxiv.org/abs/1909.11556) and [DistilBERT](https://arxiv.org/abs/1910.01108); these generic implementations do not reproduce either paper.
