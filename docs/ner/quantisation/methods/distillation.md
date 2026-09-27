# Distillation

[Methods](README.md) · [Architecture reduction](architecture-reduction.md) · [References](../references.md)

The Python API implements actual temperature-scaled token KL with `T²`, padding masks, hard-label cross entropy, hidden-state MSE with optional learned projections, and attention-map MSE. `DistillationTrainer` freezes/evaluates the teacher under `no_grad`, detaches teacher targets, and updates only the student from canonical pre-encoded batch iterables. Layer mapping is explicit and deterministic.

There is **no corpus-to-artifact distillation CLI or recovery-training orchestrator yet**. Therefore B8 and distillation-dependent C cells remain unsupported study artifacts even though the objective and batch training APIs execute. No command silently labels an initialized student as distilled.

The bibliography links Knowledge Distillation, DistilBERT, TinyBERT, MobileBERT, and MiniLM as scientific context, not implementation claims. ArXiv links are **PREPRINT / NON-PEER-REVIEWED VERSIONS LINKED** unless stated otherwise.
