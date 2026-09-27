# Distillation

[Methods](README.md) · [Architecture reduction](architecture-reduction.md) · [References](../references.md)

Phase 1 includes `StudentSpec`, configurable hard/logits/hidden/attention/intermediate weights, and
loss functions for logits KL and MSE matching. `DistillationTrainer.train()` deliberately raises
`NotImplementedError`. There is no distillation CLI command and no expensive training is launched.

The bibliography links Knowledge Distillation, DistilBERT, TinyBERT, MobileBERT, and MiniLM. These
entries are scientific context—not claims that the named algorithms are implemented. ArXiv links
are **PREPRINT / NON-PEER-REVIEWED VERSIONS LINKED** unless the bibliography points elsewhere.
