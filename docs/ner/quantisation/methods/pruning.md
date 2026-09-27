# Weight and structured pruning

[Methods](README.md) · [CLI prune](../cli.md#prune) · [Architecture reduction](architecture-reduction.md)

Implemented magnitude pruning finds a per-tensor magnitude threshold and writes zeros into matrix
parameters. It does not create sparse tensors or sparse kernels. Dense parameter count/shapes/FLOPs
remain unchanged, checkpoint bytes often remain unchanged, and acceleration is not expected without
a separate sparse runtime. The manifest records requested fraction, zero count, and eligible count.
Fine-tuning is recommended but not performed.

[Deep Compression](https://arxiv.org/abs/1510.00149) is linked as context — **PREPRINT VERSION
LINKED**. The implementation is a generic magnitude operation, not the complete paper pipeline.

`structured.py` contains metadata only. There is no command for structured blocks, attention-head
pruning, CoFi, or Movement Pruning. No method physically removes heads. Depth reduction is a
separate implemented experimental method.
