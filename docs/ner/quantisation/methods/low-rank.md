# Experimental low-rank SVD

[Methods](README.md) · [CLI](../cli.md#svd) · [Validation](../validation.md)

For each beneficial `nn.Linear`, SVD chooses a rank from `rank_ratio` and replaces the matrix with
two linear factors. This reduces logical parameters only when factor sizes are smaller. It is lossy
(`APPROXIMATION`), requires fine-tuning for credible quality, and can be slower without fused
low-rank kernels.

The specialized output cannot be loaded by plain AutoModel. `low_rank.json` records source, task,
ratio, and replaced layers; `low_rank_state.pt` stores weights. The loader reconstructs the source,
repeats replacement, then loads state.

Context: [Exploiting Linear Structure Within Convolutional Networks](https://arxiv.org/abs/1404.0736)
— **PREPRINT VERSION LINKED**. It is context for factorization, not an exact Transformer paper.
