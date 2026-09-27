# Weight and structured pruning

`prune` performs unstructured magnitude pruning: values become zero but dense tensor shapes remain unchanged. Do not infer dense-hardware latency gains from sparsity.

`prune-structured` physically removes attention projections through the Transformers `prune_heads` API and/or slices BERT/RoBERTa FFN channels. L1 magnitude ranking is deterministic. At least one of `--head-amount` or `--ffn-amount` must be non-zero; amounts are fractions in `[0,1)`. Unsupported encoder layouts fail before artifact publication.

```bash
lab quantisation prune-structured MODEL --output artifacts/b7 \
  --head-amount 0.25 --ffn-amount 0.25 --task token-classification
```

The output is a normal Transformers checkpoint and records removed heads, retained FFN indices, and before/after parameter counts. Recovery fine-tuning is scientifically recommended but no recovery-training CLI is currently implemented.

See the [matrix](../matrix.md), [validation](../validation.md), and [scientific references](../references.md).
