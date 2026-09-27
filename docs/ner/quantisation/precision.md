# FP16 and BF16

Both commands change stored weights and expected arithmetic dtype but do **not** reduce the
number of parameters, layers, vocabulary, or theoretical dense FLOPs. They require no
training. FP16 usually halves weight bytes relative to FP32 but CPU execution may be unsupported
or slower. BF16 has FP32-like exponent range but requires capable hardware for speed. Neither
implies lower latency or energy.

Supported: BERT, RoBERTa, XLM-R, DistilBERT and compatible `AutoModel` base, token-classification,
and sequence-classification checkpoints.

```bash
lab quantisation fp16 MODEL --output artifacts/fp16
lab quantisation bf16 MODEL --output artifacts/bf16
```

Implementation: Transformers `save_pretrained` plus PyTorch dtype conversion. Scientific
context: [mixed precision](https://arxiv.org/abs/1710.03740) and
[BFLOAT16](https://arxiv.org/abs/1905.12322). Limitations include hardware-specific kernels and
small numerical changes.
