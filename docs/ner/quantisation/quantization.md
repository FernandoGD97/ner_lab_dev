# INT8 and INT4

Dynamic INT8 PTQ replaces CPU `Linear` operations with PyTorch dynamically quantized modules.
It requires no calibration or training and can reduce linear-weight storage/memory; embeddings
remain floating point. Parameter counting is not interchangeable with byte count, and speed is
CPU/backend/workload dependent. Output uses `quantized_model.pt` and the documented
`load_dynamic_int8`, not `AutoModel`.

```bash
lab quantisation int8 MODEL --output artifacts/int8
```

Static calibrated INT8 and QAT are scaffolds for later work. INT4 is intentionally unsupported
in Phase 1: common backends are hardware-specific, and GPTQ, SmoothQuant and AWQ evidence is
primarily for decoder-only LLMs rather than BERT/RoBERTa/XLM-R encoders. This implementation is
generic PyTorch PTQ, not Q-BERT, I-BERT, TernaryBERT, BinaryBERT, or ZeroQuant.

References: [integer quantization](https://arxiv.org/abs/1712.05877),
[GPTQ](https://arxiv.org/abs/2210.17323), [SmoothQuant](https://arxiv.org/abs/2211.10438), and
[AWQ](https://arxiv.org/abs/2306.00978).
