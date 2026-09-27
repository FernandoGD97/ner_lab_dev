# Weight and vocabulary pruning

Magnitude pruning zeros the smallest matrix weights. It changes neither tensor shapes nor dense
parameter count and, without sparse serialization/kernels, need not reduce checkpoint bytes,
FLOPs, memory, latency, or energy. Fine-tuning is recommended. Structured pruning is metadata
only because a scientifically useful implementation needs training and backend-aware shapes.

```bash
lab quantisation prune MODEL --amount 0.2 --output artifacts/pruned
```

Vocabulary analysis reports token/document and subword/word rates. Mutation is deliberately
limited to a contiguous **tail** of WordPiece vocabularies, preserving every retained token ID,
every special token, embedding consistency, and `config.vocab_size`. Arbitrary deletion would
renumber tokens. XLM-R/SentencePiece is analysis-only in Phase 1 rather than silently corrupting
the tokenizer. Pruning an existing vocabulary is not training a new tokenizer. The manifest
records old/new vocabulary, removed embedding parameters and observed serialized-byte delta.

```bash
lab quantisation prune-vocabulary MODEL --keep-token TOKEN --output artifacts/vocab
```

Supported mutation: BERT WordPiece. Limitation: corpus tokenization quality must be measured
before and after by the researcher. References include [Deep Compression](https://arxiv.org/abs/1510.00149),
[Movement Pruning](https://arxiv.org/abs/2005.07683), and multilingual evidence in the bibliography.
