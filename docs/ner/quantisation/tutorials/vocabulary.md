# Tutorial 4 — WordPiece vocabulary pruning

[Tutorials](README.md) · [Vocabulary method](../methods/vocabulary.md) · [Analysis](../analysis.md)

First inspect important token IDs with your tokenizer, then choose tokens that must remain. Because
only the tail above the highest required ID can be removed, a high-ID kept token may prevent any
pruning.

```bash
SOURCE=models/bert-biomedical-ner
OUTPUT=artifacts/a6-vocab-tail
lab quantisation prune-vocabulary "$SOURCE" --output "$OUTPUT" \
  --keep-token disease --keep-token syndrome --task token-classification
lab quantisation validate "$OUTPUT" --source "$SOURCE"
```

Inspect manifest `details` for original/new size and removed embeddings. Benchmark token counts and
fragmentation. **Do not use this command on XLM-R/SentencePiece**: the implementation intentionally
raises. XLM-R mutation is not currently supported; train/reconcile a tokenizer in a separate,
scientifically explicit experiment instead.
