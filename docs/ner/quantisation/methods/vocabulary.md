# Vocabulary pruning

[Methods](README.md) · [Tutorial](../tutorials/vocabulary.md) · [Analysis](../analysis.md)

The implemented mutation supports only BERT WordPiece tokenizers and removes a contiguous tail.
Special tokens and every `--keep-token` are retained; the new size is one plus their maximum source
ID. Thus all remaining IDs are unchanged. The input embedding is resized, `config.vocab_size`
updated, stale fast-tokenizer JSON removed, and exact prefix `vocab.txt` written.

This can remove embedding parameters and serialized bytes. It does not change encoder depth/width,
and it can increase subword fragmentation and compute. Measure tokens/document and subwords/word.
The manifest records old/new size, embedding parameters removed, observed byte delta, and ID
preservation.

XLM-R/SentencePiece mutation is **not implemented** because arbitrary removal would renumber IDs;
analysis is available through Python `analyse_vocabulary`, not a dedicated CLI command. Training a
new tokenizer is a different experiment.

Context: [Vocabulary Reduction for Neural Machine Translation](https://aclanthology.org/P17-2086/)
and [REMBERT](https://arxiv.org/abs/2010.12821) (**PREPRINT VERSION LINKED** for REMBERT).
