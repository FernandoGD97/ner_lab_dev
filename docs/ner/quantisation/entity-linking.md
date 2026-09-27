# Entity Linking compression benchmarks

[Home](README.md) · [CLI](cli.md) · [Configuration](configuration.md) · [Measurements](measurements.md)

The repository already provides mention records, gazetteers, lexical/matrix/FAISS/dense candidate
generators, reciprocal-rank fusion, a SentenceTransformers cross-encoder, and Recall@K/MRR
metrics. Phase 3 instruments those components; it does not implement a second linker.

## Configuration and commands

```yaml
experiment: {id: el_compression_v1, seed: 42}
task:
  type: entity_linking
  mentions: test_mentions.tsv
  gazetteer: terminology.tsv
  training_mentions: train_mentions.tsv
retrieval:
  candidate_k: [200, 100, 50, 25, 10]
  method: matrix
  index_type: flat_ip
  distance_metric: cosine
biencoder: {model: null, embedding_dimension: null}
crossencoder: {enabled: false, batch_size: 16}
energy: {enabled: false, min_duration_seconds: 60}
output: {directory: results}
repetitions: 5
```

```bash
lab quantisation el validate el.yaml
lab quantisation el run el.yaml
lab quantisation el summarize results/el_compression_v1
```

Training concepts define seen/unseen: seen iff the gold concept ID occurs in the supplied training
annotations. The exact definition is stored. Frequency buckets use those training concept counts,
never test counts.

## Decomposition

Offline fields cover entity/terminology encoding, index construction, serialization, and index
bytes. Online fields cover mention/context preparation, mention encoding, search, candidate
preparation, cross-encoder reranking, postprocessing, and total time. Candidate K and embedding
output dimension are independent axes; hidden size is never treated as embedding dimension.
Candidate pairs and pairs/s expose reranker work.

Some existing retrievers expose mention encoding and search only as one call. Such runs retain the
measured combined retrieval interval and a `timing_scope` explanation; unavailable subcomponents
are NA rather than fabricated. Likewise, the present FAISS class exposes FlatIP/FlatL2 only.
Scalar quantization, PQ, IVF, IVF-PQ, and HNSW validate as `UNSUPPORTED` pending real repository
adapters. Index compression is always labeled as an index axis, never Transformer compression.

## Quality, cost, and energy

Outputs include Accuracy@1, Recall@1/5/25/50/100/200, MRR, maximum-K candidate recall, and
training-derived seen/unseen accuracy. Cross-encoder results use the existing reranker. Offline
index building is excluded from primary online deployment time.

Optional CodeCarbon tracking repeats the online pipeline for the configured minimum duration and
records end-to-end EL energy separately from offline construction. Normalizations include J/1000
mentions, candidate pairs, and retrieval queries. The current combined retriever interface cannot
scientifically separate bi-encoder from index energy; those component fields remain pending until
the retriever exposes encoding and search independently.

Current configuration limitations: EL `experiment.seed`, `analysis` toggles, `energy.modes`, and
`crossencoder.batch_size` are parsed but are not used to change execution. When energy is enabled,
the implementation measures one `el_end_to_end` online region. Candidate-K order follows the YAML
list; no randomized EL execution order or resume/force mechanism exists.

Compressed bi-encoders and rerankers may point at Phase 1 artifacts only when their existing loader
supports them. There is no unsafe projection for 768→384 embeddings. A smaller FAISS index is not
a smaller Transformer, and reducing K is not model compression.
