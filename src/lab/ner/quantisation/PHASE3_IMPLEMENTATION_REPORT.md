# Phase 3 implementation report

## Files and CLI

New `analysis/` modules implement stored-result NER robustness, error comparison, Pareto fronts,
document-paired bootstrap, correlations, compression-family summaries, paper tables, joint-task
exports, and optional plots. New `entity_linking/` modules extend Phase 2 configuration concepts,
metrics, schemas, validation, component timing, K ablation, energy, and summaries around the
existing `lab.nel` pipeline. `lab.nel.evaluation` is now lazy so retrieval metrics do not require
the optional ontology/networkx dependency.

Commands added: `quantisation experiment analyse`, `pareto`, `correlations`,
`compare-predictions`, and `joint-report`; and `quantisation el validate`, `run`, and `summarize`.
All analysis commands read saved Phase 2 raw files and do not rerun inference.

## NER robustness

Gold and false-positive mentions are grouped by entity type, character bucket, whitespace-token
length, tokenizer fragmentation, training frequency, and seen/unseen status. Training frequency is
the count of a case-folded, whitespace-normalized mention string in an explicitly supplied training
annotation file. The source and definition are persisted; test data is never a frequency source.
Subword buckets are 1, 2, 3–4, and 5+. Each bucket stores TP/FP/FN, strict precision, recall, F1,
and support. Baseline-relative F1 degradation is exported for every bucket.

The reusable half-open-offset error taxonomy distinguishes correct, false positive/negative,
wrong label, left/right/both-boundary error, and other overlap. Prediction comparison exports exact
recovery/false-positive counts and detailed examples with error transitions, frequency, length,
and subword availability.

## Scientific analysis

Pareto fronts maximize F1 while independently minimizing checkpoint bytes, parameters, peak VRAM,
forward/end-to-end time, or energy; no composite score/rank exists. Predictive uncertainty uses a
seeded paired bootstrap over documents with percentile intervals. Pearson and rank-based Spearman
correlations are descriptive, not causal. Paper TSV/CSV exports cover characteristics, global
results, and robustness. Optional plots always retain their source TSV. Compression-family
summaries do not collapse methods into a leaderboard.

## EL architecture and integration

Inspection found existing mention/gazetteer schemas, lexical and matrix generators, dense and
FAISS bi-encoders, RRF, candidate evaluation, and a SentenceTransformers cross-encoder. Phase 3
instruments these components rather than replacing them. Offline index construction is separated
from online context, retrieval, candidate preparation, reranking, and postprocessing. Accuracy@1,
Recall@1/5/25/50/100/200, MRR, candidate recall, candidate pairs/rate, and training-derived
seen/unseen accuracy are stored. Candidate K is an independent axis.

Existing retriever APIs often combine mention encoding and index search; the measured combined
scope is recorded and inseparable fields remain NA. Current FAISS supports FlatIP/FlatL2;
SQ/PQ/IVF/IVF-PQ/HNSW are explicitly unsupported. Index size and output embedding dimension remain
independent from Transformer size/hidden width. Cross-encoder timing uses the existing reranker.
Optional end-to-end online CodeCarbon energy is normalized per mentions, candidate pairs, and
queries; separate component energy awaits separable retriever calls.

## Tests and example analysis

Synthetic CPU fixtures cover frequency/seen assignment, fragmentation buckets, all error classes,
prediction differences, Pareto membership, paired bootstrap, Pearson/Spearman export, paper tables,
Recall@K/MRR, candidate K configuration primitives, index/offline/online schemas, and energy
normalization. A synthetic stored A0/A3 study executes the full raw-data analysis workflow and
produces robustness, prediction-difference, Pareto, correlation, bootstrap, and paper tables. It is
explicitly synthetic and no scientific outcome is claimed.

## Limitations and recommended experiments

A locally loadable tokenizer is needed for per-mention subwords. Mention-string seen/unseen is not
concept seen/unseen. Bootstrap intervals need enough documents; correlation needs enough distinct
models. No advanced FAISS adapter, scientifically trained embedding projection, or cross-encoder
compression loader was invented. Recommended next experiments use real held-out biomedical data,
training-derived mention/concept counts, enough repetitions, matched PyTorch/ONNX comparisons,
separable dense-retriever instrumentation, Flat versus validated compressed indexes, K ablations,
and the same artifacts across NER and EL. No cross-task combined score is produced.
