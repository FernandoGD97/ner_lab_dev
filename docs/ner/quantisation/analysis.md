# Phase 3 NER analysis

[Home](README.md) · [CLI](cli.md) · [Experiments](experiments.md) · [Tutorial](tutorials/experiments.md)

All analyses consume `benchmark_long.tsv` and stored `gold.tsv`/`predictions.tsv`; they do not
rerun inference. `lab quantisation experiment analyse RESULTS` writes an `analysis/` directory.
Use `--training-frequency-source TRAIN.tsv` only with training annotations. Test predictions are
never used to estimate training frequency.

The resolved Phase 2 YAML may include `statistics.bootstrap_iterations`, `confidence_level`, and
`seed`, plus `analysis.training_frequency_source`, `seen_definition`, and `plots`. A supplied CLI
training-frequency path takes precedence. The current implementation rereads resolved `plots`
after receiving the CLI flag, so set `analysis.plots: true` in YAML rather than relying on
`--plots` when a resolved experiment exists.

## Robustness and long-tail definitions

A *seen mention* is a case-folded, whitespace-normalized gold mention string that occurs in the
explicitly supplied training annotation source. *Unseen* means count zero under that definition.
The source path and definition are persisted in `frequency_definition.json`. Default buckets are
unseen, singleton, 2–5, 6–20, and >20. These are mention-string frequencies, not concept
frequencies and not test-set prevalence.

Gold mentions are independently described by entity type, character length, whitespace-token
length, and tokenizer subwords when the saved tokenizer is locally available. Fragmentation
buckets are one, two, three-to-four, and five-or-more subwords. Character buckets are short
(≤5), medium (6–15), and long (>15). Every robustness row retains TP/FP/FN, strict precision,
recall, F1, and gold support. False positives are attributed using their own mention properties;
the primary global metrics remain unchanged.

`robustness.tsv` includes baseline-relative `delta_f1 = compressed_f1 - baseline_f1` for every
bucket. The paper robustness export combines count statistics rather than averaging bucket F1.
Tokenizer-unavailable rows are explicitly `unavailable`.

## Error taxonomy and prediction differences

The additional taxonomy uses the repository's half-open offsets and distinguishes correct,
false-positive, false-negative, wrong-label, left-boundary, right-boundary, both-boundary, and
other overlapping predictions. It does not replace canonical strict scoring.

```bash
lab quantisation experiment compare-predictions RESULTS --baseline A0 --model A3 \
  --training-frequency-source train.tsv
```

The detailed TSV preserves spans and baseline/compressed error categories, then labels recovered
by both, baseline only, compressed only, label change, boundary change, or shared error. Summary
JSON counts shared/new/removed false positives and exact recoveries. Subword count remains NA if
no locally loadable tokenizer can support it.

## Pareto, uncertainty, and correlations

Pareto analysis maximizes F1 and minimizes one measured resource. A point is non-dominated only
when no other point is no worse on both axes and strictly better on at least one. Separate files
cover checkpoint bytes, parameters, peak allocated VRAM, forward time, end-to-end time, and energy.
There is no combined score and no global rank.

Paired bootstrap samples documents with replacement and reports baseline F1, compressed F1,
delta, and percentile confidence interval. Pairing preserves each sampled document's gold and both
prediction sets. Iteration count, confidence level, seed, and document unit are stored. Repeated
resource summaries from Phase 2 remain descriptive; five runs do not justify strong certainty.

Pearson and rank-based Spearman correlations are exported for available resource pairs. Spearman
is computed from average ranks without requiring SciPy. Correlations are marked
`descriptive_not_causal`; compression family, hardware, sequence length, and runtime confound them.
Missing FLOP estimates are not invented.

## Paper tables and plots

The analysis exports TSV and CSV model-characteristic, main-result, and robustness tables.
Compression-family summaries preserve precision, quantization, vocabulary/structural pruning,
distillation, architecture, low rank, combined, and runtime categories without ranking them.
Optional `--plots` writes the exact TSV behind every figure before rendering a neutral Matplotlib
scatter plot. Plotting is separate from execution and gracefully remains data-only without
Matplotlib.
