# Tutorials 5–7 — controlled A0/A1/A3 study and analysis

[Tutorials](README.md) · [Configuration](../configuration.md) · [Results](../experiments.md)

## 5. Configure and run

Create `study.yaml` (paths are placeholders):

```yaml
experiment: {id: ner_a0_a1_a3, seed: 42}
dataset: {path: data/test.parquet, split: test}
inference:
  batch_size: 16
  max_length: 256
  device: cpu
  warmup_batches: 10
  repetitions: 5
  pad_to_multiple_of: 8
  number_of_workers: 0
models:
  - {id: A0, path: models/biomedical-ner, runtime: pytorch}
  - {id: A1, path: artifacts/a1-fp16, runtime: pytorch}
  - {id: A3, path: artifacts/a3-int8-dynamic, runtime: pytorch, backend: pytorch_dynamic_int8}
energy: {enabled: false}
statistics: {bootstrap_iterations: 10000, confidence_level: 0.95, seed: 42}
analysis:
  training_frequency_source: data/train-spans.tsv
  seen_definition: normalized_mention_string
  plots: false
output: {directory: results}
protocol: controlled
```

CPU is required here because A3 is dynamic INT8. FP16 CPU validation/execution may be unsupported on
your hardware; validation must report that failure rather than silently changing the protocol.

```bash
lab quantisation experiment validate study.yaml
lab quantisation experiment run study.yaml --dry-run
lab quantisation experiment run study.yaml
# After an interruption/failure:
lab quantisation experiment status results/ner_a0_a1_a3
lab quantisation experiment run study.yaml --resume
```

## 6. Summarize and compare

```bash
RESULTS=results/ner_a0_a1_a3
lab quantisation experiment summarize "$RESULTS"
lab quantisation experiment compare "$RESULTS" --baseline A0
```

Use `benchmark_long.tsv` for individual repetitions, `benchmark_summary.tsv` for descriptive
statistics, and `comparison.tsv` for independent baseline-relative reductions/speedups/deltas.
Never treat a storage reduction as a measured latency or energy reduction.

## 7. Scientific analysis

```bash
lab quantisation experiment compare-predictions "$RESULTS" \
  --baseline A0 --model A3 --training-frequency-source data/train-spans.tsv
lab quantisation experiment pareto "$RESULTS"
lab quantisation experiment correlations "$RESULTS"
lab quantisation experiment analyse "$RESULTS" \
  --training-frequency-source data/train-spans.tsv --baseline A0
```

The training source must not be the test set. `analyse` reads saved predictions/results, produces
frequency/fragmentation robustness, paired bootstrap, family, Pareto, correlation, and paper tables,
and does not rerun inference. Set `analysis.plots: true` in the original YAML for reliable optional
plot rendering (see the precedence caveat in the [CLI reference](../cli.md#experiment-analyse)).
