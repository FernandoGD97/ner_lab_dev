# Quantisation CLI cheat sheet

[Full CLI reference](cli.md) · [Tutorials](tutorials/README.md) · [Troubleshooting](troubleshooting.md)

```bash
# Discover
lab quantisation --help
lab quantisation methods
lab quantisation method fp16

# Inspect
lab quantisation inspect MODEL --task token-classification

# Transform
lab quantisation fp16 MODEL --output OUT --task token-classification
lab quantisation bf16 MODEL --output OUT --task token-classification
lab quantisation int8 MODEL --output OUT --task token-classification
lab quantisation prune MODEL --output OUT --amount 0.2
lab quantisation prune-vocabulary MODEL --output OUT --keep-token TOKEN
lab quantisation reduce-depth MODEL --output OUT --layers 6
lab quantisation svd MODEL --output OUT --rank-ratio 0.5
lab quantisation export-onnx MODEL --output OUT
lab quantisation run-recipe recipe.yaml [--output OUT]

# Validate
lab quantisation validate ARTIFACT --source SOURCE

# Controlled experiment
lab quantisation experiment validate study.yaml
lab quantisation experiment run study.yaml --dry-run
lab quantisation experiment run study.yaml
lab quantisation experiment run study.yaml --resume
lab quantisation experiment run study.yaml --force
lab quantisation experiment status RESULTS
lab quantisation experiment summarize RESULTS
lab quantisation experiment compare RESULTS --baseline A0

# Post-hoc analysis (no inference rerun)
lab quantisation experiment analyse RESULTS --training-frequency-source TRAIN.tsv
lab quantisation experiment compare-predictions RESULTS --baseline A0 --model A3
lab quantisation experiment pareto RESULTS
lab quantisation experiment correlations RESULTS
lab quantisation experiment joint-report NER_RESULTS --el-results EL_RESULTS

# Entity Linking
lab quantisation el validate el.yaml
lab quantisation el run el.yaml
lab quantisation el summarize EL_RESULTS
```

`quantization` aliases only the top-level group. All examples use `quantisation`.

## Hardened executable primitives

```bash
lab quantisation int8-static MODEL --output OUT --calibration-corpus TRAIN.parquet
lab quantisation int4 MODEL --output OUT --group-size 64
lab quantisation prune-structured MODEL --output OUT --head-amount 0.25 --ffn-amount 0.25
lab quantisation student MODEL --output OUT --layers 6 --hidden-size 384 --attention-heads 6
```

The `student` output is initialized, not trained. See the [definitive matrix](matrix.md).
