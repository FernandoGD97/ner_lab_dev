# Quantization

## Dynamic INT8 (`int8` / registry `int8-dynamic`)

PyTorch dynamically quantises `Linear` weights and activations on CPU. No calibration is used. The specialized artifact contains `quantized_model.pt` and is loaded by canonical NER inference. It is CPU-only and is not a normal `AutoModel` checkpoint.

## Calibrated static INT8 (`int8-static`)

This deterministic reference W8A8 backend samples local calibration text using `--calibration-seed`, observes each `Linear` input range, stores symmetric INT8 weights and calibrated activation ranges, and reconstructs through the canonical inference loader. It accepts local corpus Parquet (`text` column), JSON/JSONL, or line-delimited text. It dequantises for floating Linear compute, so it validates storage/numerics but **does not claim optimized INT8 latency**.

```bash
lab quantisation int8-static MODEL --output artifacts/a4 \
  --calibration-corpus train.parquet --calibration-samples 128 --calibration-seed 42
```

## Weight-only INT4 (`int4`)

Weights of selected `Linear` modules are affine-quantised per contiguous group and two nibbles are packed into each byte. Scales and optional asymmetric zero points are stored. The classifier is excluded by default. The reference CPU loader dequantises weights for each forward operation; checkpoint reduction is real, optimized-kernel acceleration is not claimed.

```bash
lab quantisation int4 MODEL --output artifacts/a5 --group-size 64
# add --asymmetric to store per-group zero points
```

Static INT8 and INT4 reject missing calibration input/invalid group sizes and never fall back to FP32. QAT remains unimplemented. These methods are generic equivalents, not Q-BERT, I-BERT, GPTQ, or AWQ implementations.

See the [matrix](../matrix.md), [artifacts](../artifacts.md), and [benchmark protocol](../benchmarking.md).
