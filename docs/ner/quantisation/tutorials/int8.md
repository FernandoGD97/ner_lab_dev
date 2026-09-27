# Tutorial 3 — dynamic INT8 PTQ

[Tutorials](README.md) · [Quantization](../methods/quantization.md) · [Validation](../validation.md)

```bash
SOURCE=models/biomedical-ner
OUTPUT=artifacts/a3-int8-dynamic
lab quantisation int8 "$SOURCE" --output "$OUTPUT" --task token-classification
lab quantisation validate "$OUTPUT" --source "$SOURCE"
```

Expect `quantized_model.pt`, not `model.safetensors`. Configure a controlled A0/A3 study on `cpu`;
dynamic quantized Linear modules are CPU-only. This command performs no calibration and is not
static INT8/QAT/Q-BERT/I-BERT. Compare packed bytes, RSS, model time, end-to-end time, energy, and
quality independently.
