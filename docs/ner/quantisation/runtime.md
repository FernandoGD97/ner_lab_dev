# Low-rank and runtime optimization

Experimental SVD replaces beneficial linear matrices with two low-rank factors. This changes
parameter count but may increase latency without fused kernels, needs fine-tuning, and produces a
specialized PyTorch artifact loadable with `load_low_rank`, not `AutoModel`.

ONNX export changes execution representation, not learned parameters. It requires the optional
`onnx` package; performance depends on the ONNX Runtime execution provider and optimization.
The artifact contains `model.onnx`, configuration, tokenizer and a manifest whose backend is
`onnx`; it is not a normal Hugging Face model directory.

```bash
lab quantisation export-onnx MODEL --output artifacts/onnx
```

Neither method makes claims about latency, memory, FLOPs, or energy without measurement.
