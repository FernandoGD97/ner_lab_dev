# Runtime optimization: ONNX

[Methods](README.md) · [Artifacts](../artifacts.md) · [Troubleshooting](../troubleshooting.md)

`export-onnx` exports a CPU example trace with dynamic batch/sequence axes, input names from the
tokenizer, output name `output`, and fixed opset 17. It changes representation, not learned
parameters. Performance depends on a separate ONNX Runtime/provider configuration.

The module does not implement ONNX Runtime benchmarking, ONNX quantization, OpenVINO, TensorRT, or
provider selection. The controlled PyTorch experiment validator marks backend `onnx` unsupported.
A smaller/faster engine must not be attributed solely to model compression.

Reference: [ONNX project](https://onnx.ai/) (project specification, not a preprint).
