# INT8 and INT4

Three executable, explicitly distinct backends exist: dynamic CPU INT8 (`int8`), calibrated CPU reference W8A8 (`int8-static`), and packed CPU reference weight-only INT4 (`int4`). Specialized artifacts load through canonical NER inference but are not normal `AutoModel` checkpoints. The reference static/INT4 implementations dequantise for floating Linear compute and therefore make no latency claim.

See [the detailed quantization guide](methods/quantization.md), [definitive matrix](matrix.md), and [benchmark fairness rules](benchmarking.md). QAT, Q-BERT, I-BERT, GPTQ, SmoothQuant, and AWQ are not implemented.
