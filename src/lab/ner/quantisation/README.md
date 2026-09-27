# `lab.ner.quantisation`

`lab.ner.quantisation` is the Transformer compression, controlled benchmarking, and post-hoc
analysis subsystem inside `ner_lab`. It operates on already-trained Hugging Face checkpoints and,
for NER experiments, deliberately reuses `lab.ner.inference` rather than maintaining a second
prediction pipeline.

```text
trained checkpoint
        ↓
compression transformation
        ↓
new artifact (the source is never overwritten)
        ↓
reload / smoke validation
        ↓
existing ner_lab NER inference
        ↓
quality, latency, memory, throughput, energy
        ↓
robustness, Pareto, bootstrap, correlation, and paper exports
```

A central scientific rule is that these are different observations:

```text
parameter reduction != checkpoint reduction != memory reduction
                    != latency reduction != energy reduction
```

No result should be inferred from another. For example, unstructured zeros do not make a dense
kernel sparse, and a smaller vocabulary can increase sequence fragmentation.

## Start here

The installed executable is **`lab`**, declared in `pyproject.toml`. The production entry point is
the repository's central `argparse` CLI; it mounts this module as `lab quantisation` and also
accepts the US-spelling alias `lab quantization`. This module additionally defines a Typer `app`
for embedding/testing, but no separate Typer executable is installed.

```bash
uv pip install -e ".[ner]"             # compression + NER experiments
uv pip install -e ".[all]"             # also install NEL/FAISS dependencies
lab quantisation --help
lab quantisation methods
```

## Documentation map

| Need | Page |
|---|---|
| Every command and option | [CLI reference](../../../../docs/ner/quantisation/cli.md) |
| System design and execution flows | [Architecture](../../../../docs/ner/quantisation/system-architecture.md) |
| YAML and Pydantic fields | [Configuration](../../../../docs/ner/quantisation/configuration.md) |
| Artifact files and manifests | [Artifacts](../../../../docs/ner/quantisation/artifacts.md) |
| Reload and smoke checks | [Validation](../../../../docs/ner/quantisation/validation.md) |
| Controlled runs, outputs, resume | [Experiments and results](../../../../docs/ner/quantisation/experiments.md) |
| Timing, memory, and energy | [Measurement methodology](../../../../docs/ner/quantisation/measurements.md) |
| Robustness and scientific analysis | [Analysis](../../../../docs/ner/quantisation/analysis.md) |
| Entity Linking | [Entity Linking](../../../../docs/ner/quantisation/entity-linking.md) |
| Implemented and scaffolded methods | [Methods index](../../../../docs/ner/quantisation/methods/README.md) |
| Complete walkthroughs | [Tutorials](../../../../docs/ner/quantisation/tutorials/README.md) |
| Add a method, command, backend, or test | [Extension guide](../../../../docs/ner/quantisation/extending.md) |
| Errors and remedies | [Troubleshooting](../../../../docs/ner/quantisation/troubleshooting.md) |
| Terms | [Glossary](../../../../docs/ner/quantisation/glossary.md) |
| Commands only | [CLI cheat sheet](../../../../docs/ner/quantisation/CLI_CHEATSHEET.md) |

## Support levels

* **Implemented, normal HF artifact:** FP16, BF16, magnitude pruning, tail-only WordPiece
  vocabulary pruning, and experimental depth reduction.
* **Implemented, specialized runtime artifact:** dynamic CPU INT8 and experimental SVD.
* **Implemented runtime export:** ONNX export; the normal validation command intentionally refuses
  to pretend ONNX is a Hugging Face/PyTorch artifact.
* **Executable reference artifacts:** calibrated static INT8, packed weight-only INT4, structured head/FFN pruning, and initialized student construction.
* **API but no end-to-end artifact command:** distillation losses/training loop. Recovery fine-tuning and combination orchestration remain unsupported. They have no compression command and must not be reported as completed
  algorithms.
* **Baseline metadata only:** FP32 (`methods`/`method fp32`); there is no `fp32` transformation.

See the [limitations table](../../../../docs/ner/quantisation/methods/README.md) before designing a
study.
