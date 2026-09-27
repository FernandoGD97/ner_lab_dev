# Quantisation documentation report

## Files created or substantially revised

The package entry point is `src/lab/ner/quantisation/README.md`. The existing repository docs tree
is the single detailed manual: `docs/ner/quantisation/README.md`, `cli.md`,
`system-architecture.md`, `configuration.md`, `artifacts.md`, `validation.md`, `experiments.md`,
`measurements.md`, `analysis.md`, `entity-linking.md`, method-family pages, tutorials,
`extending.md`, `troubleshooting.md`, `glossary.md`, `references.md`, and `CLI_CHEATSHEET.md`.
Earlier focused method/benchmark pages remain linked from the index rather than duplicated into a
second tree.

## Coverage

All 13 top-level operational/discovery commands, 10 experiment commands, and three EL commands are
documented with installed syntax, arguments/options/defaults, inputs/outputs, side effects,
validation, failures, and examples. The group alias `quantization` and lack of a separate installed
Typer executable are explicit.

Implemented FP16, BF16, dynamic INT8, magnitude/vocabulary pruning, depth reduction, SVD, and ONNX
export are documented separately from FP32 metadata, INT4/structured/student/distillation scaffolds,
and unsupported runtimes. Fidelity, `Publication`, the bibliography, manifest reference flow, and
preprint-link caveats are covered.

Experiment documentation includes every Pydantic field/default/constraint, controlled versus
maximum-throughput behavior, A0–C6 label semantics, subprocess isolation, all result files, a
complete `benchmark_long.tsv` dictionary, formulas, status/resume/force behavior, timing/CUDA,
memory, throughput, CodeCarbon modes, and post-hoc analysis. EL documents existing-component reuse,
combined timing limitations, quality, K, indexes, energy, and unsupported FAISS families.

Tutorials cover inspection, FP16/BF16, INT8, safe WordPiece vocabulary pruning (and explicit XLM-R
non-support), A0/A1/A3 controlled execution, summaries/comparison, resume, and Phase 3 analysis.
Developer guidance covers a method, Typer + central parser command, backend, config, and verification.

## Implementation/documentation inconsistencies discovered

* The installed CLI is central `argparse`; a parallel Typer `app` exists for embedding/testing. Both
  route to shared handlers. Documentation avoids calling Typer a separately installed executable.
* `CompressionMethod.__call__` invokes compatibility checks, but CLI transformation handlers call
  `apply()` directly. Metadata support lists are therefore not automatic checkpoint checks.
* Compression recipe `method.mode` is parsed but deliberately excluded before `apply()`.
* Generated `compression_recipe.yaml` omits method-specific transformation options; manifest
  `details` must also be retained for reproduction.
* Specialized manifest `checkpoint_bytes` is based on source inspection; Phase 2 independently
  remeasures artifact-directory bytes.
* The validation report's task/label booleans are strongest only with `--source`; specialized paths
  currently report them true after successful loading.
* The ONNX export has no ONNX Runtime validation/benchmark adapter.
* Resolved `analysis.plots` can override the CLI `--plots` value; documentation recommends YAML.
* `PENDING` is a defined run status but is not currently persisted by the parent runner.
* Resume recognizes exact `SUCCESS` only and does not verify payload/config hashes.
* `maximum_throughput` permits per-model batches but performs no automatic tuning.
* EL retrievers can combine mention encoding and index search; inseparable measurements remain NA.
* EL seed/analysis toggles/energy modes/cross-encoder batch size are parsed but do not currently
  alter execution; EL also has no resume/force orchestration.

These inconsistencies were documented rather than silently changing behavior, in accordance with
the documentation-only scope.

## Known documentation gaps

No real captured command output is promised because values depend on local models/hardware. No
OpenVINO/TensorRT/static INT8/QAT/INT4 instructions are presented as available because those
features do not exist. The docs describe ONNX provider validation as external/pending. Bibliography
peer-review status is not structured, so arXiv links are conservatively marked as preprint versions.

## Verification

Central help was generated for the root group and all 26 nested commands. Typer help and existing
CLI smoke checks verify the mirrored app. Documentation command tokens are checked by a new
lightweight verification script. Markdown links and examples use the installed `lab` executable.
