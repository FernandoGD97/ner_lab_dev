# Complete CLI reference

[Documentation home](README.md) · [Configuration](configuration.md) · [Tutorials](tutorials/README.md) · [Troubleshooting](troubleshooting.md)

## Entry point and command tree

After installing the project, invoke the central executable:

```bash
lab quantisation --help
# Equivalent group alias:
lab quantization --help
```

`pyproject.toml` maps `lab` to `lab.cli:main`. The central application is implemented with
`argparse`, and `lab.cli.build_parser()` calls `lab.ner.quantisation.cli.add_parser()` through a
lazy import. In parallel, `lab.ner.quantisation.cli` defines an embeddable Typer `app`, with nested
`experiment_app` and `el_app`. The installed syntax documented below is the central `lab` syntax.
There is no `nerlab` executable and no `python -m ner.cli` entry point.

```text
lab
└── quantisation (alias: quantization)
    ├── inspect
    ├── methods
    ├── method
    ├── fp16
    ├── bf16
    ├── int8
    ├── export-onnx
    ├── prune
    ├── prune-vocabulary
    ├── reduce-depth
    ├── svd
    ├── validate
    ├── run-recipe
    ├── experiment
    │   ├── validate
    │   ├── run
    │   ├── status
    │   ├── summarize
    │   ├── compare
    │   ├── analyse
    │   ├── pareto
    │   ├── correlations
    │   ├── compare-predictions
    │   └── joint-report
    └── el
        ├── validate
        ├── run
        └── summarize
```

All relative paths are interpreted from the process working directory. The CLI does not change
into the YAML file's directory.

## Common task selection

Commands with `--task` accept strings consumed by `load_checkpoint`. The implemented values are:

| Value | Behavior |
|---|---|
| `auto` | Default. Inspects `config.architectures`; chooses token classification, sequence classification, or base model. |
| `base` | Uses `AutoModel`. |
| `token-classification` | Uses `AutoModelForTokenClassification`. |
| `sequence-classification` | Uses `AutoModelForSequenceClassification`. |

Any other value eventually fails with a Python key error; use one of the values above. Commands
print JSON for inspection/metadata/validation and print an output path for transformations.

---

## Discovery and inspection commands

### `lab quantisation inspect MODEL`

**Purpose / when to use:** load a local path or Hugging Face model identifier and inspect it before
choosing a transformation. This command loads the complete model and tokenizer; it is not a
config-only operation.

```bash
lab quantisation inspect MODEL [--task auto]
```

| Input | Required | Default | Meaning |
|---|---:|---:|---|
| `MODEL` | yes | — | Local checkpoint directory or Hugging Face identifier. |
| `--task` | no | `auto` | One of the task values in the table above. |

**Output:** JSON containing source, architecture/model type, loaded class, resolved task, total and
trainable parameters, observed parameter dtypes, local directory bytes (zero for a non-local model
identifier), vocabulary/hidden/FFN sizes, layer/head counts, label count, tokenizer class, and
special-token map. It writes no files.

**Validation and failures:** Hugging Face performs config/model/tokenizer loading. Expect failure for
missing paths, offline uncached Hub models, a task/class mismatch, missing PyTorch, or unsupported
custom code. Example:

```bash
lab quantisation inspect artifacts/ner-fp32 --task token-classification
```

### `lab quantisation methods`

**Purpose:** list the FP32 baseline metadata plus every registered implementation.

```bash
lab quantisation methods
```

No arguments or side effects. The JSON array includes name, family, scientific category, training
requirement, supported architecture/task labels, fidelity, limitations, publications, and reference
implementations. Categories are `model_compression`, `computational_compression`, and
`runtime_optimization`. There is no explicit `experimental` boolean; experimental status is stated
in limitations/fidelity and this manual. Listing metadata is not a compatibility test for a
particular checkpoint.

### `lab quantisation method NAME`

**Purpose:** show one metadata record before using a command.

```bash
lab quantisation method fp16
lab quantisation method int8-dynamic
```

`NAME` is required. Registered names are `fp32`, `fp16`, `bf16`, `int8-dynamic`,
`magnitude-pruning`, `vocabulary-pruning`, `depth-reduction`, `svd`, and `onnx`. `fp32` has metadata
but no transformation command. An unknown name raises `ValueError`.

---

## Transformation commands

All transformation commands:

* refuse to use the source directory as the output;
* refuse a non-empty output directory;
* create the output directory;
* copy source `encoding.json` when present;
* load the source model and tokenizer;
* write method-specific model/runtime files;
* write `compression_manifest.json` and `compression_recipe.yaml`;
* never modify the source checkpoint.

Normal Hugging Face methods call their validation hook after writing. Dynamic INT8, SVD, and ONNX
have specialized runtime constraints described below.

### `fp16`

```bash
lab quantisation fp16 MODEL --output OUTPUT [--task auto]
```

| Parameter | Required | Default | Description |
|---|---:|---:|---|
| `MODEL` | yes | — | Source checkpoint. |
| `--output` | yes | — | New, absent or empty directory. |
| `--task` | no | `auto` | Model AutoClass selection. |

**Effect:** calls `model.half()`, saves a normal safetensors Hugging Face checkpoint and tokenizer,
writes provenance, then reload-validates. Parameter count and dense FLOPs do not change.
**Common failures:** CPU FP16 smoke kernels may be unavailable; source/output collision; missing
weights/tokenizer; insufficient disk. See the [FP16 tutorial](tutorials/fp16.md).

### `bf16`

```bash
lab quantisation bf16 MODEL --output OUTPUT [--task auto]
```

Arguments, outputs, and validation are the same as FP16. The model is converted with
`torch.bfloat16`. Efficient execution requires BF16-capable hardware; checkpoint creation does not
guarantee speedup. See [precision methods](methods/precision.md).

### `int8`

```bash
lab quantisation int8 MODEL --output OUTPUT [--task auto]
```

**Effect:** performs PyTorch dynamic post-training quantization of `torch.nn.Linear` modules on CPU.
It writes `quantized_model.pt`, config/tokenizer files, provenance, and copied `encoding.json`; it
does **not** write a normal `model.safetensors` that `AutoModel` can load. The canonical NER loader
recognizes backend `pytorch_dynamic_int8`, and `validate` uses the specialized loader.

**Limitations/failures:** CPU only; not static/calibrated PTQ, QAT, Q-BERT, or I-BERT; packed INT8
parameters do not expose a trustworthy dense nonzero count. See the [INT8 tutorial](tutorials/int8.md).

### `prune`

```bash
lab quantisation prune MODEL --output OUTPUT [--amount 0.2] [--task auto]
```

`--amount` is a floating-point requested fraction; default `0.2`. The implementation does not
explicitly range-check it, so researchers should use `0 < amount < 1`. Each matrix parameter
(`ndim >= 2`) has its lowest-magnitude values zeroed in place. The normal Hugging Face artifact is
validated. Dense shapes, parameter count, dense FLOPs, and usually file size do not fall merely
because values are zero. Fine-tuning is recommended but not performed.

### `prune-vocabulary`

```bash
lab quantisation prune-vocabulary MODEL --output OUTPUT \
  [--keep-token TOKEN ...] [--task auto]
```

`--keep-token` is repeatable and defaults to no user tokens. All tokenizer special tokens plus user
kept tokens define the highest ID that must remain; only the contiguous vocabulary tail above that
ID is removed. The embedding is resized and `vocab.txt` rebuilt while retained IDs stay unchanged.

**Supported mutation:** `BertTokenizer`/WordPiece only. XLM-R/SentencePiece is explicitly
analysis-only and this command raises rather than renumbering it unsafely. It also fails when a
requested token is missing or when retained IDs leave no tail to prune. See the
[vocabulary tutorial](tutorials/vocabulary.md).

### `reduce-depth`

```bash
lab quantisation reduce-depth MODEL --output OUTPUT --layers N [--task auto]
```

`--layers` is required and must be between 1 and the source depth minus one. It keeps a prefix of a
recognized BERT-like encoder `ModuleList`, updates `num_hidden_layers`/`n_layers`, saves and validates
a normal checkpoint. It is experimental and requires subsequent fine-tuning or distillation for a
credible model. Unsupported encoder layouts raise an error.

### `svd`

```bash
lab quantisation svd MODEL --output OUTPUT [--rank-ratio 0.5] [--task auto]
```

`--rank-ratio` defaults to `0.5` and must be strictly between zero and one. Beneficial linear layers
are replaced by two factors initialized from a truncated SVD. The output is a specialized runtime
artifact: `low_rank_state.pt` plus `low_rank.json`, config, tokenizer, encoding, manifest, and recipe.
Load with the internal specialized loader/canonical NER loader, not plain `AutoModel`. It fails if
no layer benefits at the requested ratio. Fine-tuning is recommended; latency may increase without
fused kernels.

### `export-onnx`

```bash
lab quantisation export-onnx MODEL --output OUTPUT [--task auto]
```

Exports `model.onnx` with opset 17, dynamic batch and sequence axes, plus config/tokenizer/provenance.
The optional `onnx` package is required. This is a runtime artifact, not an ordinary Hugging Face
checkpoint. `lab quantisation validate` intentionally reports that an installed ONNX Runtime
provider is required rather than pretending to validate it through PyTorch. No CLI option exposes
a different opset.

### `validate`

```bash
lab quantisation validate MODEL [--source SOURCE]
```

`MODEL` is the artifact directory. `--source` is optional and enables source task/label comparisons.
The command prints a boolean JSON report; it does not print PASS/WARNING status words. See the
[validation reference](validation.md) for exact checks and backend behavior.

### `run-recipe`

```bash
lab quantisation run-recipe RECIPE [--output OUTPUT]
```

Loads the compression YAML with Pydantic. `--output` overrides only `output.path`; otherwise the
YAML path is used. Method-specific extra keys under `method` are passed to `apply`, except `name` and
`mode`, which are removed; notably, `mode` is validated/stored but currently not forwarded. Unknown
method IDs fail in the registry. Generated files are exactly those of the selected method.

---

## Experiment commands

Experiment commands require the Pydantic YAML described in [configuration.md]. Result directories
are `output.directory / experiment.id`.

### `experiment validate`

```bash
lab quantisation experiment validate CONFIG
```

Loads/validates YAML, checks the dataset path and each model. Models are classified `READY`,
`MISSING`, `INVALID`, or `UNSUPPORTED`; the output is JSON. A normal/specialized PyTorch artifact
must have `encoding.json`, a locally loadable config/tokenizer, a recognized A0–C6 ID, and a backend
in `transformers`, `pytorch_dynamic_int8`, or `pytorch_low_rank`. Validation writes nothing.

### `experiment run`

```bash
lab quantisation experiment run CONFIG [--dry-run] [--resume] [--force]
```

| Option | Default | Exact behavior |
|---|---:|---|
| `--dry-run` | false | Writes resolved configuration, environment, order, and study manifest; launches no workers. |
| `--resume` | false | Skips only run JSON files whose status is exactly `SUCCESS`; retries other existing runs. |
| `--force` | false | Deletes the complete result directory before recreating it. |

`--resume` and `--force` are mutually exclusive. Each model × repetition runs in a fresh subprocess.
Non-ready models become `SKIPPED` or `UNSUPPORTED`; one failure does not abort later entries. Outputs
are detailed in [experiments.md](experiments.md). A pre-existing run without either resume or force
raises `FileExistsError` when reached.

### `experiment status`

```bash
lab quantisation experiment status RESULTS_DIR
```

Reads terminal per-run `run.json` files and prints counts for all run statuses. It does not inspect
GPU processes, repair files, or launch work. Corrupt/unreadable and non-terminal `RUNNING` records
are omitted by the collector.

### `experiment summarize`

```bash
lab quantisation experiment summarize RESULTS_DIR
```

Rebuilds `benchmark_summary.tsv` from terminal per-run JSON. Successful runs contribute numeric
statistics; raw run records remain untouched. Prints the written path.

### `experiment compare`

```bash
lab quantisation experiment compare RESULTS_DIR [--baseline A0]
```

Rebuilds `comparison.tsv`. The baseline defaults to `A0`. Only successful runs are averaged. It
calculates separate parameter/checkpoint/peak-VRAM/energy reductions, forward/end-to-end speedups,
and precision/recall/F1 deltas. Missing denominators remain empty; it does not rank models.

### `experiment analyse`

```bash
lab quantisation experiment analyse RESULTS_DIR \
  [--training-frequency-source TRAIN.tsv] [--baseline A0] [--plots]
```

Reads saved benchmark and prediction files; it never reruns inference. It writes robustness tables,
frequency-definition metadata, paired bootstrap results, correlations, compression-family summary,
Pareto files, paper TSV/CSV tables, and optional plot data/images under `analysis/`. The training
source must be a TSV span table or parquet with a `text` column and must genuinely come from
training data.

**Current precedence caveat:** analysis/statistics settings are also read from
`resolved_experiment.yaml`. The resolved `analysis.plots` value is read after the CLI argument and
therefore can override `--plots`; set `analysis.plots: true` in YAML for reliable plotting. This is
a discovered implementation inconsistency, not intended precedence.

### `experiment pareto`

```bash
lab quantisation experiment pareto RESULTS_DIR
```

Reads `benchmark_long.tsv`, averages successful numeric runs by model, and writes separate
`analysis/pareto_f1_*.tsv` files. It has no options and produces no plots.

### `experiment correlations`

```bash
lab quantisation experiment correlations RESULTS_DIR
```

Reads `benchmark_long.tsv` and writes `analysis/correlations.tsv` with Pearson and rank-based
Spearman coefficients for available configured pairs. These are descriptive, not causal.

### `experiment compare-predictions`

```bash
lab quantisation experiment compare-predictions RESULTS_DIR \
  [--baseline A0] --model MODEL_ID [--training-frequency-source TRAIN.tsv]
```

`--model` is required. Reads the baseline gold/prediction TSVs and the selected model prediction
TSV. Writes detailed and summary files under `analysis/`, including exact recoveries,
shared/new/removed false positives, label/boundary changes, mention length, optional training
frequency, and subword counts when the selected artifact tokenizer is locally loadable.

### `experiment joint-report`

```bash
lab quantisation experiment joint-report NER_RESULTS \
  --el-results EL_RESULTS [--baseline A0]
```

Joins model-level NER and EL tables by `model_id` and writes
`NER_RESULTS/analysis/joint_ner_el.tsv`. It keeps task metrics separate and constructs no combined
score. Both long tables must already exist.

---

## Entity Linking commands

### `el validate`

```bash
lab quantisation el validate CONFIG
```

Pydantic-validates EL YAML, checks mention/gazetteer paths, optional cross-encoder path, and index
support. The current adapter marks only `flat_ip` and `flat_l2` ready; scalar/PQ/IVF/IVF-PQ/HNSW
are explicit `UNSUPPORTED`. Prints JSON and writes nothing.

### `el run`

```bash
lab quantisation el run CONFIG
```

Uses existing `lab.nel` loaders, candidate generator, retrieval metrics, and optional reranker. It
runs configured K values/repetitions and writes `el_benchmark_long.tsv`, `el_predictions.tsv`,
`el_benchmark_summary.tsv`, and `el_validation.json`. If validation is not ready it writes only the
validation report and returns. Optional dependencies depend on the selected NEL method; install
`.[all]` for the complete stack.

### `el summarize`

```bash
lab quantisation el summarize RESULTS_DIR
```

Reads `el_benchmark_long.tsv`, groups by model ID and candidate K, and rewrites
`el_benchmark_summary.tsv` with mean, median, and standard deviation. It does not rerun linking.

## Exit and error behavior

The installed central CLI catches `ValueError`, `FileNotFoundError`, and `TypeError`, prints
`lab: MESSAGE`, and exits 2. `argparse` syntax errors also exit 2. Other exceptions—including many
optional-dependency, PyTorch, pandas, or runtime failures—can surface with a traceback. Successful
handlers return through `lab.cli.main()` with exit 0. Help exits 0.

## Newly executable hardening commands

### `quantisation int8-static`

`lab quantisation int8-static MODEL --output DIR --calibration-corpus PATH [--calibration-samples 128] [--calibration-seed 42] [--batch-size 8] [--max-length 256] [--task auto]`

Produces a CPU reference W8A8 artifact. Calibration input is required; invalid/empty inputs fail. The corpus is read only and calibration provenance is recorded.

### `quantisation int4`

`lab quantisation int4 MODEL --output DIR [--group-size 64] [--asymmetric] [--task auto]`

Produces a packed CPU weight-only INT4 artifact. Group size must be positive. Without `--asymmetric`, symmetric signed quantisation is used.

### `quantisation prune-structured`

`lab quantisation prune-structured MODEL --output DIR [--head-amount 0] [--ffn-amount 0] [--task auto]`

Physically prunes supported attention heads and/or BERT-like FFN channels. Supplying two zero amounts is an error.

### `quantisation student`

`lab quantisation student MODEL --output DIR --layers N --hidden-size N [--intermediate-size N] [--attention-heads N] [--seed 42]`

Builds and deterministically initializes a structurally valid token-classification student. This is **not a trained/distilled model**; use it as training input, not as a completed B8 result.
