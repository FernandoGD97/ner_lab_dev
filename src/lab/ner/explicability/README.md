# NER explicability

`lab.ner.explicability` records how a pretrained Transformer changes while it is
fine-tuned for named-entity recognition (NER), derives compact scientific tables,
and turns those tables into validated SVG figures and an HTML report.

The scientific questions are longitudinal: how far representations move, which
layers change, whether entity classes become more compact or separable, when
examples are learned or forgotten, and where parameter updates accumulate. The
same stable observations are followed from `step_000_pretrained`, through numbered
training snapshots, to `step_final`.

> **CLI implementation note:** the local project uses Python `argparse`, not Typer.
> The installed executable is `lab`. There is no `ner` executable and there are no
> hidden command aliases. This documentation uses the commands that actually exist.

## Quick start

1. Add `explicability` to the existing `ner.train_model` YAML configuration:

   ```yaml
   task: ner.train_model
   # Existing train_model keys such as split_dir, output_dir, base_model,
   # target_label, and language remain required.
   explicability:
     enabled: true
     snapshots:
       trigger: evaluation
       split: validation
       embedding_scope: last_hidden_state
       dtype: float16
       cleanup_policy: on_success
   ```

2. Train through the existing task runner:

   ```bash
   lab run train.yaml
   ```

3. Produce permanent analytical tables, figures, and the report:

   ```bash
   lab explicability analyze RUN_DIRECTORY
   lab explicability plot RUN_DIRECTORY --profile paper
   lab explicability finalize RUN_DIRECTORY
   ```

`analyze` and `plot` are deliberately manual. Successful training records that
post-processing is pending; it does not silently run expensive analyses or delete
snapshots. `finalize` validates registered outputs, writes `SUCCESS`, and removes
`_snapshots/` only when `cleanup_policy: on_success` and every check succeeds.

## What happens during training

When `enabled: true`, the existing Hugging Face Trainer receives an
`ExplicabilityCallback`. The sequence is:

```text
build pretrained model
→ capture step_000_pretrained on the validation rows
→ run normal training/evaluation
→ capture step_NNN after each configured evaluation or epoch
→ Trainer restores its final/best weights
→ capture step_final
→ record awaiting_postprocessing in manifest.json
```

Extraction runs under inference mode and writes CPU chunks incrementally; the
complete corpus is not retained in GPU memory. With explicability disabled (the
default), no callback or explicability directory is created.

## Command tree

```text
lab
├── run CONFIG [--random-state/--seed N] [--output-dir DIR]
└── explicability
    ├── analyze RUN_DIRECTORY
    ├── plot RUN_DIRECTORY [--profile paper|presentation]
    │                         [--analysis NAME] [--figure FIGURE_ID]
    │                         [--no-animation]
    ├── animate RUN_DIRECTORY
    ├── compare OUTPUT_DIRECTORY RUN_DIRECTORY [RUN_DIRECTORY ...]
    │                          [--profile paper|presentation]
    └── finalize RUN_DIRECTORY [--completed-analysis NAME ...]
                                  [--mandatory-analysis NAME ...]
                                  [--required-output RELATIVE_PATH ...]
```

Run `lab explicability <command> --help` for the parser-generated synopsis.

## Outputs

All artifacts live below `<run>/explicability/`:

```text
config.yaml                  resolved explicability configuration
manifest.json                lifecycle, snapshots, analyses, figures, sizes, status
_snapshots/                  temporary complete embeddings and parameter states
tables/                      permanent ZSTD Parquet analytical tables
figures/                     canonical SVG figures grouped by topic
animations/                  animated SVG; optional MP4 when dependencies exist
examples/                    reserved case-study artifacts
report/index.html            searchable scientific report embedding the SVGs
SUCCESS                      written only by successful finalization
```

Figures can be regenerated after `_snapshots/` has been removed because plotting
reads `tables/`, `config.yaml`, and `manifest.json`, not raw embeddings.

## Documentation

* [CLI reference](docs/cli.md)
* [Configuration reference](docs/configuration.md)
* [Lifecycle and recovery workflows](docs/workflow.md)
* [Outputs and Parquet schemas](docs/outputs.md)
* [Scientific analyses and metrics](docs/analyses.md)
* [Visualization, animation, and report](docs/visualization.md)
* [Storage, disk estimates, DDP, and cleanup](docs/storage.md)
* [HPC and SLURM guidance](docs/hpc.md)
* [Troubleshooting](docs/troubleshooting.md)
* [Developer guide and extension tutorials](docs/developer_guide.md)
* [Scientific Visualization Guide](SCIENTIFIC_VISUALIZATION_GUIDE.md)
* [Local scientific reference catalogue](references/README.md)

## Scientific caveats

Dimensionality reduction is a visualization, not evidence of a mechanism. Probe
scores measure decodability, not causal use. Attention is not automatically an
explanation. Correlation with F1 does not imply causality. Attribution requires
faithfulness checks. Representation shifts are meaningful only under stable
observation matching and a shared/aligned coordinate system.
