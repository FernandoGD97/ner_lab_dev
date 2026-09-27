# CLI reference

## Actual CLI framework and error model

The root application is defined by `build_parser()` in `lab/cli.py` and installed
as the `lab` console script. It is an `argparse.ArgumentParser`; the project does
**not** currently define a Typer application. `explicability` is a nested argparse
subparser. Imports are lazy so `lab --help` does not import PyTorch or the NER
stack.

There are no command aliases. The only option alias relevant to training is the
root task runner's `--seed`, which is an alias of `--random-state`.

Exit behavior:

| Situation | Exit status | Behavior |
|---|---:|---|
| Successful command | 0 | Handler returns normally. |
| No root handler, such as bare `lab` or bare `lab explicability` | 1 | Root help is printed. |
| Parser error or missing required argument | 2 | argparse prints usage and an error. |
| Handler raises `ValueError`, `FileNotFoundError`, or `TypeError` | 2 | `lab: <message>` is printed. |
| Other exception, such as `OSError`, `ImportError`, or some Parquet/XML errors | non-zero | The exception is not translated by `main()` and may include a traceback. |

No explicability command reads special environment variables. Distributed rank
discovery during **training snapshot capture** uses initialized
`torch.distributed`, then `RANK`/`WORLD_SIZE`, with `SLURM_PROCID`/`SLURM_NTASKS`
as fallbacks. Those variables do not change the post-processing CLI syntax.

## `lab run`

### Purpose

Run the project's existing YAML-configured task. This is how explicability is
enabled for `ner.train_model`; there are no dedicated snapshot options on the
command line.

### Syntax

```bash
lab run CONFIG [--random-state N | --seed N] [--output-dir DIRECTORY]
```

### Arguments and options

| Name | Required | Default | Meaning |
|---|---:|---:|---|
| `CONFIG` | yes | — | YAML file whose top-level `task` selects a registered task. |
| `--random-state N`, `--seed N` | no | YAML value/task default | Overrides top-level `random_state`; it does not rewrite `explicability.analyses.sample_seed` or visualization seeds. |
| `--output-dir DIRECTORY` | no | YAML value | Overrides the task's top-level run directory. |

### Example

```bash
lab run configs/disease_ner.yaml --seed 17 --output-dir runs/disease_seed17
```

The YAML must contain `task: ner.train_model` and an `explicability` mapping. The
task runner removes only the `task` key and passes all other top-level keys to
`train_model()`.

### Failure behavior

A non-mapping YAML document, missing `task`, unknown task, or invalid task argument
is reported with exit status 2. Training/runtime failures outside the three caught
exception classes propagate.

## `lab explicability analyze`

### Purpose

Transform complete Part 1 snapshots into Part 2 permanent ZSTD Parquet tables.

### Syntax

```bash
lab explicability analyze RUN_DIRECTORY
```

### Arguments and options

`RUN_DIRECTORY` is required. It is the training run directory that contains the
`explicability/` child, not normally the `explicability/` directory itself. There
are no command options: enabled analyses and numerical settings come from
`explicability/config.yaml`.

### Input requirements

* `explicability/config.yaml`;
* `explicability/manifest.json` with complete registered snapshots;
* `explicability/_snapshots/<snapshot>/COMPLETE` and complete rank shards;
* parameter snapshots when `parameter_drift: true`.

### Generated outputs and side effects

Enabled methods write `explicability/tables/*.parquet`. The manifest gains
`completed_analyses`, `mandatory_analyses`, `required_outputs`, analysis parameters,
sample count, runtime, storage size, and status `analyses_complete`. Snapshots are
not deleted and `SUCCESS` is not written.

### Example

```bash
lab explicability analyze runs/disease_seed17
```

### Failure behavior

The command fails before cleanup if stable IDs do not intersect, a snapshot is
incomplete, an enabled analysis produces no rows, parameter states are absent, or
an optional corpus-wide method (`attribution`, `influence`, `topology`, or
`adaptation_phases`) is enabled. Those targeted/dependency-gated methods are not
accepted by the all-snapshot orchestrator.

## `lab explicability plot`

### Purpose

Regenerate Part 3 SVG figures, `figure_manifest.parquet`, the numerical fingerprint,
animations when requested, and `report/index.html` from permanent tables.

### Syntax

```bash
lab explicability plot RUN_DIRECTORY \
  [--profile {paper,presentation}] \
  [--analysis NAME] [--figure FIGURE_ID] [--no-animation]
```

### Options

| Option | Default | Meaning |
|---|---|---|
| `--profile` | value in `config.yaml` (`paper` by default) | Override typography and physical canvas for this regeneration. Scientific values do not change. |
| `--analysis NAME` | all applicable analyses | Generate catalogue entries whose analysis name equals `NAME`. This is an exact internal analysis name, for example `cka` or `probing`. |
| `--figure FIGURE_ID` | all applicable figures | Generate one exact catalogue ID, for example `18_dataset_cartography`. |
| `--no-animation` | false | Skip animated SVG and optional MP4 during this plotting pass. |

`--analysis` and `--figure` may both be supplied, but a figure must satisfy both
filters. A selective pass is marked `visualization_status: partial` and cannot
unlock finalization. Existing manifest rows for regenerated IDs are replaced;
other rows remain. A full unfiltered pass is required for
`visualization_status: complete`.

### Input requirements

`config.yaml`, `manifest.json`, and the source tables for applicable figures.
Raw `_snapshots/` are not required.

### Example

```bash
lab explicability plot runs/disease_seed17 --analysis cka --no-animation
lab explicability plot runs/disease_seed17 --profile presentation
```

### Failure behavior

Missing/empty source tables cause figures to be skipped; if the requested
selection yields no applicable figure, the command exits with a value error.
Malformed SVG, missing axis metadata, missing source Parquet, empty source data, or
missing manifest membership fails validation and leaves snapshots untouched.
Optional PDF/PNG export raises `ImportError` after SVG is written if `cairosvg` is
not installed.

## `lab explicability animate`

### Purpose and syntax

```bash
lab explicability animate RUN_DIRECTORY
```

Regenerate `animations/temporal_embedding_evolution.svg` independently from
`tables/projection.parquet`. The highest stored layer and token representation are
used. Frames require identical stable observation IDs and fixed shared coordinates.

MP4 is additionally attempted at
`animations/temporal_embedding_evolution.mp4`. If optional `cairosvg` or `imageio`
is unavailable, `export_mp4()` returns no MP4 without failing animated-SVG creation.
The configured `animation_fps`, palette, visual sampling limit, and seed apply.
Although `animation_gif` exists in configuration, the current implementation does
not emit GIF files.

## `lab explicability compare`

### Purpose

Compare already generated corpus-adaptation fingerprints across runs without
snapshots.

### Syntax

```bash
lab explicability compare OUTPUT_DIRECTORY RUN_DIRECTORY [RUN_DIRECTORY ...] \
  [--profile {paper,presentation}]
```

### Arguments and options

| Name | Required/default | Meaning |
|---|---|---|
| `OUTPUT_DIRECTORY` | required | Destination for the comparison table and SVG; this is not a run directory. |
| `RUN_DIRECTORY ...` | one or more required | Runs containing `tables/corpus_adaptation_fingerprint.parquet`. |
| `--profile` | `paper` | Comparison-figure profile. |

### Outputs

* `OUTPUT_DIRECTORY/cross_run_fingerprints.parquet`
* `OUTPUT_DIRECTORY/cross_run_adaptation_fingerprint.svg`

Run/model/seed fields are copied from each manifest. A missing fingerprint raises
`FileNotFoundError` and exits with status 2.

## `lab explicability finalize`

### Purpose

Perform the only operation that writes `SUCCESS` and may delete `_snapshots/`.
Finalization does not automatically run `analyze` or `plot`.

### Syntax

```bash
lab explicability finalize RUN_DIRECTORY \
  [--completed-analysis NAME ...] \
  [--mandatory-analysis NAME ...] \
  [--required-output RELATIVE_PATH ...]
```

Each optional flag is repeatable (`action="append"`).

| Option | Default | Meaning |
|---|---|---|
| `--completed-analysis NAME` | empty list | Add names to `completed_analyses` for this finalization call. Use only for externally completed work whose outputs are also registered. |
| `--mandatory-analysis NAME` | manifest value | Replace the mandatory-analysis list for this call. Supplying the option once or more replaces rather than extends the manifest list. |
| `--required-output RELATIVE_PATH` | manifest value | Replace required outputs for this call. Paths are relative to `<run>/explicability/`. Parquet files are opened to validate readability. |

### Validation and side effects

Finalization refuses unfinished training, status `analyses_complete` without a full
Part 3 pass, pending mandatory analyses, incomplete snapshots, invalid manifests,
missing required files, unreadable Parquet, or invalid figure artifacts. It then:

1. writes the success manifest state;
2. writes `explicability/SUCCESS`;
3. removes `_snapshots/` only for `cleanup_policy: on_success`;
4. records `removed`, `retained`, or `cleanup_failed` in `cleanup_state`.

If cleanup itself fails, permanent outputs and `SUCCESS` remain valid and the error
is recorded. Never use command-line overrides to hide genuinely pending work.
