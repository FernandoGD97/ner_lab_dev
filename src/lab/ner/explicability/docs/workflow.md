# Workflow, recovery, and usage profiles

## Actual lifecycle

```text
lab run CONFIG
  ↓ automatic when enabled
initialize directory and capture step_000_pretrained
  ↓
normal Trainer training/evaluation
  ↓ callback at evaluation or epoch events
capture step_NNN
  ↓ after Trainer returns/restores its best model
capture step_final
  ↓
manifest status = awaiting_postprocessing

lab explicability analyze RUN
  ↓ Part 2 ZSTD Parquet tables
manifest status = analyses_complete

lab explicability plot RUN
  ↓ SVG catalogue, fingerprint, animation, and HTML report
figure/source/style validation
  ↓
manifest status = visualizations_complete

lab explicability finalize RUN
  ↓ mandatory analysis/output/manifest/figure validation
write SUCCESS
  ↓
delete _snapshots only for cleanup_policy=on_success
```

The implementation does **not** automatically call `analyze`, `plot`, or
`finalize` after training. This separation supports HPC wall-time and resource
planning. Do not call `finalize` immediately after training when Part 2/3 outputs
are required.

## Trigger behavior

`trigger: evaluation` captures from `TrainerCallback.on_evaluate`. The callback
ignores the extra training-set evaluation performed by the existing
`metrics_scope: both` path, recognized by `train_` metric keys. A numbered snapshot
therefore corresponds to normal validation evaluation.

`trigger: epoch` captures from `on_epoch_end`. With the default epoch evaluation
strategy the schedules are often similar, but they are not identical: changing
Trainer evaluation frequency affects only the evaluation trigger.

The pretrained and final snapshots are unconditional once explicability is enabled.
`step_final` is the model left in memory by Trainer—normally the selected best
checkpoint because local defaults enable `load_best_model_at_end`.

## End-to-end examples

Training examples change YAML; no snapshot-trigger CLI options exist.

### A. Train with explicability enabled

```yaml
task: ner.train_model
split_dir: data/prepared_split
output_dir: runs/disease_baseline
base_model: path/or-model-identifier
target_label: DISEASE
language: en
explicability:
  enabled: true
```

```bash
lab run disease.yaml
lab explicability analyze runs/disease_baseline
lab explicability plot runs/disease_baseline
lab explicability finalize runs/disease_baseline
```

The non-explicability keys are actual `train_model` parameters; their values are
illustrative and must point to locally available model and split resources.

### B. Snapshot after every epoch

```yaml
explicability:
  enabled: true
  snapshots:
    trigger: epoch
```

Run with `lab run CONFIG`. There is no `--trigger` option.

### C. Snapshot after every validation evaluation

```yaml
explicability:
  enabled: true
  snapshots:
    trigger: evaluation
```

Trainer's configured evaluation schedule determines frequency.

### D. Manual completion or recovery

```bash
lab explicability analyze RUN_DIRECTORY
lab explicability plot RUN_DIRECTORY
lab explicability finalize RUN_DIRECTORY
```

Use this after a normal training job, after moving post-processing to another
allocation, or after an analysis/plot job failed. Stages use deterministic names
and atomic table writes where implemented.

### E. Regenerate only figures

```bash
lab explicability plot RUN_DIRECTORY --no-animation
lab explicability plot RUN_DIRECTORY --analysis cka --no-animation
lab explicability plot RUN_DIRECTORY --figure 18_dataset_cartography --no-animation
```

Raw snapshots are unnecessary. Selective commands leave visualization status
partial. Run an unfiltered `plot` before finalization.

### F. Presentation figures

```bash
lab explicability plot RUN_DIRECTORY --profile presentation
```

The override applies to that generation and is recorded in figure metadata; it
does not rewrite `config.yaml`.

### G. Preserve raw snapshots

```yaml
explicability:
  enabled: true
  snapshots:
    cleanup_policy: never
```

Finalization writes `SUCCESS`, retains `_snapshots/`, and records
`cleanup_state: retained`.

## Recovery scenarios

### Training completed; post-processing did not run

With status `awaiting_postprocessing`, run `analyze`, `plot`, then `finalize` in a
new process or allocation.

### Training or snapshot capture failed

The wrapper records `training_failed` when the error passes through it. Completed
and `.incomplete` snapshot directories remain. Do not finalize; diagnose disk,
model, or DDP errors. Training into the same claimed run directory is not a general
resume mechanism.

### SLURM wall time expired during analysis

Snapshots remain because analysis never cleans them. Re-run `analyze`. A partially
written dot-prefixed `.tmp` file is not a registered permanent output.

### Plotting failed

Fix the missing/corrupt table or optional export dependency and rerun `plot`.
`finalize` refuses `analyses_complete`, partial visualization, and invalid figures.

### A finalized run needs new styling

If permanent tables, `config.yaml`, and `manifest.json` remain, rerun `plot`; raw
embeddings are unnecessary. A selective plotting run marks the manifest partial.

## Recommended profiles

These are recommendations made from existing keys, not named software modes.

### Lightweight

* `embedding_scope: last_hidden_state`, `dtype: float16`, `trigger: epoch`;
* disable methods not needed, especially `parameter_drift` if parameter snapshots
  are too costly;
* use the paper profile.

### Representation research

* use `selected_layers` for embedding, middle, and final layers;
* use `trigger: evaluation`;
* keep trajectories, CKA, geometry, intrinsic dimension, neighborhoods, and
  dynamics enabled;
* set `sample_size` only for an explicit documented analytical sample.

### Full research

* use `all_hidden_states`, evaluation triggers, and full population;
* optionally enable SVCCA, PWCCA, and MDL probing;
* plan scratch/storage carefully and retain snapshots only as long as required.
