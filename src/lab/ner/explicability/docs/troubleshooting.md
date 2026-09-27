# Troubleshooting

## Insufficient disk before a snapshot

**Symptom:** `OSError: Insufficient disk for explicability snapshot before writing`.

**Cause:** estimated embeddings plus optional parameters and safety margin exceed
reported free space.

**Recovery:** snapshots are checked before chunk writing. Choose a larger run
filesystem and restart training, or reduce stored layers/dtype/snapshot frequency;
disable `parameter_drift` if parameter histories are unnecessary. There is no CLI
option to relocate snapshots within an existing run.

## `.incomplete` snapshot or chunk directory

**Symptom:** `Incomplete chunked array` or Part 2 refuses a snapshot.

**Cause:** interruption, rank failure, or write error before atomic completion.

**Recovery:** do not manufacture `COMPLETE`. Preserve the directory for diagnosis.
If a fully captured run is unavailable, recapture through training; `analyze` cannot
derive missing vectors.

## Training failed or wall time expired

**Symptom:** manifest status `training_failed` or remains `collecting`; no final
snapshot.

**Recovery:** do not run `finalize`. Completed snapshots remain safe, but the
current CLI has no resume-training-from-snapshots command. Fix training/resources
and run a properly claimed training run. If training actually completed and only
post-processing expired, status should be `awaiting_postprocessing`; run:

```bash
lab explicability analyze RUN
lab explicability plot RUN
lab explicability finalize RUN
```

## Missing rank shards / DDP completion error

**Symptom:** `Snapshot ... has incomplete rank shards: [...]`, or a barrier waits.

**Cause:** declared ranks did not all run capture, launcher variables do not match
the process group, or one process failed.

**Recovery:** inspect per-rank directories/logs and `RANK`, `WORLD_SIZE`,
`SLURM_PROCID`, `SLURM_NTASKS`. Relaunch consistently; do not have multiple ranks
write one shard.

## Invalid selected layer

**Symptom:** `Selected layer outside [0, N]` or duplicate resolved layers.

**Cause:** `selected_layers` references a nonexistent hidden-state tuple index, or
positive/negative indices resolve to the same layer.

**Recovery:** remember index 0 is the embedding output and the available count is
`num_hidden_layers + 1`. Correct YAML and rerun training.

## `selected_layers` is empty

With `embedding_scope: selected_layers`, a non-empty list is mandatory. Use
`last_hidden_state` when only the final encoder state is needed.

## Missing parameter states

**Symptom:** `parameter_drift enabled but snapshots contain no parameter states`.

**Cause:** snapshots were created without parameter capture (for example an older
run or `parameter_drift: false`) but analysis now enables it.

**Recovery:** disable `parameter_drift` in the saved run configuration before
analysis only if the study does not require it, or recapture training with it
enabled. Parameter trajectories cannot be reconstructed from embeddings.

## Missing optional UMAP dependency

**Symptom:** `fixed_umap`/`aligned_umap` raises an `ImportError` naming
`umap-learn`.

**Recovery:** install that optional dependency in the execution environment or use
`projection_method: shared_pca`. The project does not declare `umap-learn` in its
current dependency sets.

## Attribution/influence/topology/phase option rejected

**Symptom:** `Targeted/optional analyses require explicit case inputs...`.

**Cause:** the all-snapshot pipeline explicitly rejects `attribution`, `influence`,
`topology`, and `adaptation_phases`. Attribution/influence helper functions need
caller-supplied models/cases/gradients; topology and phase detection are not
implemented.

**Recovery:** keep those booleans false for `lab explicability analyze`. Use the
library helpers in custom targeted code only where implemented, then write a
schema-compatible permanent table if it should feed visualization.

## Enabled analysis produced no results

**Symptom:** `Enabled analysis 'NAME' produced no results`.

**Cause:** too few stable/gold-labelled observations, unavailable layers, or
required metadata did not survive intersection/filtering.

**Recovery:** inspect snapshot metadata and manifest counts. Disable the
inapplicable analysis or recapture appropriate observations. Do not create an empty
placeholder table; final validation rejects empty required sources.

## Missing/corrupt Parquet

**Symptom:** `plot` skips figures, requested selection yields no applicable figure,
or `finalize` says required outputs are missing/unreadable.

**Recovery:** rerun `lab explicability analyze RUN` while snapshots remain. PyArrow
must be installed. Remove stale dot-prefixed temporary files only after confirming
no writer is active.

## Corrupt or incompatible manifest

**Symptom:** missing fields, unsupported schema version, duplicate snapshot IDs, or
JSON parse error.

**Recovery:** restore the manifest from run backup/provenance. Do not guess snapshot
completion or delete raw data. `finalize` requires schema version 1 and registered
complete snapshots unless cleanup is already recorded as removed.

## Figure generation says no applicable figures

**Cause:** `--analysis` or `--figure` does not exactly match the catalogue, or the
source table is absent/empty. Names are case-sensitive internal identifiers.

**Recovery:** inspect `tables/`, use an implemented ID such as
`18_dataset_cartography`, or run an unfiltered plot:

```bash
lab explicability plot RUN --no-animation
```

## SVG/figure validation failure

**Symptom:** malformed/missing SVG, missing dimensions/axis labels, clipped text,
style metadata error, missing source, or absent figure-manifest entry.

**Recovery:** keep snapshots; fix the source/renderer problem and rerun `plot`.
Do not invoke `finalize` with overrides intended to bypass figure validation.

## PDF/PNG or MP4 is absent

PDF/PNG require optional `cairosvg`. MP4 requires CairoSVG, ImageIO, and a compatible
video codec. Animated SVG does not need them. `animation_gif` is currently unused;
no GIF is generated.

## Cleanup did not execute

Check:

* `cleanup_policy` (`never` intentionally retains snapshots);
* `visualization_status` and main status;
* pending mandatory analyses;
* required output readability;
* `SUCCESS` existence;
* `cleanup_state` (`cleanup_failed` includes `cleanup_error`).

Run `finalize` only after a complete `analyze` and unfiltered `plot`. If cleanup
failed after `SUCCESS`, retry removal manually only after independently verifying
the permanent run artifacts; the CLI does not expose a cleanup-only command.

## Bare `lab explicability` prints root help

This is current argparse behavior: the nested subparser is not marked required,
and `main()` sees no handler, prints root help, and returns status 1. Use a concrete
subcommand or `lab explicability <command> --help`.
