# HPC and SLURM guidance

This page describes behavior implemented in the local code; it does not prescribe
a site-specific scheduler script.

## Storage placement

All explicability artifacts are rooted in the training `output_dir`. Choose that
directory before training. There is no separate snapshot/scratch-path setting and
no CLI relocation option. If your site provides node-local or parallel scratch,
point the run `output_dir` there only if the run and its permanent results will be
copied/preserved according to site policy.

Complete hidden states and optional parameter states can dominate disk usage. Use
the sizing formula in [storage.md](storage.md), preserve a meaningful
`disk_safety_margin_bytes`, and check quotas as well as filesystem free space—the
implementation uses `shutil.disk_usage().free`, not a scheduler quota API.

## Distributed snapshot behavior

Under initialized `torch.distributed`, rank and world size come from the process
group. Otherwise capture reads:

1. `RANK`, falling back to `SLURM_PROCID`, default 0;
2. `WORLD_SIZE`, falling back to `SLURM_NTASKS`, default 1.

Each rank processes deterministic row indices and writes a separate
`rank_NNNNN` shard. No corpus embedding tensor is gathered to rank zero. Rank zero
writes the shared parameter NPZ and publishes the snapshot after barriers and
shard checks.

Do not manually set multi-rank variables for a single process. Every declared rank
must execute the callback and reach the barrier. A killed rank leaves the snapshot
incomplete, which is intentionally rejected by Part 2.

## Splitting training and post-processing jobs

A practical scheduler-neutral sequence is:

```text
GPU allocation:  lab run CONFIG
CPU/high-memory allocation: lab explicability analyze RUN
CPU allocation:  lab explicability plot RUN --profile paper
CPU allocation:  lab explicability finalize RUN
```

The commands do not submit jobs or express scheduler dependencies. Arrange those
with your local scheduler. Analysis uses NumPy/Pandas and is sequential despite a
validated `workers` configuration field; request memory appropriate to one layer
across aligned checkpoints. Plotting reads Parquet and is usually lighter.

## Wall-time interruptions

* Snapshot chunks use incomplete markers and atomic publication. Never rename an
  incomplete directory to `COMPLETE` manually.
* Analysis tables use dot-prefixed temporary files and atomic replace. Rerun
  `analyze` after interruption; snapshots have not been cleaned.
* Plotting uses deterministic paths. Rerun `plot`; a complete unfiltered run is
  required before finalization.
* `finalize` validates before cleanup. If the process stops after `SUCCESS` but
  during cleanup, inspect `cleanup_state` and `_snapshots/`; permanent outputs are
  still authoritative.

## Moving a run

Most explicability paths inside the artifact tree are relative, and CLI commands
accept the new run root. However, model/dataset provenance strings in manifests may
retain original absolute paths. Move the complete run—including `config.yaml`,
`manifest.json`, tables, and snapshots if analysis is pending—rather than moving
only shard subdirectories.
