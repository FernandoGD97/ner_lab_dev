# Storage, sizing, distributed capture, and cleanup

## Temporary snapshot data

Temporary data lives under:

```text
<run>/explicability/_snapshots/<snapshot_id>/
├── COMPLETE
├── snapshot.json
├── parameters.npz                 # only when parameter_drift is enabled
└── rank_00000/
    ├── COMPLETE
    ├── shard.json
    ├── observations.parquet
    └── embeddings/
        ├── COMPLETE
        ├── index.json
        └── chunk_000000.npz ...
```

Embeddings use the local `chunked-npz-v1` format: independently compressed NumPy
NPZ chunks plus a JSON index. Each array has shape
`(stored_layers, observations_in_chunk, hidden_size)`. Observation IDs are stored
beside the array in each chunk. High-dimensional embeddings are deliberately not
written to Parquet.

Capture first writes dot-prefixed `.incomplete` directories and `INCOMPLETE`
markers. Chunk publication, shard completion, and snapshot publication use atomic
renames/markers. An interrupted directory is therefore not accepted by readers as
a complete snapshot.

### Snapshot names

* `step_000_pretrained`: before `Trainer.train()`;
* `step_001`, `step_002`, …: trigger order, not necessarily literal optimizer step;
* `step_final`: weights left loaded after training (normally the best checkpoint).

`snapshot.json` records checkpoint, epoch, global step, split, scope, layers,
hidden size, dtype, model identifier, observation/example counts, world size,
shard summaries, disk projection, bytes, extraction/write time, parameter-state
presence, and completion.

### Observation metadata

`observations.parquet` contains one row per captured model position, including:

* stable `observation_id`, `example_id`, `segment_id`, `document_id`, and
  `word_observation_id`;
* encoded row/window/token/word positions, tokenizer token ID/string, and character
  offsets where available;
* gold and predicted BIO labels and entity types;
* full token logits/probabilities, predicted confidence, gold confidence, entropy,
  gold-versus-best-other margin, and correctness.

Entity IDs and word/span pooling helpers exist in the package, but Part 1 snapshot
metadata is token-position oriented; it does not persist a separate entity table.

## Disk estimation

Before writing each snapshot, the implemented embedding estimate is:

```text
observations × hidden_size × stored_layers × bytes_per_value
× (1 + metadata_overhead_fraction)
```

If parameter drift is enabled, it adds:

```text
sum(parameter.numel × 4 bytes)
```

because parameter snapshots are stored after conversion to float32. Capture logs
expected new bytes, current `_snapshots` usage, projected usage, free filesystem
bytes, and the configured safety margin. It rejects the snapshot before chunks are
written when `expected + safety_margin > available`.

For illustration only, one million observations at hidden size 768, one layer,
and float16 require about 1.43 GiB before metadata overhead/compression. Float32
doubles that. Twelve stored layers are approximately twelve times one layer; ten
snapshots are approximately ten times one snapshot. Actual NPZ compression depends
on the representations and metadata.

### Cost controls

* `float16` halves embedding bytes relative to `float32`.
* `last_hidden_state` stores one layer.
* `selected_layers` scales with the selected count.
* `all_hidden_states` includes embedding output plus every Transformer layer.
* More frequent Trainer evaluation creates more `evaluation` snapshots.
* `epoch` creates one numbered capture per epoch callback.
* Disabling `parameter_drift` avoids parameter NPZ files.

## DDP and launcher behavior

Rank/world size come from initialized `torch.distributed`; otherwise the code checks
`RANK`/`WORLD_SIZE`, falling back to `SLURM_PROCID`/`SLURM_NTASKS`. Encoded row
indices are deterministically assigned as `rank, rank + world_size, ...`. Each rank
writes only `rank_NNNNN`, so ranks do not share embedding files and large matrices
are not gathered to rank zero. Rank zero waits at barriers, verifies all shard
`COMPLETE` markers, writes snapshot metadata/optional parameters, and atomically
publishes the snapshot.

Launcher environment variables must describe the initialized process group
correctly. Setting `WORLD_SIZE>1` without launching all ranks leaves missing shards
and makes snapshot completion fail.

## Permanent data

Part 2 writes deterministic, tidy tables under `tables/` with PyArrow Parquet and
ZSTD compression. Tables contain derived metrics, coordinates, top-k neighbors,
or compact summaries—not duplicate corpus-sized embedding matrices. They remain
after snapshot cleanup and are the source for Part 3.

Part 3 additionally writes `corpus_adaptation_fingerprint.parquet` and
`figure_manifest.parquet`. These are permanent and required for fingerprint/cross-
run comparison and reproducible figure auditing.

## Figures, animations, and report

* `figures/**/*.svg`: canonical static publication outputs. SVG has physical size,
  not a scientifically meaningful DPI.
* Optional `.pdf`/`.png`: CairoSVG derivatives when configured/installed.
* `animations/temporal_embedding_evolution.svg`: always possible from projection
  Parquet when animation is enabled.
* `animations/temporal_embedding_evolution.mp4`: optional local codec stack.
* No GIF writer is currently implemented despite the reserved `animation_gif` key.
* `report/index.html`: permanent searchable report that embeds relative SVG paths.

## Cleanup policies

### `on_success` (default)

`finalize` validates training status, mandatory analyses, every required output,
the manifest, figure sources, SVG structure/style, and figure-manifest membership.
It then writes the success manifest and `SUCCESS`, removes `_snapshots/`, and writes
`cleanup_state: removed`. Cleanup is never attempted before `SUCCESS` exists.

### `never`

The same validation and `SUCCESS` write occur, but `_snapshots/` is retained and
the manifest records `cleanup_state: retained`.

On any earlier failure, snapshots remain. If deletion itself raises `OSError`, the
manifest records `cleanup_failed` and the permanent outputs/`SUCCESS` remain. Do
not manually delete snapshots before required Part 2 work is complete. Once all
permanent tables exist, figures/report can be regenerated without snapshots.
