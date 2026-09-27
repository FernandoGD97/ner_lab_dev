# NER explicability: Part 1

This package captures the same validation observations before fine-tuning and at
successive training states. It deliberately performs no dimensionality reduction
or scientific analysis. CKA, aligned UMAP, geometry, probing, attribution and
figures belong to Parts 2 and 3.

## Integration and lifecycle

`lab.ner.training.train` remains the only training pipeline. When explicability is
disabled (the default), it builds no callback and writes no explicability files.
When enabled, an `ExplicabilityCallback` captures `step_000_pretrained` immediately
before `Trainer.train()`, then captures after each evaluation or epoch. It reuses
the encoded validation rows and the in-memory model; normal evaluation remains
unchanged. A final `step_final` snapshot records the weights `Trainer` leaves
loaded (the selected best checkpoint under the repository defaults).

Each forward pass runs under `torch.inference_mode()`. Hidden states are detached,
moved to CPU, cast to the configured storage dtype and written a batch at a time.
No complete corpus-sized tensor is retained in GPU or host memory. The output tree
is:

```text
<run>/explicability/
  config.yaml
  manifest.json
  _snapshots/step_000_pretrained/rank_00000/...
  tables/ figures/ animations/ examples/ report/
```

Snapshots are first built under `.step_NNN.incomplete`. Every rank writes only its
own shard, including a Parquet observation index and compressed NPZ chunks. A
rank shard and then the snapshot receive `COMPLETE` markers before the snapshot is
atomically renamed. Interrupted directories therefore cannot be mistaken for
valid snapshots. Rows are assigned to ranks by `row_index % world_size`; shard
indices plus `(row_index, token_position)` provide deterministic global ordering
without gathering embeddings on rank zero.

Compressed, independently readable NPZ chunks were selected because NumPy is an
existing dependency while Zarr/numcodecs are not. This provides bounded-memory
incremental writes, compression, interruption isolation and simple HPC shards.
Part 2 can stream them with `iter_chunks`. High-dimensional embeddings are never
stored in Parquet. Permanent, narrow analytical tables use atomic ZSTD Parquet
through `write_table`, with stable sorting supplied by each table's schema.

## Configuration

The existing task YAML accepts this additional top-level argument:

```yaml
explicability:
  enabled: true
  snapshots:
    trigger: evaluation       # evaluation | epoch
    split: validation
    embedding_scope: last_hidden_state # selected_layers | all_hidden_states
    selected_layers: []
    dtype: float16            # float16 | float32
    include_all_examples: true
    include_all_tokens: true
    cleanup_policy: on_success # on_success | never
    disk_safety_margin_bytes: 2147483648
  analyses:
    temporal_projection: true
    trajectories: true
    centroid_dynamics: true
    cka: true
    geometry: true
    intrinsic_dimension: true
    neighborhoods: true
    training_dynamics: true
    forgetting: true
    transitions: true
    probing: true
    parameter_drift: true
    performance_relationships: true
    svcca: false
    pwcca: false
    attribution: false
    influence: false
    topology: false
  visualization:
    profile: paper            # paper | presentation
    static_format: svg
    optional_formats:
      pdf: false
      png: false
    palette: okabe_ito
    background: white
    grid: false
    spines:
      top: false
      right: false
    legend:
      smart_position: true
      outside_when_dense: true
    rasterization:
      dense_scatter: true
      dpi: 600
```

`include_all_examples=false` is reserved and rejected by behavior (Part 1 never
scientifically samples). `include_all_tokens=false` retains the model positions
on which gold labels are defined. Layer zero is the embedding output and the last
index is the final Transformer layer.

## Identity, pooling, and size checks

SHA-256-derived IDs encode document, window, token position/ID and offsets.
Observation metadata includes token and word identity, character offsets, gold
and predicted BIO labels/types, confidence, entropy, gold margin and correctness.
Reusable document text is not copied. `pool_groups` implements first-subword,
mean and max word pooling; `pool_spans` provides the same mechanics for gold or
predicted character-offset spans. The existing BIO span reconstruction remains
the authoritative way to obtain spans.

Before any chunk is written, the conservative estimate is:

```text
observations * hidden_size * stored_layers * bytes_per_element
  * (1 + metadata_overhead_fraction)
```

The check logs expected bytes, current snapshot usage, projected usage, free
filesystem bytes and the safety margin, and raises before extraction if the new
snapshot plus margin does not fit.

## Recovery and cleanup guarantees

Successful training records `awaiting_postprocessing`; Part 1 does **not** claim
that future analyses exist and does not automatically clean snapshots. Parts 2/3
will register mandatory analyses and required outputs, then either call
`finalize_run` or use:

```bash
lab explicability finalize RUN_DIRECTORY \
  --completed-analysis NAME \
  --mandatory-analysis NAME \
  --required-output tables/result.parquet
```

Finalization refuses failed training, pending analyses, invalid manifests,
incomplete snapshots and missing/unreadable required Parquet files. Only after it
writes the final manifest and `SUCCESS` may `cleanup_policy=on_success` remove
`_snapshots`. Failures preserve snapshots. Cleanup failure is recorded without
invalidating permanent outputs or `SUCCESS`; `never` always retains raw data.

Extraction/write durations and bytes are recorded per shard and snapshot, and
the manifest records total temporary/permanent sizes and post-processing time.

## Part 2 analytical processing

After successful training, run:

```bash
lab explicability analyze RUN_DIRECTORY
```

The command reads the Part 1 configuration and independently executes each enabled
analysis. It writes ZSTD Parquet tables for shared temporal projection,
trajectories, centroid dynamics, CKA, geometry, intrinsic dimension, compact kNN
evolution, dataset cartography, forgetting, transitions, frozen linear probes,
parameter drift, representation shifts, performance relationships, hard examples,
and layer specialization. SVCCA and PWCCA are implemented but disabled by default.

Shared PCA is fitted once over all aligned checkpoints. Fixed UMAP fits one common
mapping, and AlignedUMAP receives explicit identity relations; both require the
optional `umap-learn` package. Independent per-checkpoint nonlinear maps are never
interpreted as motion. Sampling is off by default. If `sample_size` is set, the
pipeline uses an explicit seeded, label-stratified sample and records its policy,
seed, and size in every output.

Exact nearest neighbors are calculated in query blocks and only top-k identities
and summaries are retained. Linear CKA has a sufficient-statistic streaming
implementation. Analyses process one layer at a time and release its representations
before proceeding. Covariance spectra, rather than all-pairs matrices, are retained.

Frozen probes use ID-hashed train/validation/test assignments shared across every
layer and checkpoint. Their scores measure decodability, not causal use. Optional
attribution and TracIn-style influence functions require explicitly selected cases;
the all-snapshot command refuses to run them corpus-wide. Topology and UMAP remain
dependency-gated. See `references/` for methodological sources and caveats.

`analyze` updates the manifest to `analyses_complete` but deliberately does not
write `SUCCESS` or remove snapshots: Part 3 still needs to validate figures and the
report before invoking finalization.

## Part 3 publication outputs

`lab explicability plot RUN` consumes only permanent Parquet tables and generates
canonical SVG figures, the corpus fingerprint table, figure manifest, temporal
animation, and `report/index.html`. `--profile presentation`, `--analysis NAME`,
`--figure FIGURE_ID`, and `--no-animation` support deterministic regeneration.
Selective runs remain partial and cannot trigger cleanup. A complete plotting pass
validates every generated figure and its source table, then marks visualization
complete; `lab explicability finalize RUN` validates again before writing `SUCCESS`
and applying the cleanup policy. See `SCIENTIFIC_VISUALIZATION_GUIDE.md`.
